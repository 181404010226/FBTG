import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from Paper_global_vars import global_vars
from Paper_DataSetCIFAR import data_config, create_loader, _get_dataset, get_mixup_fn, collate_mixup_fn
from convmixer import ConvMixer
from functools import partial
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NodeDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=data_config['mean'], 
                              std=data_config['std'])
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        if not isinstance(img, torch.Tensor):
            img = self.transform(img)
        return img, self.targets[idx]
    
def create_node_dataset(dataset='cifar10', node_idx=0, train=True):
    """
    Create a modified dataset for a specific node in the decision tree.
    
    Args:
        dataset (str): 'cifar10' or 'cifar100'
        node_idx (int): Index of the node (0-7)
        train (bool): Whether to use training or test set
    """
    # Node configurations
    node_configs = [
        {'classes': [[0,1,8,9], [2,3,4,5,6,7]], 'num_classes': 2},
        {'classes': [[0,8], [1,9]], 'num_classes': 2},
        {'classes': [[0], [8]], 'num_classes': 2},
        {'classes': [[1], [9]], 'num_classes': 2},
        {'classes': [[2,6], [3,5], [4,7]], 'num_classes': 3},
        {'classes': [[2], [6]], 'num_classes': 2},
        {'classes': [[3], [5]], 'num_classes': 2},
        {'classes': [[4], [7]], 'num_classes': 2}
    ]

    # Get the original dataset
    original_dataset = _get_dataset(dataset, train)
    
    # Get configuration for the specified node
    node_config = node_configs[node_idx]
    valid_classes = sum(node_config['classes'], [])
    
    # Create new targets and filter indices
    new_targets = []
    valid_indices = []
    
    for idx, (_, target) in enumerate(original_dataset):
        if target in valid_classes:
            # Find which group this class belongs to
            for group_idx, group in enumerate(node_config['classes']):
                if target in group:
                    new_targets.append(group_idx)  # 直接使用组索引作为标签
                    valid_indices.append(idx)
                    break
    
    # Create a filtered dataset
    filtered_data = []
    filtered_targets = []
    
    for idx in valid_indices:
        img, _ = original_dataset[idx]
        filtered_data.append(img)
        filtered_targets.append(new_targets[valid_indices.index(idx)])
    
    # 将标签转换为张量
    targets_tensor = torch.tensor(filtered_targets, dtype=torch.long)
    
    return filtered_data, targets_tensor

def create_node_loader(node_idx, dataset='cifar10', train=True, distributed=False):
    """
    Create a data loader for a specific node
    """
    data, targets = create_node_dataset(dataset, node_idx, train)
    node_dataset = NodeDataset(data, targets)

    if train:
        mixup_fn = get_mixup_fn(num_classes=2 if node_idx !=4 else 3)  # Adjust num_classes based on node
        collate_fn = partial(collate_mixup_fn, mixup_fn=mixup_fn)
        
        loader = create_loader(
            node_dataset,
            input_size=data_config['input_size'],
            batch_size=global_vars.train_batch_size,
            is_training=True,
            use_prefetcher=False,
            no_aug=False,
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            scale=(0.75, 1.0),
            ratio=(3./4., 4./3.),
            hflip=0.5,
            vflip=0.,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=1,  # Reduced from 8 to 1 to avoid potential multiprocessing issues
            distributed=distributed,
            collate_fn=collate_fn,  # Apply Mixup
            pin_memory=True
        )
    else:
        loader = create_loader(
            node_dataset,
            input_size=data_config['input_size'],
            batch_size=global_vars.test_batch_size,
            is_training=False,
            use_prefetcher=False,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=1,  # Reduced from 8 to 1
            distributed=distributed,
            pin_memory=True
        )
    
    return loader

def train_single_node(node_idx, num_epochs=300):
    # Create model for this node
    num_classes = 2 if node_idx != 4 else 3
    model = ConvMixer(dim=256, depth=8, kernel_size=5, patch_size=1, n_classes=num_classes).to(device)
    
    # Create data loaders for this node
    train_loader = create_node_loader(node_idx, train=True)
    valid_loader = create_node_loader(node_idx, train=False)
    
    optimizer = optim.AdamW(model.parameters(), lr=global_vars.max_lr, weight_decay=0.001)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=global_vars.max_lr,
        total_steps=num_epochs,
        pct_start=0.3,
        anneal_strategy='cos',
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
    )
    
    scaler = GradScaler()
    best_acc = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            with autocast():
                outputs = model(data)
                batch_loss = torch.sum(-target * F.log_softmax(outputs, dim=-1), dim=-1).mean()
            
            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Output accuracy every 100 batches
            if (batch_idx + 1) % 100 == 0:
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for val_data, val_target in valid_loader:
                        val_data, val_target = val_data.to(device), val_target.to(device)
                        val_outputs = model(val_data)
                        _, predicted = val_outputs.max(1)
                        correct += (predicted == val_target).sum().item()
                        total += val_target.size(0)
                accuracy = correct / total
                print(f'Node {node_idx}, Epoch {epoch+1}, Batch {batch_idx+1}: Accuracy = {accuracy:.4f}')
                model.train()
        
        scheduler.step()
        
        # Validation phase at the end of each epoch
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = outputs.max(1)
                _, target_class = target.max(1)
                correct += (predicted == target_class).sum().item()
                total += target.size(0)
        
        accuracy = correct / total
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'accuracy': accuracy,
                'epoch': epoch + 1
            }, f'node_{node_idx}_best.pth')
        
        print(f'Node {node_idx}, Epoch {epoch+1}: Best Accuracy = {best_acc:.4f}')
def train_all_nodes():
    for node_idx in range(8):
        print(f"\nTraining Node {node_idx}")
        print("=" * 50)
        train_single_node(node_idx)

if __name__ == "__main__":
    train_all_nodes()
