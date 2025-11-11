import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from convmixer import ConvMixer
from Paper_Tree import SequentialDecisionTree
from Paper_global_vars import global_vars
from Paper_DataSetCIFAR import cifar10_mean, cifar10_std


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_loaders(batch_size):
    data_root = os.path.join(os.path.dirname(__file__), 'CIFAR10RawData')
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])

    train_set = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader


def accuracy(outputs, targets):
    preds = outputs.argmax(dim=1)
    return (preds == targets).float().mean().item()


def make_node_targets(node, labels):
    # labels: (batch,)
    targets = torch.zeros_like(labels)
    for j, group in enumerate(node.judge):
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for cls in group:
            mask |= (labels == cls)
        targets[mask] = j
    return targets


def evaluate_direct(model, loader, device):
    model.eval()
    total_acc = 0.0
    total_cnt = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            acc = accuracy(logits, labels)
            total_acc += acc * images.size(0)
            total_cnt += images.size(0)
    return total_acc / max(1, total_cnt)


def evaluate_tree(tree, loader, device):
    tree.eval()
    total_acc = 0.0
    total_cnt = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            probs = tree(images)  # (batch, 10) in [0,1]
            preds = probs.argmax(dim=1)
            acc = (preds == labels).float().mean().item()
            total_acc += acc * images.size(0)
            total_cnt += images.size(0)
    return total_acc / max(1, total_cnt)


def train_direct(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def _normalize_to_logits(final_outputs, eps=1e-9):
    # 将各类的乘积权重正规化为概率，再转为 logits（对数概率作为 logits）
    probs = final_outputs.clamp(min=eps)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=eps)
    logits = torch.log(probs)
    return logits


def train_tree_global_loss(tree, loader, optimizers, criterion, device):
    """
    训练策略：
    1) 在推理模式（无梯度）下对整棵树做一次前向，得到 batch 的全局输出与全局 loss（用于日志）。
    2) 随后逐节点训练：仅为当前节点打开梯度并重新前向，其他节点保持推理模式；
       使用相同的全局目标函数（基于整棵树的最终输出）进行反向传播与更新。

    这样每个局部模型都相对于同一个全局目标被优化，同时避免构建一张巨大的计算图。
    """
    total_global_loss = 0.0
    per_node_loss_sum = [0.0 for _ in range(len(tree.nodes))]

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        # Step 1: 全局前向（推理模式），记录全局 loss
        tree.set_training_mode('record')
        tree.eval()
        with torch.no_grad():
            final_outputs = tree(images, batch_id=batch_idx)  # (B, 10) 乘积权重
            global_logits = _normalize_to_logits(final_outputs)
            global_loss = criterion(global_logits, labels)
        total_global_loss += global_loss.item() * images.size(0)

        # Step 2: 逐节点反向，每次只训练一个节点，其余节点推理
        for i, _ in enumerate(tree.nodes):
            optim = optimizers[i]
            optim.zero_grad()

            # 仅当前节点训练，其余节点推理；同时将当前节点设为 train，其余设为 eval 以避免 dropout/BN 随机性
            tree.set_training_mode('pipeline', node_idx=i)
            for j, node in enumerate(tree.nodes):
                if j == i:
                    node.train()
                else:
                    node.eval()

            final_outputs_i = tree(images, batch_id=batch_idx)  # 重新前向，只有第 i 个节点带梯度；其他节点使用缓存
            logits_i = _normalize_to_logits(final_outputs_i)
            loss_i = criterion(logits_i, labels)
            loss_i.backward()
            optim.step()

            per_node_loss_sum[i] += loss_i.item() * images.size(0)

        # 清理缓存（虽未在 pipeline 使用缓存，这里保持接口一致）
        tree.clear_cache()

    # 返回一个全局 loss（按样本数平均）以及各节点 loss（用于观察）
    avg_global_loss = total_global_loss / len(loader.dataset)
    avg_node_losses = [ls / len(loader.dataset) for ls in per_node_loss_sum]
    return avg_global_loss, avg_node_losses


def main():
    device = get_device()
    train_loader, test_loader = get_loaders(global_vars.train_batch_size)

    # Direct model
    direct_model = ConvMixer(dim=256, depth=8, kernel_size=5, patch_size=2, n_classes=10).to(device)
    direct_opt = optim.AdamW(direct_model.parameters(), lr=global_vars.max_lr)
    ce = nn.CrossEntropyLoss()

    # Tree model
    tree = SequentialDecisionTree().to(device)
    tree_opts = [optim.AdamW(node.parameters(), lr=global_vars.max_lr) for node in tree.nodes]

    epochs = min(25, global_vars.num_epochs)

    print('Starting comparison training...')
    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch}/{epochs}')
        t0 = time.time()
        direct_loss = train_direct(direct_model, train_loader, direct_opt, ce, device)
        t1 = time.time()
        direct_acc = evaluate_direct(direct_model, test_loader, device)
        print(f'  Direct:   loss={direct_loss:.4f} acc={direct_acc:.4f} time={(t1-t0):.2f}s')
       
    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch}/{epochs}')
        t1 = time.time()
        global_loss, node_losses = train_tree_global_loss(tree, train_loader, tree_opts, ce, device)
        t2 = time.time()
        tree_acc = evaluate_tree(tree, test_loader, device)
        print(f'  Tree:     global_loss={global_loss:.4f} node_losses={[round(l,4) for l in node_losses]} acc={tree_acc:.4f} time={(t2-t1):.2f}s')
    
    print('Done.')


if __name__ == '__main__':
    main()