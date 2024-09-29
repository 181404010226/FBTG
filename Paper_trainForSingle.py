import torch
from Paper_global_vars import global_vars
from torch import optim
import torch.nn.functional as F
import os
import gc
import timm
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from Paper_Tree import *
from torch.utils.data.distributed import DistributedSampler
from Paper_DataSetCIFAR import create_train_loader, create_valid_loader
import torch
import torch.optim as optim
import os
from Paper_global_vars import global_vars
from Paper_Tree import *
from Paper_DataSetCIFAR import create_train_loader, create_valid_loader  
from convmixer import FilteredConvMixer

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loader_train = create_train_loader(global_vars.dataset, distributed=False)
    valid_data = create_valid_loader(global_vars.dataset, distributed=False)

    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    print(f"Using device: {device}")

    # 初始化模型
    #model_class = globals()[global_vars.model_name]
    #model = model_class().to(device)
    # Initialize the model
    filter_params = {
        'input_channels': 3,
        'hidden_dim': 64,
        'kernel_size': 3
    }
    convmixer_params = {
        'dim': 256,
        'depth': 8,
        'kernel_size': 5,
        'patch_size': 1,
        'n_classes': 10
    }
    model = FilteredConvMixer(filter_params, convmixer_params).to(device)
    
    optimizer = getattr(optim, global_vars.optimizer)(
        model.parameters(), 
        lr=global_vars.max_lr, 
        weight_decay=0.001
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=global_vars.max_lr,
        total_steps=global_vars.num_epochs,
        pct_start=0.3,
        anneal_strategy='cos',
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
    )

    best_models = []
    best_accuracies = []

    # Initialize GradScaler
    scaler = GradScaler()

    for epoch in range(global_vars.num_epochs):
        # Training phase
        model.train()
        batch_losses = []
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(loader_train):
            data, target = data.to(device), target.to(device)

            with autocast():
                outputs, filtered_x = model(data, k=50)
                
                if hasattr(model, 'isTree') and model.isTree:
                    if (epoch==0 and batch_idx==0):
                        print("SequentialDecisionTree")
                    normalized_probs = outputs / outputs.sum(dim=1, keepdim=True)
                    batch_loss = torch.sum(-target * torch.log(normalized_probs + 1e-7), dim=-1).mean()
                else:
                    if (epoch==0 and batch_idx==0):
                        print("single model")
                    batch_loss = torch.sum(-target * F.log_softmax(outputs, dim=-1), dim=-1).mean()
              
                predicted_labels = outputs.argmax(dim=1)
                train_correct += (predicted_labels == target.argmax(dim=1)).sum().item()
                train_total += len(target)

            scaler.scale(batch_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            batch_losses.append(batch_loss.item())
            if (batch_idx + 1) % 10 == 0:
                avg_loss = sum(batch_losses[-10:]) / len(batch_losses[-10:])
                print(f"Batches {batch_idx-8}-{batch_idx+1}/{len(loader_train)}: Avg Loss: {avg_loss:.4f}")
                print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
                batch_losses = []

        scheduler.step()
        
        train_accuracy = train_correct / train_total if train_total > 0 else 0
        print(f"Epoch {epoch+1}/{global_vars.num_epochs} - Train Accuracy: {train_accuracy:.4f}({train_correct}/{train_total})")

        # Validation phase
        model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_data):
                data, target = data.to(device), target.to(device)
                outputs, filtered_x = model(data, k=50)
                predicted_labels = outputs.argmax(dim=1)
                total_correct += (predicted_labels == target).sum().item()
                total_samples += len(target)
        # 保存过滤后的图像和原始图像
        filtered_x.save(f"View/filtered_x_{epoch+1}.png")
        data.save(f"View/original_x_{epoch+1}.png")

        accuracy = total_correct / total_samples
        print(f"Test Accuracy: {accuracy:.4f}({total_correct}/{total_samples})")

  
        # 保存前十个最佳模型
        if len(best_models) < 10 or accuracy > min(best_accuracies):
            # 保存模型和优化器
            checkpoint = {
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'accuracy': accuracy,
                'epoch': epoch + 1
            }
            
            if len(best_models) == 10:
                # 移除准确率最低的模型
                min_acc_index = best_accuracies.index(min(best_accuracies))
                min_acc = best_accuracies[min_acc_index]
                
                # 删除文件系统中的模型文件
                for filename in os.listdir(global_vars.save_path):
                    if filename.startswith("checkpoint_") and filename.endswith(f"acc_{min_acc:.4f}.pth"):
                        os.remove(os.path.join(global_vars.save_path, filename))
                        print(f"Removed file: {filename}")
                
                best_models.pop(min_acc_index)
                best_accuracies.pop(min_acc_index)
            
            best_models.append(checkpoint)
            best_accuracies.append(accuracy)
            
            # 按准确率降序排序
            best_models, best_accuracies = zip(*sorted(zip(best_models, best_accuracies), 
                                                    key=lambda x: x[1], reverse=True))
            best_models = list(best_models)
            best_accuracies = list(best_accuracies)
            
            # 保存模型和优化器
            save_path_checkpoint = os.path.join(global_vars.save_path, f"checkpoint_epoch_{epoch+1}_acc_{accuracy:.4f}.pth")
            os.makedirs(global_vars.save_path, exist_ok=True)
            torch.save(checkpoint, save_path_checkpoint)
            print(f"Saved checkpoint to {save_path_checkpoint}")

            # 训练结束后，打印最佳模型信息
            print("\nTop 10 Best Models:")
            for i, checkpoint in enumerate(best_models, 1):
                print(f"{i}. Epoch: {checkpoint['epoch']}, Accuracy: {checkpoint['accuracy']:.4f}")

        