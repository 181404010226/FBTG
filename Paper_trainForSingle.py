import torch
from Paper_global_vars import global_vars
from Paper_DataSet import valid_data,loader_train
from torch import optim
import torch.nn.functional as F
from astroformer import MaxxVit, model_cfgs
import os
import numpy as np
import random
import csv
import timm
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


if __name__ == "__main__":
    dist.init_process_group(backend='nccl')
    
    # Set the device
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

        # 检查可用的GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    # 使用所有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    root = os.path.join(os.path.dirname(__file__), "CIFAR10RawData")
    save_path = os.path.join("/hy-tmp/best_models")

    train_sampler = DistributedSampler(loader_train.dataset)

    # 初始化模型并移至GPU
    # model = SequentialDecisionTree().to(device)
    # model = ResNet14({'in_channels': 3, 'out_channels': 10, 'activation': 'CosLU'}).to(device)
    # model = ConvMixer(dim=256, depth=8, kernel_size=5, patch_size=1, n_classes=10).to(device)
    # model = rdnet_tiny(num_classes=1000).to(device)  # Assuming 10 classes for CIFAR-10
    model = MaxxVit(model_cfgs['astroformer_3'], num_classes=10).to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = optim.AdamW(model.parameters(), weight_decay=0.001)

    scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=0.0001,
                total_steps=global_vars.num_epochs,
                pct_start=0.3,
                anneal_strategy='cos',
                cycle_momentum=True,
                base_momentum=0.85,
                max_momentum=0.95,
            )

    best_models = []
    best_accuracies = []

    # 初始化 GradScaler
    scaler = GradScaler()

    for epoch in range(global_vars.num_epochs):
        train_sampler.set_epoch(epoch)
        # 训练阶段
        model.train()

        batch_losses = []
        
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(loader_train):
            data, target = data.to(device), target.to(device)
            
            with autocast():
                outputs = model(data)
                batch_loss = torch.sum(-target *F.log_softmax(outputs, dim=-1), dim=-1).mean()

                if dist.get_rank() == 0:
                    # 统计训练正确率（如果需要）
                    predicted_labels = outputs.argmax(dim=1)
                    train_correct += (predicted_labels == target.argmax(dim=1)).sum().item()
                    train_total += len(target)

            # 使用 scaler 来缩放损失并执行反向传播
            scaler.scale(batch_loss).backward()
            # 添加梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            batch_losses.append(batch_loss.item())

            if dist.get_rank() == 0:
                # 打印每10个batch的平均loss和学习率
                if (batch_idx + 1) % 10 == 0:
                    avg_loss = sum(batch_losses[-10:]) / len(batch_losses[-10:])
                    print(f"Batches {batch_idx-8}-{batch_idx+1}/{len(loader_train)}: Avg Loss: {avg_loss:.4f}")
                    print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
                    batch_losses = []  # Reset the list after printing

        scheduler.step()

        if dist.get_rank() == 0:
            # 输出训练阶段正确率
            train_accuracy = train_correct / train_total if train_total > 0 else 0
            print(f"Epoch {epoch+1}/{global_vars.num_epochs} - Train Accuracy: {train_accuracy:.4f}")

        if dist.get_rank() == 0:
            # 测试代码
            model.eval()
            total_correct = 0
            total_samples = 0

            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(valid_data):
                    data, target = data.to(device), target.to(device)
                    outputs = model(data)
                    

                    # 获取预测标签（在对数空间中，最大值对应原空间的最大概率）
                    predicted_labels = outputs.argmax(dim=1)
                    
                    # 计算正确预测的数量
                    total_correct += (predicted_labels == target).sum().item()
                    total_samples += len(target)

                accuracy = total_correct / total_samples
                print(f"Test Accuracy: {accuracy:.4f}")

                # 保存前十个最佳模型
                if len(best_models) < 10 or accuracy > min(best_accuracies):
                    # 保存模型和优化器
                    if isinstance(model, torch.nn.DataParallel):
                        model_state = model.module.state_dict()
                    else:
                        model_state = model.state_dict()
                        
                    if len(best_models) == 10:
                        # 移除准确率最低的模型
                        min_acc_index = best_accuracies.index(min(best_accuracies))
                        min_acc = best_accuracies[min_acc_index]
                        
                        # 删除文件系统中的模型文件
                        for filename in os.listdir(save_path):
                            if filename.endswith(f"acc_{min_acc:.4f}.pth"):
                                os.remove(os.path.join(save_path, filename))
                                print(f"Removed file: {filename}")
                        
                        best_models.pop(min_acc_index)
                        best_accuracies.pop(min_acc_index)
                    
                    best_models.append(model_state)
                    best_accuracies.append(accuracy)
                    
                    # 按准确率降序排序
                    best_models, best_accuracies = zip(*sorted(zip(best_models, best_accuracies), 
                                                            key=lambda x: x[1], reverse=True))
                    best_models = list(best_models)
                    best_accuracies = list(best_accuracies)
                    
                    # 保存模型和优化器
                    save_path_model = os.path.join(save_path, f"model_epoch_{epoch+1}_acc_{accuracy:.4f}.pth")
                    save_path_optimizer = os.path.join(save_path, f"optimizer_epoch_{epoch+1}_acc_{accuracy:.4f}.pth")
                    os.makedirs(save_path, exist_ok=True)
                    torch.save(model_state, save_path_model)
                    torch.save(optimizer.state_dict(), save_path_optimizer)
                    print(f"Saved model to {save_path_model}")
                    print(f"Saved optimizer to {save_path_optimizer}")


            # 训练结束后，打印最佳模型信息
            print("\nTop 10 Best Models:")
            for i, (_, acc) in enumerate(zip(best_models, best_accuracies), 1):
                print(f"{i}. Accuracy: {acc:.4f}")

        dist.barrier()
        
