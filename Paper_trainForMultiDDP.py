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
from Paper_Tree import SequentialDecisionTree, SequentialDecisionTreeCIFAR100
from torch.utils.data.distributed import DistributedSampler
from Paper_DataSetCIFAR import create_train_loader, create_valid_loader
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from convmixer import ConvMixer
import psutil 


if __name__ == "__main__":

    dist.init_process_group(backend='nccl')

    loader_train = create_train_loader(distributed=True)
    valid_data = create_valid_loader(distributed=True)
    # Set the device
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # 检查可用的GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    # 使用所有可用的GPU
    print(f"Using device: {device}")

    root = os.path.join(os.path.dirname(__file__), "CIFAR10RawData")
    # 检查/hy-tmp是否存在
    if os.path.exists("/hy-tmp"):
        save_path = os.path.join("/hy-tmp/best_models")
    else:
        save_path = os.path.join("/root/autodl-tmp")

    # 初始化模型并移至GPU
    # model = SequentialDecisionTree().to(device)
    # model = ResNet14({'in_channels': 3, 'out_channels': 10, 'activation': 'CosLU'}).to(device)
    # model = ConvMixer(dim=256, depth=8, kernel_size=5, patch_size=1, n_classes=10).to(device)
    # model = rdnet_tiny(num_classes=1000).to(device)  # Assuming 10 classes for CIFAR-10
    # model = MaxxVit(model_cfgs['astroformer_0'], num_classes=10)
    # model = SequentialDecisionTreeCIFAR100().to(device)
    model = SequentialDecisionTree().to(device)
    # 应用 torch.compile()
    model = torch.jit.script(model)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # 在将模型移至GPU之前，先将模型参数转换为同步批归一化
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = optim.AdamW(model.parameters(), weight_decay=0.001)

    scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=0.00025,
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
        # 训练阶段
        model.train()

        batch_losses = []
        
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(loader_train):
            data, target = data.to(device), target.to(device)

            with autocast():
                outputs = model(data)
                
                # 使用 module 来访问原始模型的方法
                if isinstance(model, DDP):
                    is_tree = model.module.isTree
                else:
                    is_tree = model.isTree

                # 新增判断
                if is_tree:
                    if (epoch==0 and batch_idx==0):
                        print("SequentialDecisionTree")
                    normalized_probs = outputs / outputs.sum(dim=1, keepdim=True)
                    batch_loss = torch.sum(-target * torch.log(normalized_probs + 1e-7), dim=-1).mean()
                else:
                    if (epoch==0 and batch_idx==0):
                        print("single model")
                    batch_loss = torch.sum(-target * F.log_softmax(outputs, dim=-1), dim=-1).mean()
              

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

            if dist.get_rank() == 0:
                batch_losses.append(batch_loss.item())
                # 打印每10个batch的平均loss和学习率
                if (batch_idx + 1) % 10 == 0:
                    avg_loss = sum(batch_losses[-10:]) / len(batch_losses[-10:])
                    print(f"Batches {batch_idx-8}-{batch_idx+1}/{len(loader_train)}: Avg Loss: {avg_loss:.4f}")
                    print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
                    batch_losses = []  # Reset the list after printing


        scheduler.step()
        
        # 输出训练阶段正确率
        train_accuracy = train_correct / train_total if train_total > 0 else 0
        print(f"Epoch {epoch+1}/{global_vars.num_epochs} - Train Accuracy: {train_accuracy:.4f}({train_correct}/{train_total})")

        # 在所有进程上进行验证
        model.eval()
        total_correct = torch.zeros(1).to(device)
        total_samples = torch.zeros(1).to(device)

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_data):
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                predicted_labels = outputs.argmax(dim=1)
                total_correct += (predicted_labels == target).sum()
                total_samples += len(target)

        # 在所有进程间同步结果
        dist.all_reduce(total_correct)
        dist.all_reduce(total_samples)

        if dist.get_rank() == 0:
            accuracy = total_correct.item() / total_samples.item()
            print(f"Test Accuracy: {accuracy:.4f}({total_correct.item()}/{total_samples.item()})")

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
                    for filename in os.listdir(save_path):
                        if filename.startswith("checkpoint_") and filename.endswith(f"acc_{min_acc:.4f}.pth"):
                            os.remove(os.path.join(save_path, filename))
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
                save_path_checkpoint = os.path.join(save_path, f"checkpoint_epoch_{epoch+1}_acc_{accuracy:.4f}.pth")
                os.makedirs(save_path, exist_ok=True)
                torch.save(checkpoint, save_path_checkpoint)
                print(f"Saved checkpoint to {save_path_checkpoint}")

                # 训练结束后，打印最佳模型信息
                print("\nTop 10 Best Models:")
                for i, checkpoint in enumerate(best_models, 1):
                    print(f"{i}. Epoch: {checkpoint['epoch']}, Accuracy: {checkpoint['accuracy']:.4f}")

        
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Rank {dist.get_rank()} reached the barrier.")
        dist.barrier()
        print(f"Rank {dist.get_rank()} passed the barrier.")


    # 清理
    dist.destroy_process_group()
