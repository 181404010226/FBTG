import torch
from Paper_global_vars import global_vars
from torch import optim
import torch.nn.functional as F
import os
import gc
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from Paper_Tree import *
from Paper_DataSetCIFAR import create_train_loader, create_valid_loader

torch.set_float32_matmul_precision('high')

if __name__ == "__main__":
    dist.init_process_group(backend='nccl')

    loader_train = create_train_loader(global_vars.dataset, distributed=True)
    valid_data = create_valid_loader(global_vars.dataset, distributed=True)

    # Set the device
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Initialize model
    model = globals()[global_vars.model_name]().to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = getattr(optim, global_vars.optimizer)(model.parameters(), weight_decay=0.001)

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
                outputs = model(data)
                
                is_tree = hasattr(model.module, 'isTree') if isinstance(model, DDP) else model.isTree

                if is_tree:
                    normalized_probs = outputs / outputs.sum(dim=1, keepdim=True)
                    batch_loss = torch.sum(-target * torch.log(normalized_probs + 1e-7), dim=-1).mean()
                else:
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

            if dist.get_rank() == 0:
                batch_losses.append(batch_loss.item())
                if (batch_idx + 1) % 10 == 0:
                    avg_loss = sum(batch_losses[-10:]) / len(batch_losses[-10:])
                    print(f"Batches {batch_idx-8}-{batch_idx+1}/{len(loader_train)}: Avg Loss: {avg_loss:.4f}")
                    print(f"Learning rate: {scheduler.get_last_lr()[0]:.9f}")
                    batch_losses = []

        scheduler.step()
        
        train_accuracy = train_correct / train_total if train_total > 0 else 0
        print(f"Epoch {epoch+1}/{global_vars.num_epochs} - Train Accuracy: {train_accuracy:.4f}({train_correct}/{train_total})")

        # Validation phase
        model.eval()
        total_correct = torch.zeros(1).to(device)
        total_samples = torch.zeros(1).to(device)

        with torch.no_grad():
            for data, target in valid_data:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                predicted_labels = outputs.argmax(dim=1)
                total_correct += (predicted_labels == target).sum()
                total_samples += len(target)

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

        
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Rank {dist.get_rank()} reached the barrier.")
        dist.barrier()
        print(f"Rank {dist.get_rank()} passed the barrier.")


    # 清理
    dist.destroy_process_group()
