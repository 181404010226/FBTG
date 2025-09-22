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

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loader_train = create_train_loader(global_vars.dataset, distributed=False)
    valid_data = create_valid_loader(global_vars.dataset, distributed=False)

    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    print(f"Using device: {device}")

    # 初始化模型
    model_class = globals()[global_vars.model_name]
    model = model_class().to(device)
    # Initialize the model
    # model = ConvMixerWithNeuronBundles(128,4, 32, 5, 1, 10).to(device)
    
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
        div_factor=25,
    )

    best_models = []
    best_accuracies = []

    # Initialize GradScaler
    scaler = GradScaler()

    # 检查是否是树模型，如果是则使用流水线训练
    is_tree_model = hasattr(model, 'isTree') and model.isTree
    num_nodes = len(model.nodes) if is_tree_model else 1

    for epoch in range(global_vars.num_epochs):
        if is_tree_model:
            print(f"\n=== Epoch {epoch+1}/{global_vars.num_epochs} - Pipeline Training ===")
            
            # 第一阶段：记录所有节点输出
            print("Phase 1: Recording all node outputs...")
            model.set_training_mode('record')
            model.eval()
            
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(loader_train):
                    data, target = data.to(device), target.to(device)
                    _ = model(data, batch_id=batch_idx)
                    
                    if (batch_idx + 1) % 100 == 0:
                        print(f"Recorded batch {batch_idx+1}/{len(loader_train)}")
            
            print("Phase 1 completed. Starting pipeline training...")
            
            # 第二阶段：流水线式训练每个节点
            for node_idx in range(num_nodes):
                print(f"\nTraining node {node_idx+1}/{num_nodes}...")
                model.set_training_mode('pipeline', node_idx)
                model.train()
                
                batch_losses = []
                train_correct = 0
                train_total = 0
                
                for batch_idx, (data, target) in enumerate(loader_train):
                    data, target = data.to(device), target.to(device)

                    with autocast():
                        outputs = model(data, batch_id=batch_idx)
                        normalized_probs = outputs / outputs.sum(dim=1, keepdim=True)
                        batch_loss = torch.sum(-target * torch.log(normalized_probs + 1e-7), dim=-1).mean()
                      
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
                    if (batch_idx + 1) % 50 == 0:
                        avg_loss = sum(batch_losses[-50:]) / len(batch_losses[-50:])
                        print(f"  Node {node_idx+1} - Batches {batch_idx-48}-{batch_idx+1}: Avg Loss: {avg_loss:.4f}")
                
                node_train_accuracy = train_correct / train_total if train_total > 0 else 0
                print(f"  Node {node_idx+1} - Train Accuracy: {node_train_accuracy:.4f}")
            
            # 清空缓存以释放内存
            model.clear_cache()
            
        else:
            # 原有的单模型训练逻辑
            print(f"\n=== Epoch {epoch+1}/{global_vars.num_epochs} - Single Model Training ===")
            # Training phase
            model.train()
            batch_losses = []
            train_correct = 0
            train_total = 0

            for batch_idx, (data, target) in enumerate(loader_train):
                data, target = data.to(device), target.to(device)

                with autocast():
                    outputs = model(data)
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
        
        # 验证阶段保持不变
        if is_tree_model:
            model.set_training_mode('normal')  # 验证时使用正常模式
        
        train_accuracy = train_correct / train_total if train_total > 0 else 0
        print(f"Epoch {epoch+1}/{global_vars.num_epochs} - Overall Train Accuracy: {train_accuracy:.4f}({train_correct}/{train_total})")

        # Validation phase
        model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_data):
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                predicted_labels = outputs.argmax(dim=1)
                total_correct += (predicted_labels == target).sum().item()
                total_samples += len(target)

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

        