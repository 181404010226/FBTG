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
from torch.utils.data.distributed import DistributedSampler
from Paper_DataSetCIFAR import create_train_loader, create_valid_loader
import torch
import torch.optim as optim
import os
from Paper_global_vars import global_vars
from Paper_DataSetCIFAR import create_train_loader, create_valid_loader  
from Paper_NeuronBundle import create_convmixer
from timm.models.swin_transformer import SwinTransformerBlock
from torch.nn import TransformerEncoderLayer, LayerNorm

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loader_train = create_train_loader(global_vars.dataset, distributed=False)
    valid_data = create_valid_loader(global_vars.dataset, distributed=False)

    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    print(f"Using device: {device}")

    # Initialize the model
    model = create_convmixer(64,4, 32, 5, 4, 1000).to(device)

     # Do a dummy forward pass to initialize all layers
    with torch.no_grad():
        dummy_input = torch.randn(1, *global_vars.input_size).to(device)
        _ = model(dummy_input)

    # 将参数分组，为CNN和Transformer设置不同的学习率
    transformer_params = []
    cnn_params = []
    other_params = []


    for name, param in model.named_parameters():
        if any(isinstance(m, TransformerEncoderLayer) for m in [module for module in model.modules()]):
            if any(n in name for n in ['self_attn', 'linear', 'norm']):
                transformer_params.append(param)
            else:
                other_params.append(param)
        else:
            if 'conv' in name or 'batchnorm' in name:
                cnn_params.append(param)
            else:
                other_params.append(param)

    # for name, param in model.named_parameters():
    #     if any(isinstance(m, SwinTransformerBlock) for m in [module for module in model.modules()]):
    #         if any(n in name for n in ['attn', 'mlp', 'norm']):
    #             transformer_params.append(param)
    #         else:
    #             other_params.append(param)
    #     else:
    #         if 'conv' in name or 'batchnorm' in name:
    #             cnn_params.append(param)
    #         else:
    #             other_params.append(param)

    # Print parameter counts for each group
    print(f"Number of transformer parameters: {sum(p.numel() for p in transformer_params):,}")
    print(f"Number of CNN parameters: {sum(p.numel() for p in cnn_params):,}")
    print(f"Number of other parameters: {sum(p.numel() for p in other_params):,}")

    # 设置不同的学习率
    param_groups = [
        {'params': transformer_params, 'lr': global_vars.max_lr * 0.2},  # Transformer层使用较小的学习率
        {'params': cnn_params, 'lr': global_vars.max_lr},  # CNN层使用正常学习率
        {'params': other_params, 'lr': global_vars.max_lr}  # 其他层使用正常学习率
    ]
    
    optimizer = getattr(optim, global_vars.optimizer)(
        param_groups,
        weight_decay=0.001
    )

    # 为每个参数组设置不同的学习率调度
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=[group['lr'] for group in param_groups],
        total_steps=global_vars.num_epochs,
        pct_start=0.3,
        anneal_strategy='cos',
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=100,
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
                outputs= model(data)
                
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
            total_batches = len(loader_train)
            
            if epoch == 0 and batch_idx < 100:
                # For epoch 0, print first 100 batches individually
                print(f"Batch {batch_idx+1}/{total_batches}: Loss: {batch_loss.item():.4f}")
            elif (batch_idx + 1) % 100 == 0:
                # For other epochs, print average every 100 batches
                avg_loss = sum(batch_losses) / len(batch_losses)
                print(f"Batch {batch_idx+1}/{total_batches}: Avg Loss: {avg_loss:.4f}")
                print("Learning rates:", [f"{lr:.8f}" for lr in scheduler.get_last_lr()])
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

        