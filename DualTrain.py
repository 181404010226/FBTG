import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from DualConvMixer import HourglassModel
from torchvision.models import vgg16
from torchvision.transforms.functional import to_pil_image
from torch.optim.lr_scheduler import OneCycleLR
from network.AutoEncoder import AutoEncoder
import numpy as np
import torch.nn.functional as F
from torchvision.models import vgg16
from loss.perceptual_similarity.perceptual_loss import PerceptualLoss
import kornia
import numpy as np
import random

# 设置随机种子
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建保存结果的文件夹
os.makedirs("results", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# 数据预处理

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: kornia.color.rgb_to_lab(x.unsqueeze(0)).squeeze(0))
])

batch_size = 128
# 加载STL10数据集
train_dataset = datasets.STL10(root='./STL10Data', split='train+unlabeled', download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

# 加载CIFAR-10数据集
# train_dataset = datasets.CIFAR10(root='./CIFAR10RawData', train=True, download=True, transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

# 初始化模型
# model = HourglassModel().to(device)
model = AutoEncoder(image_dims=(3, 256, 256), batch_size=batch_size,
                    C=16,activation='leaky_relu').to(device)

perceptual_loss = PerceptualLoss(model='net-lin', net='alex', 
                                 colorspace='Lab', use_gpu=torch.cuda.is_available()).to(device)
# 优化器
optimizer = optim.AdamW(model.parameters(), lr=0.001)

import torch.nn.functional as F
from kornia.losses import ssim_loss

def lab_loss(output, target):
    # 计算 Delta E 2000 颜色差异

    # Separate channel losses
    l_loss = F.mse_loss(output[:, 0], target[:, 0]) / (100.0 ** 2)  # Normalize by square of range
    a_loss = F.mse_loss(output[:, 1], target[:, 1]) / (255.0 ** 2)  # Normalize by square of range
    b_loss = F.mse_loss(output[:, 2], target[:, 2]) / (255.0 ** 2)  # Normalize by square of range
    
    # SSIM loss (applied on L channel only)
    ssim = ssim_loss(output[:, 0].unsqueeze(1), target[:, 0].unsqueeze(1), window_size=11)
    
    # Separate TV loss for each channel
    tv_loss_l = (torch.mean(torch.abs(output[:, 0, :, :-1] - output[:, 0, :, 1:])) + 
                 torch.mean(torch.abs(output[:, 0, :-1, :] - output[:, 0, 1:, :]))) / 100.0  # L range is 0-100
    tv_loss_a = (torch.mean(torch.abs(output[:, 1, :, :-1] - output[:, 1, :, 1:])) + 
                 torch.mean(torch.abs(output[:, 1, :-1, :] - output[:, 1, 1:, :]))) / 255.0  # a range is -128 to 127
    tv_loss_b = (torch.mean(torch.abs(output[:, 2, :, :-1] - output[:, 2, :, 1:])) + 
                 torch.mean(torch.abs(output[:, 2, :-1, :] - output[:, 2, 1:, :]))) / 255.0  # b range is -128 to 127
    tv_loss = tv_loss_l + 0.5 * (tv_loss_a + tv_loss_b)  # Weighting channels differently

    # Separate high frequency loss for each channel
    high_freq_output_l = output[:, 0] - F.avg_pool2d(output[:, 0].unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
    high_freq_target_l = target[:, 0] - F.avg_pool2d(target[:, 0].unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
    high_freq_loss_l = F.mse_loss(high_freq_output_l, high_freq_target_l) / (100.0 ** 2)

    high_freq_output_ab = output[:, 1:] - F.avg_pool2d(output[:, 1:], kernel_size=3, stride=1, padding=1)
    high_freq_target_ab = target[:, 1:] - F.avg_pool2d(target[:, 1:], kernel_size=3, stride=1, padding=1)
    high_freq_loss_ab = F.mse_loss(high_freq_output_ab, high_freq_target_ab) / (255.0 ** 2)

    high_freq_loss = high_freq_loss_l + 0.5 * high_freq_loss_ab  # Weighting channels differently

    # 更新总损失计算
    total_loss =  0.2 * (l_loss + 0.5 * (a_loss + b_loss)) + 0.1 * ssim + 0.01 * tv_loss + 0.1 * high_freq_loss 
    return total_loss, l_loss, a_loss, b_loss, ssim, tv_loss, high_freq_loss


# 训练函数
def train(epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, _) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        
        # Calculate perceptual loss and take the mean to get a scalar
        loss_perceptual = perceptual_loss(output, data).mean()
        
        # MSE loss is already a scalar
        # loss_mse = nn.MSELoss()(output, data)
        
        # Combine losses (you can adjust the weights)
        # loss =loss_perceptual# + 0.5 * loss_mse
        # LAB-specific loss
        loss_lab, l_loss, a_loss, b_loss, ssim, tv_loss, high_freq_loss = lab_loss(output, data)

        # Combine losses
        loss = 0.3 * loss_perceptual + 0.7 * loss_lab

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    print(f'Epoch {epoch}, Batch {batch_idx}, Backward Loss: {loss.item():.4f}, '
      f' L: {l_loss.item():.4f}, '
      f'a: {a_loss.item():.4f}, b: {b_loss.item():.4f}, '
      f'SSIM: {ssim.item():.4f}, TV: {tv_loss.item():.4f}, HF: {high_freq_loss.item():.4f}')
    print(f'pLoss: {loss_perceptual.item():.4f}',
          f'lr: {optimizer.param_groups[0]["lr"]:.8f}')
    return total_loss / len(train_loader)



# 保存图像对比
def save_image_comparison(epoch, data, output):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(5):
        # 在保存图像对比函数中
        axes[0, i].imshow(to_pil_image(kornia.color.lab_to_rgb(data[i].unsqueeze(0)).squeeze(0).cpu()))
        axes[0, i].axis('off')
        axes[0, i].set_title('Original')
        axes[1, i].imshow(to_pil_image(kornia.color.lab_to_rgb(output[i].unsqueeze(0)).squeeze(0).cpu().clamp(0, 1)))
  
        # axes[1, i].imshow(to_pil_image(output[i].cpu().clamp(-1, 1)))
        axes[1, i].axis('off')
        axes[1, i].set_title('Reconstructed')
    
    plt.tight_layout()
    plt.savefig(f'results/256*256LAB_FULL_stl10_epoch_{epoch}.png')
    plt.close()


# 训练循环
num_epochs = 50

for epoch in range(1, num_epochs + 1):
    avg_loss= train(epoch)
    print(f'Epoch {epoch}, Average Loss: {avg_loss:.4f}')
    
    # 保存模型检查点
    if epoch % 5 == 0:
        torch.save(model.state_dict(), f'checkpoints/model_epoch_{epoch}.pth')
    
    # 生成并保存图像对比
    if epoch % 1 == 0:
        model.eval()
        with torch.no_grad():
            data = next(iter(train_loader))[0][:5].to(device)
            output = model(data)
            save_image_comparison(epoch, data, output)

print("Training completed!")