import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import kornia
from tqdm import tqdm
import numpy as np
from network.AutoEncoder import AutoEncoder
import multiprocessing
import os

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义一个可序列化的函数来替代 lambda 函数
def rgb_to_lab(x):
    return kornia.color.rgb_to_lab(x.unsqueeze(0)).squeeze(0)

def main():
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Lambda(rgb_to_lab)
    ])

    # 加载STL10数据集
    batch_size = 128
    dataset = datasets.STL10(root='./STL10Data', split='train+unlabeled', download=True, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 初始化模型
    model = AutoEncoder(image_dims=(3, 256, 256), batch_size=batch_size, C=16, activation='leaky_relu').to(device)

    # 加载预训练模型
    model.load_state_dict(torch.load("STL10压缩.pth", map_location=device, weights_only=True))

    output_dir = "compressed_output"
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, desc="Compressing")):
            batch = batch.to(device)
            
            # 压缩
            compressed, coding_shape, symbols = model.compress(batch)
            
            # 保存压缩结果
            compressed_file = os.path.join(output_dir, f"compressed_{i}.npz")
            np.savez_compressed(compressed_file, 
                                compressed=compressed.cpu().numpy(), 
                                coding_shape=coding_shape,
                                symbols=symbols.cpu().numpy())
            
            # 可选：解压缩以验证
            # batch_shape = batch.shape[0]
            # broadcast_shape = batch.shape[2:]
            # reconstructed, _ = model.decompress(compressed, batch_shape, broadcast_shape, coding_shape)
            
            # # 保存重建结果（可选）
            # reconstructed_file = os.path.join(output_dir, f"reconstructed_{i}.png")
            # reconstructed_image = reconstructed[0].cpu().permute(1, 2, 0).numpy()
            # reconstructed_image = (reconstructed_image * 255).astype(np.uint8)
            # Image.fromarray(reconstructed_image).save(reconstructed_file)
    
    print(f"Compression completed. Results saved to {output_dir}")


if __name__ == '__main__':
    multiprocessing.freeze_support()  # 这行在 Windows 上是必要的
    main()