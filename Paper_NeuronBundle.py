import torch
import torch.nn as nn
from typing import List, Optional

class ConvBlock(nn.Module):
    """基础卷积块，包含卷积、激活函数和批归一化"""
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.activation = nn.GELU()
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.batchnorm(self.activation(self.conv(x)))
        # if self.in_channels == self.out_channels:
        #     return out + x 
        return out

class NeuronBundle(ConvBlock):
    pass
    # """单个神经元束，支持循环选择不同大小的卷积核"""
    # # 添加类变量来追踪当前使用的卷积核索引
    # current_kernel_index = 0
    # possible_kernel_sizes = [1, 2, 3, 4, 5, 6, 7]
    
    # def __init__(self, in_channels: int, out_channels: int, **kwargs):
    #     # 移除传入的 kernel_size
    #     kernel_size = kwargs.pop('kernel_size', None)
        
    #     # 循环选择卷积核大小
    #     kwargs['kernel_size'] = self.possible_kernel_sizes[self.current_kernel_index]
    #     # 更新索引
    #     NeuronBundle.current_kernel_index = (self.current_kernel_index + 1) % len(self.possible_kernel_sizes)
        
    #     # 确保 padding 设置正确
    #     if 'padding' not in kwargs:
    #         kwargs['padding'] = kwargs['kernel_size'] // 2
            
    #     super().__init__(in_channels, out_channels, **kwargs)

class NeuronBundleLayer(nn.Module):
    """神经元束层，包含多个并行的神经元束"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 num_bundles: int, **kwargs):
        super().__init__()
        self.bundles = nn.ModuleList([
            NeuronBundle(in_channels, in_channels, kernel_size=kernel_size, **kwargs) 
            for _ in range(num_bundles)
        ])
        self.merge1 = ConvBlock(
            in_channels * num_bundles, 
            in_channels, 
            kernel_size=1
        )
        self.merge2 = ConvBlock(
            in_channels, 
            out_channels, 
            kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bundle_outputs = [bundle(x) for bundle in self.bundles]
        return self.merge2(self.merge1(torch.cat(bundle_outputs, dim=1)))

# class Residual(nn.Module):
#     """残差连接包装器，支持不同通道数的输入输出"""
#     def __init__(self, module: nn.Module):
#         super().__init__()
#         self.module = module

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         out = self.module(x)
#         return out + x

def create_convmixer(
    dim: int, 
    depth: int, 
    num_bundles: int, 
    kernel_size: int = 9, 
    patch_size: int = 7, 
    num_classes: int = 1000
) -> nn.Sequential:
    """创建带有神经元束的ConvMixer模型"""
    def create_bundle_block(in_dim: int, out_dim: int) -> nn.Module:
        """创建带有残差连接的神经元束层"""
        return NeuronBundleLayer(
                in_dim, out_dim,
                kernel_size=kernel_size,
                num_bundles=num_bundles,
                groups=in_dim,
                padding="same"
        )
        

    # 初始化层
    layers: List[nn.Module] = [
        ConvBlock(3, dim, kernel_size=patch_size, stride=patch_size)
    ]

    # 构建主干网络
    for stage in range(depth):
        # 添加相同维度的残差块
        layers.append(create_bundle_block(dim, dim))
        layers.extend([
            create_bundle_block(dim, dim * 2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])
        dim *= 2

    # 添加分类头
    layers.extend([
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(dim, num_classes)
    ])

    return nn.Sequential(*layers)