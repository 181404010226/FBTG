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
        return out


class NeuronBundle(nn.Module):
    """简化版神经元束，仅使用相同的卷积核"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 使用单一卷积核大小
        self.layer = ConvBlock(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size,
            padding=kernel_size//2,
            groups=in_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class NeuronBundleLayer(nn.Module):
    """神经元束层，包含多个并行的神经元束"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 num_bundles: int, **kwargs):
        super().__init__()
        self.bundles = nn.ModuleList([
            NeuronBundle(in_channels, in_channels, kernel_size=kernel_size) 
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
        # 横向ResNet风格: 每个bundle的输出加到下一个bundle的输入
        bundle_outputs = []
        bundle_input = x
        
        for bundle in self.bundles:
            bundle_output = bundle(bundle_input)
            bundle_outputs.append(bundle_output)
            # 当前输出加到下一个bundle的输入
            # bundle_input = bundle_output + x
            
        # 将所有bundle的输出拼接并通过1x1卷积融合
        return self.merge2(self.merge1(torch.cat(bundle_outputs, dim=1)))


def create_convmixer(
    dim: int, 
    depth: int, 
    num_bundles: int, 
    kernel_size: int = 5, 
    patch_size: int = 7, 
    num_classes: int = 1000
) -> nn.Sequential:
    """创建带有神经元束的ConvMixer模型"""
    def create_bundle_block(in_dim: int, out_dim: int) -> nn.Module:
        """创建神经元束层"""
        return NeuronBundleLayer(
                in_dim, out_dim,
                kernel_size=kernel_size,
                num_bundles=num_bundles,
                groups=in_dim,
                padding="same"
        )

    # 初始化层
    layers: List[nn.Module] = [
        ConvBlock(3, dim, kernel_size=patch_size,  stride=patch_size)
    ]

    # 构建主干网络
    for stage in range(depth):
        # 添加相同维度的块
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