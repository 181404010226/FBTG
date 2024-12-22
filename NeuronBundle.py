import torch
import torch.nn as nn
from typing import List, Optional

class ConvBlock(nn.Module):
    """基础卷积块，包含卷积、激活函数和批归一化"""
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.activation = nn.GELU()
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.batchnorm(self.activation(self.conv(x)))

class NeuronBundle(ConvBlock):
    """单个神经元束，继承自ConvBlock"""
    pass

class NeuronBundleLayer(nn.Module):
    """神经元束层，包含多个并行的神经元束"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 num_bundles: int, **kwargs):
        super().__init__()
        self.bundles = nn.ModuleList([
            NeuronBundle(in_channels, out_channels, kernel_size=kernel_size, **kwargs) 
            for _ in range(num_bundles)
        ])
        self.merge = ConvBlock(
            out_channels * num_bundles, 
            out_channels, 
            kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bundle_outputs = [bundle(x) for bundle in self.bundles]
        return self.merge(torch.cat(bundle_outputs, dim=1))

class Residual(nn.Module):
    """残差连接包装器，支持不同通道数的输入输出"""
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.module(x)
        if out.size(1) != x.size(1):  # 检查通道数是否不同
            # 假设输出通道数是输入的整数倍
            repeat_factor = out.size(1) // x.size(1)
            x = x.repeat(1, repeat_factor, 1, 1)  # 在通道维度上重复
        return out + x

def create_convmixer(
    dim: int, 
    depth: int, 
    num_bundles: int, 
    kernel_size: int = 9, 
    patch_size: int = 7, 
    num_classes: int = 1000
) -> nn.Sequential:
    """创建带有神经元束的ConvMixer模型"""
    layers: List[nn.Module] = [
        ConvBlock(3, dim, kernel_size=patch_size, stride=patch_size)
    ]

    for stage in range(depth):
        # 添加残差块
        layers.append(Residual(
            NeuronBundleLayer(
                dim, dim, 
                kernel_size=kernel_size,
                num_bundles=num_bundles,
                groups=dim, 
                padding="same"
            )
        ))
        
        # 在非最后阶段添加下采样层
        if stage < depth - 1:
            layers.extend([
                Residual(   
                    NeuronBundleLayer(
                        dim, dim * 2,
                        kernel_size=kernel_size,
                        num_bundles=num_bundles,
                        groups=dim, 
                        padding="same"
                    )
                ),
                nn.AvgPool2d(kernel_size=2, stride=2),
            ])
            dim *= 2

    # 添加分类头
    layers.extend([
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(dim, num_classes)
    ])

    return nn.Sequential(*layers)