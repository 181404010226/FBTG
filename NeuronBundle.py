import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuronBundle(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, N, **kwargs):  # 添加 N 参数
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels // N, kernel_size, **kwargs)
        # 激活函数
        self.activation = nn.GELU()
        self.batchnorm = nn.BatchNorm2d(out_channels // N)

    def forward(self, x):
        out = self.conv(x)
        out = self.activation(out)
        out = self.batchnorm(out)
        return out 

class NeuronBundleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, N, **kwargs):
        super().__init__()
        self.N = N
        self.neuron_bundles = nn.ModuleList([
            NeuronBundle(in_channels, out_channels, kernel_size, N=N, **kwargs) for _ in range(N)
        ])

    def forward(self, x):
        bundle_outputs = [bundle(x) for bundle in self.neuron_bundles]
        # 将 N 个 bundle 的输出在通道维度上连接，而不是求和
        return torch.cat(bundle_outputs, dim=1)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

def ConvMixerWithNeuronBundles(dim, depth, N, kernel_size=9, patch_size=7, n_classes=1000):
    # 初始维度和层
    current_dim = dim
    current_N = N  # 添加current_N来追踪N的变化
    layers = [
        nn.Conv2d(3, current_dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(current_dim)
    ]

    # 4个阶段的下采样和通道翻倍
    for stage in range(depth):
        layers.append(
            Residual(
                nn.Sequential(
                    NeuronBundleLayer(current_dim, current_dim, kernel_size=kernel_size, 
                                    N=current_N, groups=current_dim//current_N, padding="same"),  # 使用current_N
                )
            )
        )
        
        # 在每个阶段结束时进行下采样和通道数翻倍（最后一个阶段除外）
        if stage < depth-1:
            layers.extend([
                nn.Sequential(
                    NeuronBundleLayer(current_dim, current_dim * 2, kernel_size=2, stride=2, N=current_N),  # 使用current_N
                )
            ])
            current_dim *= 2
            current_N *= 2  # N也翻倍
            
            layers.extend([
                nn.Conv2d(current_dim, current_dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(current_dim)
            ])

    # 最终分类层
    layers.extend([
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(current_dim, n_classes)
    ])

    return nn.Sequential(*layers)