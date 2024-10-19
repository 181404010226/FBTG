import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuronBundle(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)

    def forward(self, x):
        out = self.conv(x)
        # 计算每个通道的平均激活值
        channel_mean = out.mean(dim=(1,2, 3), keepdim=True)
        # 使用 sigmoid 函数将通道平均值映射到 (0, 1) 范围
        channel_gate = torch.sigmoid(channel_mean)
        # 将 channel_gate 广播到与 out 相同的形状，并相乘
        return out * channel_gate

class NeuronBundleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, N, **kwargs):
        super().__init__()
        self.N = N
        self.neuron_bundles = nn.ModuleList([
            NeuronBundle(in_channels, out_channels, kernel_size, **kwargs) for _ in range(N)
        ])

    def forward(self, x):
        bundle_outputs = [bundle(x) for bundle in self.neuron_bundles]
        summed = torch.stack(bundle_outputs, dim=0).sum(dim=0)
        return summed

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

def ConvMixerWithNeuronBundles(dim, depth, N, kernel_size=9, patch_size=7, n_classes=1000):
    layers = [
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim)
    ]

    for _ in range(depth):
        layers.append(
            Residual(
                nn.Sequential(
                    NeuronBundleLayer(dim, dim, kernel_size=kernel_size, N=N, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )
            )
        )
        # 可选择在神经元集束层之间添加逐点卷积层
        layers.extend([
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        ])

    layers.extend([
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    ])

    return nn.Sequential(*layers)