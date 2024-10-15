import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuronBundle(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)

    def forward(self, x):
        out = self.conv(x)
        if out.sum() > 0:
            return out
        else:
            return torch.zeros_like(out)

class NeuronBundleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, N, **kwargs):
        super().__init__()
        self.N = N
        self.neuron_bundles = nn.ModuleList([
            NeuronBundle(in_channels, out_channels, kernel_size, **kwargs) for _ in range(N)
        ])
        self.activation = nn.GELU()
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        bundle_outputs = [bundle(x) for bundle in self.neuron_bundles]
        summed = torch.stack(bundle_outputs, dim=0).sum(dim=0)
        activated = self.activation(summed)
        normalized = self.batch_norm(activated)
        return normalized

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