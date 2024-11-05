import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuronBundle(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, N, **kwargs):  # 添加 N 参数
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        # 激活函数
        self.activation = nn.GELU()
        self.batchnorm = nn.BatchNorm2d(out_channels)

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
        # 添加一个卷积层，将 N * out_channels 压缩为 out_channels
        self.merge_conv = nn.Conv2d(out_channels * N, out_channels, kernel_size=1)
        self.activation = nn.GELU()
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        bundle_outputs = [bundle(x) for bundle in self.neuron_bundles]
        # 将 N 个 bundle 的输出在通道维度上连接
        concatenated = torch.cat(bundle_outputs, dim=1)
        # 通过 merge_conv 压缩通道数
        out = self.merge_conv(concatenated)
        out = self.activation(out)
        out = self.batchnorm(out)
        return out

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

def ConvMixer(dim, depth,N, kernel_size=9, patch_size=7, n_classes=1000):
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(NeuronBundleLayer(dim, dim, kernel_size=kernel_size, 
                        N=N, groups=dim, padding="same")),  
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )