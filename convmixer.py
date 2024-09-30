import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

class PowerNonlinearity(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.a = nn.Parameter(torch.rand(dim))
        self.b = nn.Parameter(torch.rand(dim))
        self.c = nn.Parameter(torch.rand(dim))
        self.d = nn.Parameter(torch.rand(dim))

    def forward(self, x):
        return torch.pow(x, self.a) + torch.pow(x, self.b) + torch.pow(x, self.c) + self.d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

def ConvMixer(dim, depth, kernel_size=9, patch_size=7, n_classes=1000):
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )