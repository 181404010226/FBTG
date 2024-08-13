import torch.nn as nn

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

def InverseConvMixer(dim, depth, kernel_size=9, patch_size=7, n_classes=1000, output_size=224):
    return nn.Sequential(
        nn.Linear(n_classes, dim),
        nn.Unflatten(1, (dim, 1, 1)),
        *[nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
            Residual(nn.Sequential(
                nn.BatchNorm2d(dim),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same")
            ))
        ) for i in range(depth)],
        nn.BatchNorm2d(dim),
        nn.GELU(),
        nn.ConvTranspose2d(dim, 3, kernel_size=patch_size, stride=patch_size, output_padding=1),
        nn.Upsample(size=(output_size, output_size), mode='bilinear', align_corners=False)
    )