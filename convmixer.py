import torch
import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, h*w).transpose(1, 2)
        out, _ = self.attention(x, x, x)
        out = out.transpose(1, 2).view(b, c, h, w)
        return out

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
                nn.BatchNorm2d(dim),
                Residual(SelfAttention(dim))
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # Create a ConvMixer model
    model = ConvMixer(dim=256, depth=8)
    
    # Count and print the number of parameters
    num_params = count_parameters(model)
    print(f"Number of parameters: {num_params:,}")

if __name__ == "__main__":
    main()