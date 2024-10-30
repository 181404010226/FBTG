import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class ColorVariationLayer(nn.Module):
    """
    A neural network layer designed to perceive and emphasize color variations in images.
    It computes the differences between color channels and applies a learnable transformation.
    """
    def __init__(self, in_channels):
        super(ColorVariationLayer, self).__init__()
        assert in_channels >= 3, "Input must have at least 3 channels (e.g., RGB)"
        # Define learnable weights for channel differences
        self.weight_rg = nn.Parameter(torch.randn(1, 1, 1, 1))
        self.weight_bg = nn.Parameter(torch.randn(1, 1, 1, 1))
        # Optional: Add a non-linear activation
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Forward pass to compute color variations.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
        
        Returns:
            torch.Tensor: Tensor emphasizing color variations
        """
        # Assuming the first three channels are RGB
        R = x[:, 0:1, :, :]
        G = x[:, 1:2, :, :]
        B = x[:, 2:3, :, :]
        
        # Compute channel differences
        diff_rg = R - G  # Red-Green
        diff_bg = B - G  # Blue-Green
        
        # Apply learnable weights
        diff_rg = self.weight_rg * diff_rg
        diff_bg = self.weight_bg * diff_bg
        
        # Combine the differences
        color_variation = diff_rg + diff_bg
        
        # Apply activation
        color_variation = self.activation(color_variation)
        
        return color_variation

class ColorAwareConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 color_kernel_size=5, padding='same', bias=True):
        super(ColorAwareConv, self).__init__()
        # 第一个标准卷积
        self.standard_conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                                       padding=padding, bias=bias)
        # 第二个用于色彩感知的卷积
        self.color_conv = nn.Conv2d(in_channels, out_channels, color_kernel_size, 
                                    padding='same', bias=bias)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.color_Variation = ColorVariationLayer(in_channels)

    def forward(self, x):
        # 第一个标准卷积
        standard_out = self.standard_conv(x)
        standard_out = self.gelu(standard_out)
        standard_out = self.batch_norm(standard_out)
        
        # 第二个色彩感知卷积
        color_x = self.color_Variation(x)
        color_out = self.color_conv(color_x)
        color_out = self.sigmoid(color_out)
        
        # 使用色彩感知输出抑制标准卷积的输出
        suppressed_out = standard_out * color_out
        return suppressed_out

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

def ConvMixer(dim, depth, kernel_size=9, patch_size=7, n_classes=1000):
    return nn.Sequential(
        ColorVariationLayer(dim),  # Inserted ColorVariationLayer
        nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size),
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