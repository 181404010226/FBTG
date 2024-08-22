import torch
import torch.nn as nn
import torch.nn.functional as F

# class Residual(nn.Module):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn

#     def forward(self, x):
#         return self.fn(x) + x
    

# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
#         super(ConvBlock, self).__init__()
#         self.conv = nn.Sequential(
#             Residual(nn.Sequential(
#                     nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding),
#                     nn.GELU(),
#                     nn.BatchNorm2d(in_channels)
#                 )),
#             nn.Conv2d(in_channels, out_channels, kernel_size=1),
#             nn.GELU(),
#             nn.BatchNorm2d(out_channels)
#         )

#     def forward(self, x):
#         return self.conv(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=3, stride=1, padding=1):
        super(UpBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)
        self.conv = ConvBlock(in_channels, out_channels,kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.upsample(x)
        return self.conv(x)

# class DeconvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=0):
#         super(DeconvBlock, self).__init__()
#         self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         return self.relu(self.bn(self.deconv(x)))

class HourglassModel(nn.Module):
    def __init__(self, input_channels=3, latent_dim=128):
        super(HourglassModel, self).__init__()
        
        # Encoder (正卷积部分)
        self.encoder = nn.Sequential(
            ConvBlock(input_channels, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2),
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(2),
            ConvBlock(256, 512),
            ConvBlock(512, 512),
            nn.MaxPool2d(2),
            ConvBlock(512, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            UpBlock(latent_dim, 512),
            UpBlock(512, 256),
            UpBlock(256, 128),
            UpBlock(128, 64),
            nn.Conv2d(64, input_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        
        # # Decoder (反卷积部分)
        # self.decoder = nn.Sequential(
        #     DeconvBlock(latent_dim, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     DeconvBlock(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     DeconvBlock(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     DeconvBlock(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.Conv2d(64, input_channels, kernel_size=3, stride=1, padding=1),
        #     nn.Tanh()
        # )

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output

    def encode(self, x):
        return self.encoder(x)

    def decode(self, latent):
        return self.decoder(latent)

# 测试模型
if __name__ == "__main__":
    # 创建一个示例输入
    batch_size = 1
    channels = 3
    height, width = 32, 32
    x = torch.randn(batch_size, channels, height, width)

    # 初始化模型
    model = HourglassModel()

    # 前向传播
    output = model(x)

    print(model.encode(x).shape)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    # print(f"Model architecture:\n{model}")