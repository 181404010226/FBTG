import torch
import torch.nn as nn
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

try:
    from encoder import Encoder
    from generator import Generator
except ImportError:
    from .encoder import Encoder
    from .generator import Generator

from compressai.models import CompressionModel
from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN

class AutoEncoder(CompressionModel):
    def __init__(self, image_dims, batch_size, C=20, activation='relu', n_residual_blocks=8, channel_norm=True):
        super().__init__(entropy_bottleneck_channels=C)
        
        self.encoder = Encoder(image_dims, batch_size, activation, C, channel_norm)
        
        # Calculate encoder output dimensions
        with torch.no_grad():
            dummy_input = torch.zeros(1, *image_dims)
            encoder_output = self.encoder(dummy_input)
            encoder_output_dims = encoder_output.shape[1:]
        
        self.generator = Generator(encoder_output_dims, batch_size, C, activation, n_residual_blocks, channel_norm)
        
        # Add CompressAI components
        self.entropy_bottleneck = EntropyBottleneck(C)
        self.g_a = nn.Sequential(
            GDN(C),
            nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1)
        )
        self.g_s = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1),
            GDN(C, inverse=True)
        )
        
    def update(self):
        self.entropy_bottleneck.update()
    
    def forward(self, x):
        encoded = self.encoder(x)
        y = self.g_a(encoded)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        decoded = self.g_s(y_hat)
        reconstructed = self.generator(decoded)
        
        # Compress the latent representation
        compressed = self.entropy_bottleneck.compress(y)
        
        return reconstructed, compressed

    def compress(self, x):
        encoded = self.encoder(x)
        y = self.g_a(encoded)
        compressed = self.entropy_bottleneck.compress(y)
        return compressed

    def decompress(self, strings, shape):
        y_hat = self.entropy_bottleneck.decompress(strings, shape)
        decoded = self.g_s(y_hat)
        reconstructed = self.generator(decoded)
        return reconstructed



# 测试代码
if __name__ == "__main__":

    batch_size = 32
    image_dims = (3, 256, 256)  # (channels, height, width)
    
    model = AutoEncoder(image_dims, batch_size)
    
    # 创建一个随机输入张量来测试模型
    dummy_input = torch.randn(batch_size, *image_dims)
    
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # 确保输入和输出的形状相同
    assert dummy_input.shape == output.shape, "Input and output shapes do not match!"
    print("AutoEncoder test passed successfully!")

    # Save the model
    torch.save(model.state_dict(), "autoencoder_model.pth")

    # Get the file size
    file_size = os.path.getsize("autoencoder_model.pth")
    print(f"AutoEncoder model size on disk: {file_size / 1024:.2f} KB")

    # Optionally, remove the file after checking its size
    os.remove("autoencoder_model.pth")