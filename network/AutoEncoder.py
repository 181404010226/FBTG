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

from compression.hyperprior_model import HyperpriorDensity, HyperpriorEntropyModel
from network.hyperprior import Hyperprior  # Import Hyperprior


class AutoEncoder(nn.Module):
    def __init__(self, image_dims, batch_size, C=20, activation='relu', n_residual_blocks=8, channel_norm=True):
        super(AutoEncoder, self).__init__()
        
        self.encoder = Encoder(image_dims, batch_size, activation, C, channel_norm)
        
        # Calculate encoder output dimensions
        with torch.no_grad():
            dummy_input = torch.zeros(1, *image_dims)
            encoder_output = self.encoder(dummy_input)
            encoder_output_dims = encoder_output.shape[1:]
        
        self.generator = Generator(encoder_output_dims, batch_size, C, activation, n_residual_blocks, channel_norm)
        
        # Add hyperprior model
        self.hyperprior = Hyperprior(bottleneck_capacity=C, entropy_code=True)  # Use Hyperprior
        self.hyperprior.hyperprior_entropy_model.build_tables()  # Ensure CDF is built
    
    def forward(self, x):
        encoded = self.encoder(x)
        # Compress
        hyperinfo= self.hyperprior(encoded, tuple(x.size()[2:]))
        # Generate output
        reconstructed = self.generator(hyperinfo.decoded)
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