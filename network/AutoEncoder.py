import torch
import torch.nn as nn


if __name__ == "__main__":
    from encoder import Encoder
    from generator import Generator
else:
    from .encoder import Encoder
    from .generator import Generator


class AutoEncoder(nn.Module):
    def __init__(self, image_dims, batch_size, C=20, activation='relu', n_residual_blocks=8, channel_norm=True):
        super(AutoEncoder, self).__init__()
        
        self.encoder = Encoder(image_dims, batch_size, activation, C, channel_norm)
        
        # 计算encoder输出的维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, *image_dims)
            encoder_output = self.encoder(dummy_input)
            encoder_output_dims = encoder_output.shape[1:]
        
        self.generator = Generator(encoder_output_dims, batch_size, C, activation, n_residual_blocks, channel_norm)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.generator(encoded)
        return decoded

# 测试代码
if __name__ == "__main__":
    batch_size = 32
    image_dims = (3, 96, 96)  # (channels, height, width)
    
    model = AutoEncoder(image_dims, batch_size)
    
    # 创建一个随机输入张量来测试模型
    dummy_input = torch.randn(batch_size, *image_dims)
    
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # 确保输入和输出的形状相同
    assert dummy_input.shape == output.shape, "Input and output shapes do not match!"
    print("AutoEncoder test passed successfully!")