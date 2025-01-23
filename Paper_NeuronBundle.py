import torch
import torch.nn as nn
from typing import List, Optional
import math
from timm.models.swin_transformer import SwinTransformerBlock
from torch.nn import TransformerEncoderLayer, LayerNorm


class ConvBlock(nn.Module):
    """基础卷积块，包含卷积、激活函数和批归一化"""
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.activation = nn.GELU()
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.batchnorm(self.activation(self.conv(x)))
        return out

class NeuronBundle(nn.Module):
    """增强版神经元束，支持多种神经网络层"""
    _current_layer_index = 0  # 使用下划线表示这是一个内部类变量
    _total_layer_types = 8    # 总的层类型数量（4个卷积 + 1个Swin）
    
    @classmethod
    def get_next_layer_index(cls):
        """获取下一个层索引并更新类变量"""
        current = cls._current_layer_index
        cls._current_layer_index = (cls._current_layer_index + 1) % cls._total_layer_types
        return current
    
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.swin_initialized = False
        
        # 在初始化时获取当前实例应该使用的层索引
        self.current_layer_index = self.get_next_layer_index()
        
        # 定义基础卷积层
        self.conv_configs = [
            {'kernel_size': 3, 'padding': 1, 'groups': in_channels},
            {'kernel_size': 5, 'padding': 2, 'groups': in_channels},
            {'kernel_size': 7, 'padding': 3, 'groups': in_channels},
            {'kernel_size': 9, 'padding': 4, 'groups': in_channels},
            {'kernel_size': 3, 'padding': 1, 'groups': in_channels},
            {'kernel_size': 5, 'padding': 2, 'groups': in_channels},
            {'kernel_size': 7, 'padding': 3, 'groups': in_channels},
            {'kernel_size': 9, 'padding': 4, 'groups': in_channels}
        ]
        
        # 只初始化当前需要的层
        if self.current_layer_index < len(self.conv_configs):
            # 如果是卷积层，只创建需要的那一个
            self.layer = ConvBlock(
                in_channels, 
                out_channels, 
                **self.conv_configs[self.current_layer_index]
            )
        else:
            # 如果是Transformer，先设为None，等待第一次forward时初始化
            self.layer = None
            # 保存transformer的配置
            self.transformer_config = {
                'd_model': in_channels,
                'nhead': 1,  # 头的数量，必须能整除d_model
                'dim_feedforward': in_channels * 4,
                'dropout': 0.0,
                'activation': 'gelu',
                'batch_first': True,
                'norm_first': True
            }
            # # 如果是Swin Transformer，先设为None，等待第一次forward时初始化
            # self.layer = None
            # # 保存swin transformer的配置，根据timm库的参数要求调整
            # self.swin_config = {
            #     'dim': in_channels,
            #     'input_resolution': None,  # 将在forward时设置
            #     'num_heads': 1,
            #     'window_size': 7,
            #     'shift_size': 0,
            #     'mlp_ratio': 4.0,
            #     'qkv_bias': True,
            #     'proj_drop': 0.,
            #     'attn_drop': 0.,
            #     'drop_path': 0.,
            #     'norm_layer': nn.LayerNorm
            # }

    def _init_swin_block(self, H: int, W: int):
        """根据输入尺寸初始化Swin Transformer块"""
        self.swin_config['input_resolution'] = (H, W)
        try:
            self.layer = SwinTransformerBlock(**self.swin_config)
            # 将swin block添加到设备上
            self.layer = self.layer.to('cuda')
            self.swin_initialized = True
        except Exception as e:
            print(f"Swin Transformer initialization failed: {str(e)}")
            # 如果Swin初始化失败，使用默认的卷积层作为后备
            self.layer = ConvBlock(
                self.in_channels,
                self.out_channels,
                kernel_size=3,
                padding=1,
                groups=self.in_channels
            )
            self.swin_initialized = True

    def _init_transformer_block(self, H: int, W: int):
        """根据输入尺寸初始化Transformer块"""
        try:
            self.layer = TransformerEncoderLayer(**self.transformer_config)
            # 将transformer添加到设备上
            self.layer = self.layer.to('cuda')
            self.swin_initialized = True  # 保持变量名不变
        except Exception as e:
            print(f"Transformer initialization failed: {str(e)}")
            # 如果Transformer初始化失败，使用默认的卷积层作为后备
            self.layer = ConvBlock(
                self.in_channels,
                self.out_channels,
                kernel_size=3,
                padding=1,
                groups=self.in_channels
            )
            self.swin_initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 如果是Transformer且还未初始化
        if self.current_layer_index >= len(self.conv_configs) and not self.swin_initialized:
            _, _, H, W = x.shape
            self._init_transformer_block(H, W)
        
        # 处理Transformer的特殊输入格式
        if isinstance(self.layer, TransformerEncoderLayer):
            B, C, H, W = x.shape
            # 重塑为序列形式 (batch, seq_len, features)
            x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
            x = self.layer(x)
            # 重塑回原始形状
            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
            return x
            
        return self.layer(x)
    
    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     # 如果是Swin Transformer且还未初始化
    #     if self.current_layer_index >= len(self.conv_configs) and not self.swin_initialized:
    #         _, _, H, W = x.shape
    #         self._init_swin_block(H, W)
        
    #     # 处理Swin Transformer的特殊输入格式
    #     if isinstance(self.layer, SwinTransformerBlock):
    #         B, C, H, W = x.shape
    #         x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
    #         x = self.layer(x)+x
    #         x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
    #         return x
            
    #     return self.layer(x)

class NeuronBundleLayer(nn.Module):
    """神经元束层，包含多个并行的神经元束"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 num_bundles: int, **kwargs):
        super().__init__()
        self.bundles = nn.ModuleList([
            NeuronBundle(in_channels, in_channels) 
            for _ in range(num_bundles)
        ])
        self.merge1 = ConvBlock(
            in_channels * num_bundles, 
            in_channels, 
            kernel_size=1
        )
        self.merge2 = ConvBlock(
            in_channels, 
            out_channels, 
            kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bundle_outputs = [bundle(x) for bundle in self.bundles]
        return self.merge2(self.merge1(torch.cat(bundle_outputs, dim=1)))

# class Residual(nn.Module):
#     """残差连接包装器，支持不同通道数的输入输出"""
#     def __init__(self, module: nn.Module):
#         super().__init__()
#         self.module = module

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         out = self.module(x)
#         return out + x

def create_convmixer(
    dim: int, 
    depth: int, 
    num_bundles: int, 
    kernel_size: int = 9, 
    patch_size: int = 7, 
    num_classes: int = 1000
) -> nn.Sequential:
    """创建带有神经元束的ConvMixer模型"""
    def create_bundle_block(in_dim: int, out_dim: int) -> nn.Module:
        """创建带有残差连接的神经元束层"""
        return NeuronBundleLayer(
                in_dim, out_dim,
                kernel_size=kernel_size,
                num_bundles=num_bundles,
                groups=in_dim,
                padding="same"
        )
        

    # 初始化层
    layers: List[nn.Module] = [
        ConvBlock(3, dim, kernel_size=patch_size, stride=patch_size)
    ]

    # 构建主干网络
    for stage in range(depth):
        # 添加相同维度的残差块
        layers.append(create_bundle_block(dim, dim))
        layers.extend([
            create_bundle_block(dim, dim * 2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])
        dim *= 2

    # 添加分类头
    layers.extend([
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(dim, num_classes)
    ])

    return nn.Sequential(*layers)