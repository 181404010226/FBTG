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
        self.attention = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, h*w).transpose(1, 2)
        out, _ = self.attention(x, x, x)
        out = out.transpose(1, 2).view(b, c, h, w)
        return out

class SelfAttentionLevelUp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        # 添加投影层和dropout
        self.qkv = nn.Linear(dim, dim * 3)
        self.dropout = nn.Dropout(0.1)
        self.proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        
    def forward(self, x):
        b, c, h, w = x.shape
        # 调整形状为 (batch, seq_len, dim)
        x = x.view(b, c, h*w).transpose(1, 2)
        
        # 生成 Q, K, V
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.contiguous(), qkv)
        
        # 计算注意力
        attn_output, _ = self.attention(
            q * self.scale,  # 添加缩放因子
            k,
            v,
            need_weights=False
        )
        
        # 投影和dropout
        x = self.dropout(self.proj(attn_output))
        
        # 恢复原始形状
        x = x.transpose(1, 2).view(b, c, h, w)
        return x

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