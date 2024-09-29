import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

class FilterCNN(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=64, kernel_size=3):
        super(FilterCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_dim, kernel_size=kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, 1, kernel_size=1)  # 输出单通道掩码

    def forward(self, x):
        """
        前向传播函数
        :param x: 输入图像，形状 (batch_size, 3, n, n)
        :return: 掩码，形状 (batch_size, n, n)，值在0~1之间
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        mask = torch.sigmoid(self.conv3(x))  # 将输出限制在0到1之间
        mask = mask.squeeze(1)  # 移除通道维度，形状变为 (batch_size, n, n)
        return mask

    def apply_filter(self, image, mask, k):
        """
        根据掩码和百分比k过滤图像
        :param image: 输入图像，形状 (batch_size, 3, n, n)
        :param mask: 掩码，形状 (batch_size, n, n)
        :param k: 需要过滤的百分比（0-100）
        :return: 过滤后的图像，形状 (batch_size, 3, n, n)
        """
        batch_size, channels, height, width = image.size()
        # 计算每个样本需要过滤的像素数量
        num_pixels = height * width
        k_pixels = int(num_pixels * (k / 100.0))

        # 对每个样本应用过滤
        for i in range(batch_size):
            # 获取第i个样本的掩码并展开
            mask_i = mask[i].view(-1)
            # 找到k_pixels个最小掩码值的索引
            _, indices = torch.topk(mask_i, k_pixels, largest=False)
            # 将对应位置的像素值设为0
            for idx in indices:
                row = idx // width
                col = idx % width
                image[i, :, row, col] = 0
        return image    

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

class FilteredConvMixer(nn.Module):
    def __init__(self, filter_params, convmixer_params):
        """
        初始化 FilteredConvMixer 模型。
        
        :param filter_params: FilterCNN 的参数字典，例如 {'input_channels': 3, 'hidden_dim': 64, 'kernel_size': 3}
        :param convmixer_params: ConvMixer 的参数字典，例如 {'dim': 256, 'depth': 8, 'kernel_size': 9, 'patch_size': 7, 'n_classes': 1000}
        """
        super(FilteredConvMixer, self).__init__()
        self.filter_cnn = FilterCNN(**filter_params)
        self.conv_mixer = ConvMixer(**convmixer_params)
    
    def forward(self, x, k):
        """
        前向传播函数。
        
        :param x: 输入图像，形状 (batch_size, 3, n, n)
        :param k: 需要过滤的百分比（0-100）
        :return: tuple (分类结果, 过滤后的图像)
        """
        # 生成掩码
        mask = self.filter_cnn(x)
        # 应用过滤器
        filtered_x = self.filter_cnn.apply_filter(x.clone(), mask, k)
        # 分类
        classification = self.conv_mixer(filtered_x)
        return classification, filtered_x