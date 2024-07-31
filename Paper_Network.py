import torch
import torch.nn as nn
import torch.nn.functional as F
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

def ConvMixer(dim=256, depth=8, kernel_size=5, patch_size=2, n_classes=2):
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

class BinaryConvMixer(nn.Module):
    def __init__(self, model_path, dim, depth, kernel_size, patch_size, n_classes=2):
        super(BinaryConvMixer, self).__init__()
        self.model = ConvMixer(dim, depth, kernel_size, patch_size, n_classes=n_classes).to(device)
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            # Remove the 'module.' prefix from state dict keys
            if model_path.endswith('.tar'):
                new_state_dict = {k.replace('module.', ''): v for k, v in state_dict['state_dict'].items()}
            else:
                new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(new_state_dict)
        self.epsilon = 1e-7

    def forward(self, x):
        x = self.model(x)
        if torch.isnan(x).any():
            print(f"Warning: NaN values detected in output")
            print("Detailed output values:")
            for i, batch in enumerate(x):
                print(f"Batch {i}: {batch.tolist()}")
        x = torch.sigmoid(x)
        return x

# Replace all specific network classes with BinaryConvMixer
IndustrialVsNaturalNet = lambda: BinaryConvMixer("",256,8,5,1)
LandVsSkyNet = lambda: BinaryConvMixer("",256,8,5,1)
PlaneVsShipNet = lambda: BinaryConvMixer("",256,8,5,1)
CarVsTruckNet = lambda: BinaryConvMixer("",256,8,5,1)
FourLeggedVsOthersNet = lambda: BinaryConvMixer("",256,8,5,1,3)
CatVsDogNet = lambda: BinaryConvMixer("",256,8,5,1)
DeerVsHorseNet = lambda: BinaryConvMixer("",256,8,5,1)
BirdVsFrogNet = lambda: BinaryConvMixer("",256,8,5,1)

# IndustrialVsNaturalNet = lambda: BinaryConvMixer("data/train工业vx自然/model_0.9881_epoch82.pth",256,8,5,2)
# LandVsSkyNet = lambda: BinaryConvMixer("data/train飞机轮船vs汽车卡车/model_0.9888_epoch110.pth",256,8,5,2)
# PlaneVsShipNet = lambda: BinaryConvMixer("data/train飞机vs轮船/model_0.9815_epoch112.pth",256,8,5,2)
# CarVsTruckNet = lambda: BinaryConvMixer("data/train汽车vs卡车/model_0.9795_epoch104.pth",256,8,5,2)
# FourLeggedVsOthersNet = lambda: BinaryConvMixer("data/train鸟青蛙vs四脚兽/model_0.9720_epoch102.pth",256,8,5,2)
# CatDogVsDeerHorseNet = lambda: BinaryConvMixer("data/train猫狗vs马鹿/model_0.9695_epoch98.pth",256,8,5,2)
# CatVsDogNet = lambda: BinaryConvMixer("data/train猫vs狗/model_0.9175_epoch95.pth",256,8,5,2)
# DeerVsHorseNet = lambda: BinaryConvMixer("data/train马vs鹿/model_0.9885_epoch102.pth",256,8,5,2)
# BirdVsFrogNet = lambda: BinaryConvMixer("data/train鸟vs青蛙/model_0.9830_epoch104.pth",256,8,5,2)


def get_network(node_name):
    networks = {
        "Industrial vs Natural": IndustrialVsNaturalNet,
        "Sky vs Land": LandVsSkyNet,
        "Airplane vs Ship": PlaneVsShipNet,
        "Car vs Truck": CarVsTruckNet,
        "Others vs Quadrupeds": FourLeggedVsOthersNet,
        "Cat vs Dog": CatVsDogNet,
        "Deer vs Horse": DeerVsHorseNet,
        "Bird vs Frog": BirdVsFrogNet
    }
    return networks[node_name]()