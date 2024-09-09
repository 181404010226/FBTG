import torch
import torch.nn as nn
from Paper_global_vars import global_vars

class ParameterTracker(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.prev_params = None
        self.iteration_count = 0

    def forward(self, x):
        output = self.module(x)
        if self.training and global_vars.debug:
            self.iteration_count += 1
            if self.iteration_count % global_vars.debug_period == 0:
                current_params = {name: param.clone().detach() for name, param in self.module.named_parameters() if param.requires_grad}
                if self.prev_params is not None:
                    changes = [torch.abs(current_params[name] - self.prev_params[name]).mean().item() for name in current_params]
                    avg_change = sum(changes) / len(changes)
                    print(f"{self.module.__class__.__name__} average parameter change: {avg_change:.6f}")
                self.prev_params = current_params
        return output
    

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
