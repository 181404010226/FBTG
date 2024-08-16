import torch
import numpy as np
import torch
import torch.nn.functional as F

class GlobalVars:
    def __init__(self):
        self.num_epochs = 25
        self.input_size=(3, 224, 224) #astroformer #RDNet
        # self.input_size=(3, 32, 32)
        # self.input_size=(1, 28, 28)
        self.max_lr=0.000020
        self.train_batch_size = 64 // torch.cuda.device_count()
        self.test_batch_size = 64

global_vars = GlobalVars()  # 假设有10个标签