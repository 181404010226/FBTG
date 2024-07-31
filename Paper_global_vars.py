import torch
import numpy as np
import torch
import torch.nn.functional as F

class GlobalVars:
    def __init__(self):
        self.num_epochs = 500
        self.train_batch_size = 64
        self.test_batch_size = 64
        self.num_classes = 10
        self.log_image_probabilities = None

global_vars = GlobalVars()  # 假设有10个标签