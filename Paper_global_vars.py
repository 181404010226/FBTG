import torch
import os

class GlobalVars:
    def __init__(self):
        self.num_epochs = 300
        self.input_size = (3, 224, 224)  # astroformer #RDNet
        # self.input_size = (3, 32, 32)
        self.max_lr = 0.0005
        self.train_batch_size = 64 // torch.cuda.device_count()
        self.test_batch_size = 64
        
        # 新增配置
        self.dataset = 'cifar10'
        # self.model_name = 'SequentialDecisionTreeCIFAR100ForRDNet'
        # self.model_name = 'SequentialDecisionTreeForRDNet'
        self.model_name = 'SequentialDecisionTree'
        # self.model_name = 'SequentialDecisionTreeCIFAR100'
        self.optimizer = 'AdamW'
        # 如果目标路径存在，则使用目标路径，否则使用默认路径
        self.save_path = "/root/autodl-tmp" if os.path.exists("/root/autodl-tmp") else "/hy-tmp/best_models"
        self.debug = True
        self.debug_period = 10

global_vars = GlobalVars()