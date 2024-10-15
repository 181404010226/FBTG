import torch

class GlobalVars:
    def __init__(self):
        self.num_epochs = 300
        # self.input_size = (3, 224, 224)  # astroformer #RDNet
        self.input_size = (3, 32, 32)
        self.max_lr = 0.005
        self.train_batch_size = 64 // torch.cuda.device_count()
        self.test_batch_size = 64
        
        # 新增配置
        self.dataset = 'cifar10'
        # self.model_name = 'SequentialDecisionTreeCIFAR100ForRDNet'
        # self.model_name = 'SequentialDecisionTreeForRDNet'
        self.model_name = 'SequentialDecisionTree'
        # self.model_name = 'SequentialDecisionTreeCIFAR100'
        self.optimizer = 'AdamW'
        self.save_path = "/hy-tmp/best_models" if torch.cuda.is_available() else "/root/autodl-tmp"
        self.debug = True
        self.debug_period = 10

global_vars = GlobalVars()