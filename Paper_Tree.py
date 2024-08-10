import torch
import torch.nn as nn
import torch.optim as optim
from Paper_global_vars import global_vars
from astroformer import MaxxVit, model_cfgs
from convmixer import ConvMixer
import numpy as np

class DecisionNode(nn.Module):
    def __init__(self, model, judge=[-1,-1]):
        super(DecisionNode, self).__init__()
        self.model = model
        self.judge = judge
    
    def forward(self, x):
        outputs = self.model(x)
        outputs = torch.sigmoid(outputs)
        return outputs

class SequentialDecisionTree(nn.Module):
    def __init__(self):
        super(SequentialDecisionTree, self).__init__()
        self.isTree = True
        self.debug = True
        
        self.nodes = nn.ModuleList([
            DecisionNode(MaxxVit(model_cfgs['astroformer_0'], num_classes=2), judge=[[0,1,8,9],[2,3,4,5,6,7]]),
            DecisionNode(MaxxVit(model_cfgs['astroformer_0'], num_classes=2), judge=[[0,8],[1,9]]),
            DecisionNode(MaxxVit(model_cfgs['astroformer_0'], num_classes=2), judge=[[0],[8]]),
            DecisionNode(MaxxVit(model_cfgs['astroformer_0'], num_classes=2), judge=[[1],[9]]),
            DecisionNode(MaxxVit(model_cfgs['astroformer_0'], num_classes=3), judge=[[2,6],[3,5],[4,7]]),
            DecisionNode(MaxxVit(model_cfgs['astroformer_0'], num_classes=2), judge=[[2],[6]]),
            DecisionNode(MaxxVit(model_cfgs['astroformer_0'], num_classes=2), judge=[[3],[5]]),
            DecisionNode(MaxxVit(model_cfgs['astroformer_0'], num_classes=2), judge=[[4],[7]])
        ])
    
    def forward(self, x):
        final_outputs = torch.ones(x.size(0), 10, device=x.device)
        
        for node in self.nodes:
            outputs = node(x)
            if self.debug:
                print(outputs.shape)
                self.debug = False
            for i, class_indices in enumerate(node.judge):
                final_outputs[:, class_indices] *= outputs[:, i].unsqueeze(1)
        
        return final_outputs


# class SequentialDecisionTree(nn.Module):
#     def __init__(self):
#         super(SequentialDecisionTree, self).__init__()
#         self.isTree = True
        
#         self.nodes = nn.ModuleList([
#             DecisionNode(ConvMixer(dim=256, depth=16, kernel_size=9, patch_size=1, n_classes=2), judge=[[0,1,8,9],[2,3,4,5,6,7]]),
#             DecisionNode(ConvMixer(dim=256, depth=16, kernel_size=9, patch_size=1, n_classes=2), judge=[[0,8],[1,9]]),
#             DecisionNode(ConvMixer(dim=256, depth=16, kernel_size=9, patch_size=1, n_classes=2), judge=[[0],[8]]),
#             DecisionNode(ConvMixer(dim=256, depth=16, kernel_size=9, patch_size=1, n_classes=2), judge=[[1],[9]]),
#             DecisionNode(ConvMixer(dim=256, depth=16, kernel_size=9, patch_size=1, n_classes=3), judge=[[2,6],[3,5],[4,7]]),
#             DecisionNode(ConvMixer(dim=256, depth=16, kernel_size=9, patch_size=1, n_classes=2), judge=[[2],[6]]),
#             DecisionNode(ConvMixer(dim=256, depth=16, kernel_size=9, patch_size=1, n_classes=2), judge=[[3],[5]]),
#             DecisionNode(ConvMixer(dim=256, depth=16, kernel_size=9, patch_size=1, n_classes=2), judge=[[4],[7]])
#         ])

#     def forward(self, x):
#         final_outputs = torch.ones(x.size(0), 10, device=x.device)
        
#         for node in self.nodes:
#             outputs = node(x)
#             for i, class_indices in enumerate(node.judge):
#                 final_outputs[:, class_indices] *= outputs[:, i].unsqueeze(1)
        
#         return final_outputs

class SequentialDecisionTreeCIFAR100(nn.Module):
    def __init__(self):
        super(SequentialDecisionTreeCIFAR100, self).__init__()
        self.isTree = True
        # 第一层节点：区分20个大类
        self.root_node = DecisionNode(ConvMixer(dim=256, depth=16, kernel_size=7, patch_size=1, n_classes=20))
        
        # 创建20个子节点，每个对应一个大类
        self.sub_nodes = nn.ModuleList([
            DecisionNode(ConvMixer(dim=256, depth=16, kernel_size=7, patch_size=1, n_classes=5))
            for _ in range(20)
        ])
        
        # 定义CIFAR-100的20个大类到100个小类的映射
        self.coarse_labels = [
            4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
            3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
            6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
            0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
            5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
            16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
            10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
            2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
            16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
            18, 1, 2, 15, 6, 0, 17, 8, 14, 13
        ]
    
    def forward(self, x):
        # 获取大类的概率分布
        coarse_probs = self.root_node(x)
        
        # 对每个大类，获取其小类的概率分布
        fine_probs = torch.stack([sub_node(x) for sub_node in self.sub_nodes], dim=1)
        
        # 计算最终的100类概率分布
        final_probs = coarse_probs.unsqueeze(2) * fine_probs
        
        # 重新排列概率以匹配原始的100个类别
        final_probs_reordered = torch.zeros_like(final_probs.view(x.size(0), -1))
        for i, coarse_label in enumerate(self.coarse_labels):
            final_probs_reordered[:, i] = final_probs[:, coarse_label, i % 5]
        
        return final_probs_reordered


# class SequentialDecisionTreeCIFAR100(nn.Module):
#     def __init__(self):
#         super(SequentialDecisionTreeCIFAR100, self).__init__()
#         self.isTree = True
        
#         # 第一层节点：区分10个大类
#         self.root_node = DecisionNode(ConvMixer(dim=256, depth=8, kernel_size=5, patch_size=1, n_classes=10))
        
#         # 10个节点：每个对应一个大类，区分其下的10个小类
#         self.sub_nodes = nn.ModuleList([
#             DecisionNode(ConvMixer(dim=256, depth=8, kernel_size=5, patch_size=1, n_classes=10))
#             for _ in range(10)
#         ])
    
#     def forward(self, x):
#         # 获取大类的概率分布
#         coarse_probs = self.root_node(x)
        
#         # 对每个大类，获取其小类的概率分布
#         fine_probs = torch.stack([sub_node(x) for sub_node in self.sub_nodes], dim=1)
        
#         # 计算最终的100类概率分布
#         final_probs = coarse_probs.unsqueeze(2) * fine_probs
#         final_probs = final_probs.view(x.size(0), -1)  # 展平为 (batch_size, 100)
        
#         return final_probs
