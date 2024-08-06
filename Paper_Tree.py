import torch
import torch.nn as nn
import torch.optim as optim
from Paper_global_vars import global_vars
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
        
        self.nodes = nn.ModuleList([
            DecisionNode(ConvMixer(dim=256, depth=8, kernel_size=5, patch_size=1, n_classes=2), judge=[[0,1,8,9],[2,3,4,5,6,7]]),
            DecisionNode(ConvMixer(dim=256, depth=8, kernel_size=5, patch_size=1, n_classes=2), judge=[[0,8],[1,9]]),
            DecisionNode(ConvMixer(dim=256, depth=8, kernel_size=5, patch_size=1, n_classes=2), judge=[[0],[8]]),
            DecisionNode(ConvMixer(dim=256, depth=8, kernel_size=5, patch_size=1, n_classes=2), judge=[[1],[9]]),
            DecisionNode(ConvMixer(dim=256, depth=8, kernel_size=5, patch_size=1, n_classes=3), judge=[[2,6],[3,5],[4,7]]),
            DecisionNode(ConvMixer(dim=256, depth=8, kernel_size=5, patch_size=1, n_classes=2), judge=[[2],[6]]),
            DecisionNode(ConvMixer(dim=256, depth=8, kernel_size=5, patch_size=1, n_classes=2), judge=[[3],[5]]),
            DecisionNode(ConvMixer(dim=256, depth=8, kernel_size=5, patch_size=1, n_classes=2), judge=[[4],[7]])
        ])
    
    def forward(self, x):
        node_outputs = [node(x) for node in self.nodes]
        
        # 初始化十分类输出张量
        final_outputs = torch.ones(x.size(0), 10, device=x.device)
        
        # 根据每个节点的输出和judge更新final_outputs
        for node, outputs in zip(self.nodes, node_outputs):
            for i, class_indices in enumerate(node.judge):
                final_outputs[:, class_indices] *= outputs[:, i].unsqueeze(1)
        
        return final_outputs

# class SequentialDecisionTreeCIFAR100(nn.Module):
#     def __init__(self):
#         super(SequentialDecisionTreeCIFAR100, self).__init__()
#         # 第一层节点：区分20个大类
#         self.root_node = DecisionNode(ConvMixer(dim=256, depth=8, kernel_size=5, patch_size=1, n_classes=20))
    
        
#         # 创建20个子节点，每个对应一个大类
#         self.sub_nodes = nn.ModuleList([
#             DecisionNode(ConvMixer(dim=256, depth=8, kernel_size=5, patch_size=1, n_classes=5))
#             for _ in range(10)
#         ])
    
#     def forward(self, x):
#         # 获取大类的概率分布
#         coarse_probs = self.root_node(x)
        
#         # 对每个大类，获取其小类的概率分布
#         fine_probs = torch.stack([sub_node(x) for sub_node in self.sub_nodes], dim=1)
        
#         # 计算最终的100类概率分布
#         final_probs = coarse_probs.unsqueeze(2) * fine_probs
        
#         # 重新排列概率以匹配原始的100个类别
#         final_probs_reordered = torch.zeros_like(final_probs.view(x.size(0), -1))
#         for i, coarse_label in enumerate(self.coarse_labels):
#             final_probs_reordered[:, i] = final_probs[:, coarse_label, i % 5]
        
#         return final_probs_reordered


class SequentialDecisionTreeCIFAR100(nn.Module):
    def __init__(self):
        super(SequentialDecisionTreeCIFAR100, self).__init__()
        
        # 第一层节点：区分10个大类
        self.root_node = DecisionNode(ConvMixer(dim=256, depth=8, kernel_size=5, patch_size=1, n_classes=10))
        
        # 10个节点：每个对应一个大类，区分其下的10个小类
        self.sub_nodes = nn.ModuleList([
            DecisionNode(ConvMixer(dim=256, depth=8, kernel_size=5, patch_size=1, n_classes=10))
            for _ in range(10)
        ])
    
    def forward(self, x):
        # 获取大类的概率分布
        coarse_probs = self.root_node(x)
        
        # 对每个大类，获取其小类的概率分布
        fine_probs = torch.stack([sub_node(x) for sub_node in self.sub_nodes], dim=1)
        
        # 计算最终的100类概率分布
        final_probs = coarse_probs.unsqueeze(2) * fine_probs
        final_probs = final_probs.view(x.size(0), -1)  # 展平为 (batch_size, 100)
        
        return final_probs