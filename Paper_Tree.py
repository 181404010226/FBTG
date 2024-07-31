import torch
import torch.nn as nn
import torch.optim as optim
from Paper_global_vars import global_vars
from Paper_Network import get_network
import numpy as np

class DecisionNode(nn.Module):
    def __init__(self, chinese_name, judge=[-1,-1]):
        super(DecisionNode, self).__init__()
        self.model = get_network(chinese_name)
        self.chinese_name = chinese_name
        self.judge = judge
    
    def forward(self, x):
        outputs = self.model(x)
        outputs = torch.sigmoid(outputs)
        return outputs



class SequentialDecisionTree(nn.Module):
    def __init__(self):
        super(SequentialDecisionTree, self).__init__()
        
        self.nodes = nn.ModuleList([
            DecisionNode("Industrial vs Natural", judge=[[0,1,8,9],[2,3,4,5,6,7]]),
            DecisionNode("Sky vs Land", judge=[[0,8],[1,9]]),
            DecisionNode("Airplane vs Ship", judge=[[0],[8]]),
            DecisionNode("Car vs Truck", judge=[[1],[9]]),
            DecisionNode("Others vs Quadrupeds", judge=[[2,6],[3,5],[4,7]]),
            DecisionNode("Bird vs Frog", judge=[[2],[6]]),
            DecisionNode("Cat vs Dog", judge=[[3],[5]]),
            DecisionNode("Deer vs Horse", judge=[[4],[7]])
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
