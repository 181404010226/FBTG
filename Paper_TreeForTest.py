import torch
import torch.nn as nn
import torch.optim as optim
from Paper_global_vars import global_vars
from Paper_Network import get_network
import numpy as np

class DecisionNode(nn.Module):
    def __init__(self, english_name, judge=[-1,-1], left=None, right=None, depth=0, base_lr=0.001):
        super(DecisionNode, self).__init__()
        self.model = get_network(english_name)
        self.left = left
        self.right = right
        self.english_name = english_name
        self.judge = judge
        self.node_probabilities = {}
    
    def forward(self, x, labels=None, sample_idx=None):
        outputs = self.model(x)

        global_vars.update_image_probabilities(self.judge, outputs)

        # Store probabilities for each sample
        for idx, prob in enumerate(outputs):
            if sample_idx is not None:
                idx = sample_idx[idx]
            self.node_probabilities[idx] = []
            self.node_probabilities[idx].append(prob.tolist())

        if self.left:
            self.left(x, labels, sample_idx)
        if self.right:
            self.right(x, labels, sample_idx)


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
        global_vars.initialize_image_probabilities(x.size(0))
        for node in self.nodes:
            node(x)
        return x  # 或者返回 global_vars 中的结果