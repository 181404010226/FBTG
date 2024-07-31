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
        global_vars.update_image_probabilities(self.judge, outputs)
        return outputs



class SequentialDecisionTree(nn.Module):
    def __init__(self):
        super(SequentialDecisionTree, self).__init__()
        self.nodes = nn.ModuleList()
        
        node_configs = [
            ("Industrial vs Natural", [[0,1,8,9],[2,3,4,5,6,7]]),
            ("Sky vs Land", [[0,8],[1,9]]),
            ("Airplane vs Ship", [[0],[8]]),
            ("Car vs Truck", [[1],[9]]),
            ("Others vs Quadrupeds", [[2,6],[3,5],[4,7]]),
            ("Bird vs Frog", [[2],[6]]),
            ("Cat vs Dog", [[3],[5]]),
            ("Deer vs Horse", [[4],[7]])
        ]
        
        num_gpus = torch.cuda.device_count()
        for i, (name, judge) in enumerate(node_configs):
            device = f'cuda:{i % num_gpus}' if num_gpus > 0 else 'cpu'
            self.nodes.append(DecisionNode(name, judge).to(device))

    # def __init__(self):
    #     super(SequentialDecisionTree, self).__init__()
        
    #     self.nodes = nn.ModuleList([
    #         DecisionNode("Industrial vs Natural", judge=[[0,1,8,9],[2,3,4,5,6,7]]),
    #         DecisionNode("Sky vs Land", judge=[[0,8],[1,9]]),
    #         DecisionNode("Airplane vs Ship", judge=[[0],[8]]),
    #         DecisionNode("Car vs Truck", judge=[[1],[9]]),
    #         DecisionNode("Others vs Quadrupeds", judge=[[2,6],[3,5],[4,7]]),
    #         DecisionNode("Bird vs Frog", judge=[[2],[6]]),
    #         DecisionNode("Cat vs Dog", judge=[[3],[5]]),
    #         DecisionNode("Deer vs Horse", judge=[[4],[7]])
    #     ])
    
    def forward(self, x):
        global_vars.initialize_image_probabilities(x.size(0), is_training=self.training)
        outputs = [node(x) for node in self.nodes]
        return outputs
