import torch
import torch.nn as nn
import torch.optim as optim
from Paper_global_vars import global_vars
from convmixer import ConvMixer
from resnet import ResNet8
import numpy as np
import timm
from RDNet import rdnet_tiny

class DecisionNode(nn.Module):
    def __init__(self, model, judge=[-1,-1]):
        super(DecisionNode, self).__init__()
        self.model = model
        self.judge = judge
    
    def forward(self, x):
        outputs = self.model(x)
        outputs = torch.sigmoid(outputs)
        return outputs
    
class SequentialDecisionTreeForRDNet(nn.Module):
    def __init__(self, isTest=False):
        super(SequentialDecisionTreeForRDNet, self).__init__()
        self.isTree = True
        self.debug = True
        self.isTest = isTest
     
        self.nodes = nn.ModuleList([
            DecisionNode(self.create_rdnet(2), judge=[[0,1,8,9],[2,3,4,5,6,7]]),
            DecisionNode(self.create_rdnet(2), judge=[[0,8],[1,9]]),
            DecisionNode(self.create_rdnet(2), judge=[[0],[8]]),
            DecisionNode(self.create_rdnet(2), judge=[[1],[9]]),
            DecisionNode(self.create_rdnet(3), judge=[[2,6],[3,5],[4,7]]),
            DecisionNode(self.create_rdnet(2), judge=[[2],[6]]),
            DecisionNode(self.create_rdnet(2), judge=[[3],[5]]),
            DecisionNode(self.create_rdnet(2), judge=[[4],[7]])
        ])
    
    def create_rdnet(self, num_classes):
        model = timm.create_model('rdnet_tiny', pretrained=False, num_classes=num_classes)
        if self.isTest==False:
            local_pretrained_path = 'rdnet_tiny/pytorch_model.bin'
            state_dict = torch.load(local_pretrained_path)
            
            for key in ['head.fc.weight', 'head.fc.bias']:
                if key in state_dict:
                    del state_dict[key]
    
            model.load_state_dict(state_dict, strict=False)

        model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
        return model
    
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

class SequentialDecisionTree(nn.Module):
    def __init__(self):
        super(SequentialDecisionTree, self).__init__()
        self.isTree = True
        
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
        final_outputs = torch.ones(x.size(0), 10, device=x.device)
        
        for node in self.nodes:
            outputs = node(x)
            for i, class_indices in enumerate(node.judge):
                final_outputs[:, class_indices] *= outputs[:, i].unsqueeze(1)
        
        return final_outputs

class SequentialDecisionTreeCIFAR100ForRDNet(nn.Module):
    def __init__(self,isTest=False):
        super(SequentialDecisionTreeCIFAR100ForRDNet, self).__init__()
        self.isTree = True
        self.isTest=isTest
        self.debug = True
        # 第一层节点：区分20个大类
        self.root_node = DecisionNode(self.create_rdnet(20))
        
        # 创建20个子节点，每个对应一个大类
        self.sub_nodes = nn.ModuleList([
            DecisionNode(self.create_rdnet(5))
            for _ in range(20)
        ])
        
        # Define the correct CIFAR-100 coarse to fine mapping
        self.coarse_labels = [
            [4, 30, 55, 72, 95],  # aquatic mammals
            [1, 32, 67, 73, 91],  # fish
            [54, 62, 70, 82, 92],  # flowers
            [9, 10, 16, 28, 61],  # food containers
            [0, 51, 53, 57, 83],  # fruit and vegetables
            [22, 39, 40, 86, 87],  # household electrical devices
            [5, 20, 25, 84, 94],  # household furniture
            [6, 7, 14, 18, 24],   # insects
            [3, 42, 43, 88, 97],  # large carnivores
            [12, 17, 37, 68, 76], # large man-made outdoor things
            [23, 33, 49, 60, 71], # large natural outdoor scenes
            [15, 19, 21, 31, 38], # large omnivores and herbivores
            [34, 63, 64, 66, 75], # medium-sized mammals
            [26, 45, 77, 79, 99], # non-insect invertebrates
            [2, 11, 35, 46, 98],  # people
            [27, 29, 44, 78, 93], # reptiles
            [36, 50, 65, 74, 80], # small mammals
            [47, 52, 56, 59, 96], # trees
            [8, 13, 48, 58, 90],  # vehicles 1
            [41, 69, 81, 85, 89]  # vehicles 2
        ]

    def create_rdnet(self, num_classes):
        model = timm.create_model('rdnet_tiny', pretrained=False, num_classes=num_classes)
        if self.isTest==False:
            local_pretrained_path = 'rdnet_tiny/pytorch_model.bin'
            state_dict = torch.load(local_pretrained_path)
            
            for key in ['head.fc.weight', 'head.fc.bias']:
                if key in state_dict:
                    del state_dict[key]

            model.load_state_dict(state_dict, strict=False)

        model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
        return model
    
    def forward(self, x):
        # 获取大类的概率分布
        coarse_probs = self.root_node(x)
        
        # 对每个大类，获取其小类的概率分布
        fine_probs = torch.stack([sub_node(x) for sub_node in self.sub_nodes], dim=1)
        
        # 创建用于存储重新排列后的概率的张量
        reordered_probs = torch.zeros(x.size(0), 100, device=x.device)
        
        # 重新排列概率以匹配原始的100个类别
        if self.debug:
            print("Mapping results:")
        for coarse_idx, fine_indices in enumerate(self.coarse_labels):
            for fine_idx, class_idx in enumerate(fine_indices):
                reordered_probs[:, class_idx] = fine_probs[:, coarse_idx, fine_idx]
                if self.debug:
                    print(f"Class {class_idx}: fine_probs[{coarse_idx}, {fine_idx}] -> reordered_probs[{class_idx}]")
        self.debug = False    
        
  
        # 展平并扩大5倍的coarse_probs
        expanded_coarse_probs = coarse_probs.unsqueeze(2).expand(-1, -1, 5).reshape(x.size(0), 100)
        
        # 和reordered_probs相乘
        final_probs = expanded_coarse_probs * reordered_probs
        
        return final_probs
    

class SequentialDecisionTreeCIFAR100(nn.Module):
    def __init__(self):
        super(SequentialDecisionTreeCIFAR100, self).__init__()
        self.isTree = True
        
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