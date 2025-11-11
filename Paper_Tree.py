import torch
import torch.nn as nn
from convmixer import ConvMixer


class DecisionNode(nn.Module):
    def __init__(self, model, judge):
        super().__init__()
        self.model = model
        self.judge = judge
    
    def forward(self, x):
        outputs = self.model(x)
        outputs = torch.sigmoid(outputs)
        return outputs


class SequentialDecisionTree(nn.Module):
    def __init__(self):
        super().__init__()
        self.isTree = True
        self.training_mode = 'normal'  # 'normal', 'record', 'pipeline'
        self.current_training_node = 0  # 当前训练的节点索引
        self.cached_outputs = {}  # 缓存其他节点的输出
        
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

    def set_training_mode(self, mode, node_idx=0):
        """设置训练模式
        Args:
            mode: 'normal' - 正常训练所有节点
                  'record' - 记录所有节点输出
                  'pipeline' - 流水线训练单个节点
            node_idx: 当mode为'pipeline'时，指定要训练的节点索引
        """
        self.training_mode = mode
        self.current_training_node = node_idx
        
        if mode == 'pipeline':
            # 在流水线模式下，只有当前训练的节点需要梯度
            for i, node in enumerate(self.nodes):
                for param in node.parameters():
                    param.requires_grad = (i == node_idx)
        else:
            # 其他模式下，所有节点都需要梯度
            for node in self.nodes:
                for param in node.parameters():
                    param.requires_grad = True

    def cache_outputs(self, batch_id, outputs_dict):
        """缓存某个batch的所有节点输出"""
        self.cached_outputs[batch_id] = outputs_dict

    def get_cached_output(self, batch_id, node_idx):
        """获取缓存的节点输出"""
        if batch_id in self.cached_outputs:
            return self.cached_outputs[batch_id].get(node_idx, None)
        return None

    def clear_cache(self):
        """清空缓存"""
        self.cached_outputs.clear()

    def forward(self, x, batch_id=None, node_idx=None):
        final_outputs = torch.ones(x.size(0), 10, device=x.device)
        
        if self.training_mode == 'record':
            # 记录模式：计算所有节点输出并缓存
            node_outputs = {}
            for i, node in enumerate(self.nodes):
                with torch.no_grad():
                    outputs = node(x)
                    node_outputs[i] = outputs.clone()
                    for j, class_indices in enumerate(node.judge):
                        final_outputs[:, class_indices] *= outputs[:, j].unsqueeze(1)
            
            if batch_id is not None:
                self.cache_outputs(batch_id, node_outputs)
                
        elif self.training_mode == 'pipeline':
            # 流水线模式：只训练当前节点，其他使用缓存（如无缓存则no_grad计算）
            for i, node in enumerate(self.nodes):
                if i == self.current_training_node:
                    # 当前训练的节点正常计算
                    outputs = node(x)
                else:
                    # 其他节点优先使用缓存的输出，不参与重新计算
                    cached = self.get_cached_output(batch_id, i) if batch_id is not None else None
                    if cached is not None:
                        outputs = cached
                    else:
                        # 若缓存缺失，退化为no_grad推理以保证完整性
                        with torch.no_grad():
                            outputs = node(x)
                
                for j, class_indices in enumerate(node.judge):
                    final_outputs[:, class_indices] *= outputs[:, j].unsqueeze(1)
        else:
            # 正常模式：所有节点都参与训练
            for node in self.nodes:
                outputs = node(x)
                for i, class_indices in enumerate(node.judge):
                    final_outputs[:, class_indices] *= outputs[:, i].unsqueeze(1)
        
        return final_outputs