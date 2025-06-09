 #!/usr/bin/env python3
"""
模型性能对比训练系统
对比NeuronBundle模型与timm库中经典模型的性能
支持CIFAR-10和CIFAR-100数据集
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import timm
import time
import os
import json
import logging
from typing import Dict, List, Tuple, Optional
import argparse
from collections import defaultdict
import pandas as pd
import numpy as np

# 导入你的NeuronBundle模型
from Paper_NeuronBundle import create_convmixer

class ModelComparison:
    """模型性能对比类"""
    
    def __init__(self, dataset_name: str = 'cifar10', device: str = 'auto'):
        self.dataset_name = dataset_name
        self.device = self._get_device(device)
        self.results = defaultdict(dict)
        
        # 设置日志
        self.setup_logging()
        
        # 数据集配置
        self.num_classes = 10 if dataset_name == 'cifar10' else 100
        self.img_size = 32
        
        # 训练配置
        self.epochs = 300
        self.batch_size = 128
        self.learning_rate = 0.001  # 降低学习率
        
        # Mixup配置
        self.mixup_alpha = 0.2  # Mixup参数
        
        # 模型配置字典 (优化显存使用，移除大模型)
        self.model_configs = {
            'NeuronBundleNet': {
                'constructor': self._create_neuron_bundle,
                'params': {'dim': 64, 'depth': 4, 'num_bundles': 16, 'kernel_size': 5},
                'batch_size': 256
            },
            'ResNet18': {
                'constructor': self._create_timm_model,
                'params': {'model_name': 'resnet18', 'pretrained': False},
                'batch_size': 256
            },
            'EfficientNet-B2': {
                'constructor': self._create_timm_model,
                'params': {'model_name': 'efficientnet_b2', 'pretrained': False},
                'batch_size': 256
            },
            'MobileNetV2-140': {
                'constructor': self._create_timm_model,
                'params': {'model_name': 'mobilenetv2_140', 'pretrained': False},
                'batch_size': 256
            },
            'MobileNetV3-Large': {
                'constructor': self._create_timm_model,
                'params': {'model_name': 'mobilenetv3_large_100', 'pretrained': False},
                'batch_size': 256
            },
            'RegNetX-800MF': {
                'constructor': self._create_timm_model,
                'params': {'model_name': 'regnetx_800mf', 'pretrained': False},
                'batch_size': 256
            },
            'DenseNet121': {
                'constructor': self._create_timm_model,
                'params': {'model_name': 'densenet121', 'pretrained': False},
                'batch_size': 256
            },
            'GhostNet-130': {
                'constructor': self._create_timm_model,
                'params': {'model_name': 'ghostnet_130', 'pretrained': False},
                'batch_size': 256
            }
        }
        
    def _get_device(self, device: str) -> torch.device:
        """获取计算设备"""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(device)
    
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'model_comparison_{self.dataset_name}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _create_neuron_bundle(self, **params) -> nn.Module:
        """创建NeuronBundle模型"""
        model = create_convmixer(
            dim=params['dim'],
            depth=params['depth'],
            num_bundles=params['num_bundles'],
            kernel_size=params['kernel_size'],
            patch_size=1,  # 适配CIFAR小图像
            num_classes=self.num_classes
        )
        return model
    
    def _create_timm_model(self, model_name: str, pretrained: bool = False, **kwargs) -> nn.Module:
        """创建timm模型"""
        try:
            # 对于CIFAR数据集，需要调整输入尺寸
            if 'vit' in model_name:
                model = timm.create_model(
                    model_name, 
                    pretrained=pretrained, 
                    num_classes=self.num_classes,
                    img_size=self.img_size
                )
            else:
                model = timm.create_model(
                    model_name, 
                    pretrained=pretrained, 
                    num_classes=self.num_classes
                )
            
            # 对于某些模型，可能需要调整第一层以适应32x32输入
            if hasattr(model, 'conv1') and model.conv1.kernel_size[0] == 7:
                # ResNet系列模型调整
                model.conv1 = nn.Conv2d(3, model.conv1.out_channels, 
                                      kernel_size=3, stride=1, padding=1, bias=False)
                if hasattr(model, 'maxpool'):
                    model.maxpool = nn.Identity()
            
            return model
        except Exception as e:
            self.logger.error(f"创建模型 {model_name} 失败: {e}")
            return None
    
    def get_data_loaders(self, batch_size: Optional[int] = None) -> Tuple[DataLoader, DataLoader]:
        """获取数据加载器"""
        if batch_size is None:
            batch_size = self.batch_size
            
        # 数据增强配置
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            if self.dataset_name == 'cifar10' else
            transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762))
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            if self.dataset_name == 'cifar10' else
            transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762))
        ])
        
        # 下载数据集
        if self.dataset_name == 'cifar10':
            train_dataset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=train_transform)
            test_dataset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=test_transform)
        else:
            train_dataset = torchvision.datasets.CIFAR100(
                root='./data', train=True, download=True, transform=train_transform)
            test_dataset = torchvision.datasets.CIFAR100(
                root='./data', train=False, download=True, transform=test_transform)
        
        # 调整worker数量以节省内存
        num_workers = min(4, os.cpu_count() or 1)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                               shuffle=False, num_workers=num_workers, pin_memory=True)
        
        self.logger.info(f"数据加载器创建完成 - 批量大小: {batch_size}, Workers: {num_workers}")
        return train_loader, test_loader
    
    def count_parameters(self, model: nn.Module) -> float:
        """计算模型参数量（单位：百万）"""
        return sum(p.numel() for p in model.parameters()) / 1e6
    
    def mixup_data(self, x, y, alpha=1.0):
        """Mixup数据增强"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """Mixup损失函数"""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
    def monitor_gpu_memory(self):
        """监控GPU显存使用情况"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9  # GB
            reserved = torch.cuda.memory_reserved() / 1e9    # GB
            self.logger.info(f"GPU内存使用: {allocated:.2f}GB 已分配, {reserved:.2f}GB 已保留")
            return allocated, reserved
        return 0, 0

    def get_model_batch_size(self, model_name: str) -> int:
        """获取模型对应的批量大小"""
        if model_name in self.model_configs and 'batch_size' in self.model_configs[model_name]:
            return self.model_configs[model_name]['batch_size']
        return self.batch_size

    def train_model(self, model: nn.Module, model_name: str, 
                   train_loader: DataLoader, test_loader: DataLoader) -> Dict:
        """训练单个模型"""
        self.logger.info(f"开始训练模型: {model_name}")
        self.logger.info(f"训练设置 - 学习率: {self.learning_rate}, Mixup Alpha: {self.mixup_alpha}")
        
        # 监控显存
        self.monitor_gpu_memory()
        
        model = model.to(self.device)
        
        # 优化器和调度器
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, 
                            momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-6)
        
        criterion = nn.CrossEntropyLoss()
        
        # 训练统计
        best_acc = 0
        train_losses = []
        train_accs = []
        test_accs = []
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            # 训练阶段
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # 应用Mixup数据增强
                if self.mixup_alpha > 0:
                    inputs, targets_a, targets_b, lam = self.mixup_data(inputs, targets, self.mixup_alpha)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = self.mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    # 对于Mixup，精度计算稍有不同
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += (lam * predicted.eq(targets_a).float() + 
                              (1 - lam) * predicted.eq(targets_b).float()).sum().item()
                else:
                    # 原始训练方式
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                
                if batch_idx % 100 == 0:
                    self.logger.info(f'{model_name} - Epoch: {epoch+1}/{self.epochs}, '
                                   f'Batch: {batch_idx}/{len(train_loader)}, '
                                   f'Loss: {loss.item():.4f}, '
                                   f'Acc: {100.*correct/total:.2f}%')
            
            train_acc = 100. * correct / total
            train_losses.append(running_loss / len(train_loader))
            train_accs.append(train_acc)
            
            # 测试阶段
            test_acc, test_acc_top5 = self.evaluate_model(model, test_loader)
            test_accs.append(test_acc)
            
            # 更新最佳精度
            if test_acc > best_acc:
                best_acc = test_acc
                # 保存最佳模型
                torch.save(model.state_dict(), f'best_{model_name}_{self.dataset_name}.pth')
            
            scheduler.step()
            
            # 每10个epoch输出一次进度
            if (epoch + 1) % 10 == 0:
                self.logger.info(f'{model_name} - Epoch {epoch+1}: '
                               f'Train Acc: {train_acc:.2f}%, '
                               f'Test Acc: {test_acc:.2f}%, '
                               f'Best Acc: {best_acc:.2f}%')
        
        training_time = time.time() - start_time
        
        # 最终评估
        model.load_state_dict(torch.load(f'best_{model_name}_{self.dataset_name}.pth'))
        final_acc, final_acc_top5 = self.evaluate_model(model, test_loader)
        
        # 计算参数量
        param_count = self.count_parameters(model)
        
        result = {
            'model_name': model_name,
            'dataset': self.dataset_name,
            'param_count_M': round(param_count, 2),
            'best_test_acc': round(best_acc, 2),
            'final_test_acc': round(final_acc, 2),
            'final_test_acc_top5': round(final_acc_top5, 2) if final_acc_top5 else None,
            'training_time_hours': round(training_time / 3600, 2),
            'epochs': self.epochs,
            'train_history': {
                'train_losses': train_losses,
                'train_accs': train_accs,
                'test_accs': test_accs
            }
        }
        
        self.logger.info(f'{model_name} 训练完成 - '
                        f'参数量: {param_count:.2f}M, '
                        f'最佳精度: {best_acc:.2f}%, '
                        f'训练时间: {training_time/3600:.2f}小时')
        
        return result
    
    def evaluate_model(self, model: nn.Module, test_loader: DataLoader) -> Tuple[float, Optional[float]]:
        """评估模型"""
        model.eval()
        correct = 0
        correct_top5 = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                
                # Top-1 accuracy
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Top-5 accuracy (仅对CIFAR-100计算)
                if self.num_classes == 100:
                    _, pred_top5 = outputs.topk(5, 1, True, True)
                    pred_top5 = pred_top5.t()
                    correct_top5 += pred_top5.eq(targets.view(1, -1).expand_as(pred_top5)).sum().item()
        
        acc = 100. * correct / total
        acc_top5 = 100. * correct_top5 / total if self.num_classes == 100 else None
        
        return acc, acc_top5
    
    def run_comparison(self, models_to_test: Optional[List[str]] = None) -> Dict:
        """运行模型对比"""
        if models_to_test is None:
            models_to_test = list(self.model_configs.keys())
        
        self.logger.info(f"开始模型对比实验 - 数据集: {self.dataset_name.upper()}")
        self.logger.info(f"将测试以下模型: {', '.join(models_to_test)}")
        self.logger.info(f"训练配置 - 学习率: {self.learning_rate}, Mixup Alpha: {self.mixup_alpha}, 训练轮数: {self.epochs}")
        
        results = []
        
        for model_name in models_to_test:
            if model_name not in self.model_configs:
                self.logger.warning(f"未知模型: {model_name}, 跳过")
                continue
            
            try:
                # 获取模型特定的批量大小
                model_batch_size = self.get_model_batch_size(model_name)
                self.logger.info(f"{model_name} 使用批量大小: {model_batch_size}")
                
                # 为每个模型创建独立的数据加载器
                train_loader, test_loader = self.get_data_loaders(model_batch_size)
                
                # 创建模型
                config = self.model_configs[model_name]
                model = config['constructor'](**config['params'])
                
                if model is None:
                    self.logger.error(f"模型 {model_name} 创建失败，跳过")
                    continue
                
                # 训练模型
                result = self.train_model(model, model_name, train_loader, test_loader)
                results.append(result)
                
                # 清理GPU内存
                del model, train_loader, test_loader
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # 显存监控
                self.monitor_gpu_memory()
                
            except Exception as e:
                self.logger.error(f"训练模型 {model_name} 时出错: {e}")
                # 清理内存后继续
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                continue
        
        # 保存结果
        self.save_results(results)
        
        # 生成对比表格
        self.generate_comparison_table(results)
        
        return results
    
    def save_results(self, results: List[Dict]):
        """保存结果到文件"""
        # 保存详细结果到JSON
        with open(f'model_comparison_results_{self.dataset_name}.json', 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 保存简化结果到CSV
        df_data = []
        for result in results:
            df_data.append({
                '数据集': result['dataset'].upper(),
                '模型': result['model_name'],
                '参数量(M)': result['param_count_M'],
                'Top-1 Acc(%)': result['final_test_acc'],
                'Top-5 Acc(%)': result['final_test_acc_top5'] if result['final_test_acc_top5'] else '-',
                '训练轮数': result['epochs'],
                '训练时间(h)': result['training_time_hours'],
                '备注': '已完成'
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(f'model_comparison_summary_{self.dataset_name}.csv', 
                  index=False, encoding='utf-8-sig')
        
        self.logger.info(f"结果已保存到文件")
    
    def generate_comparison_table(self, results: List[Dict]):
        """生成对比表格"""
        print(f"\n{'='*80}")
        print(f"模型性能对比结果 - {self.dataset_name.upper()}数据集")
        print(f"{'='*80}")
        
        # 表格头
        if self.num_classes == 100:
            header = f"{'模型':<20} {'参数量(M)':<12} {'Top-1 Acc(%)':<15} {'Top-5 Acc(%)':<15} {'训练时间(h)':<12} {'备注':<10}"
        else:
            header = f"{'模型':<20} {'参数量(M)':<12} {'Top-1 Acc(%)':<15} {'训练时间(h)':<12} {'备注':<10}"
        
        print(header)
        print('-' * len(header))
        
        # 按精度排序
        sorted_results = sorted(results, key=lambda x: x['final_test_acc'], reverse=True)
        
        for result in sorted_results:
            if self.num_classes == 100:
                top5_str = f"{result['final_test_acc_top5']:.2f}" if result['final_test_acc_top5'] else "-"
                row = f"{result['model_name']:<20} {result['param_count_M']:<12} {result['final_test_acc']:<15.2f} {top5_str:<15} {result['training_time_hours']:<12.2f} {'已完成':<10}"
            else:
                row = f"{result['model_name']:<20} {result['param_count_M']:<12} {result['final_test_acc']:<15.2f} {result['training_time_hours']:<12.2f} {'已完成':<10}"
            
            print(row)
        
        print(f"{'='*80}")
        
        # 显示最佳结果
        best_result = max(results, key=lambda x: x['final_test_acc'])
        print(f"\n🏆 最佳结果: {best_result['model_name']} - {best_result['final_test_acc']:.2f}%")
        print(f"   参数量: {best_result['param_count_M']:.2f}M")
        print(f"   训练时间: {best_result['training_time_hours']:.2f}小时")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='模型性能对比')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['cifar10', 'cifar100'], help='数据集选择')
    parser.add_argument('--models', type=str, nargs='+', 
                       help='要测试的模型列表')
    parser.add_argument('--epochs', type=int, default=300, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=128, help='批量大小')
    parser.add_argument('--device', type=str, default='auto', help='计算设备')
    parser.add_argument('--mixup-alpha', type=float, default=0.2, help='Mixup alpha参数')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='学习率')
    
    args = parser.parse_args()
    
    # 创建对比实验
    comparison = ModelComparison(dataset_name=args.dataset, device=args.device)
    comparison.epochs = args.epochs
    comparison.batch_size = args.batch_size
    comparison.learning_rate = args.learning_rate
    comparison.mixup_alpha = args.mixup_alpha
    
    # 运行对比
    results = comparison.run_comparison(models_to_test=args.models)
    
    print(f"\n实验完成！结果已保存到文件中。")


if __name__ == '__main__':
    main()