#!/usr/bin/env python3
"""
检查经典流行模型的参数量
重点关注50M参数量左右的模型
"""

import torch
import torch.nn as nn
import timm

def count_parameters(model: nn.Module) -> float:
    """计算模型参数量（单位：百万）"""
    return sum(p.numel() for p in model.parameters()) / 1e6

def create_timm_model(model_name: str, num_classes: int = 1000) -> nn.Module:
    """创建timm模型"""
    try:
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        return model
    except Exception as e:
        print(f"创建模型 {model_name} 失败: {e}")
        return None

def main():
    """主函数"""
    print("经典流行深度学习模型参数量对比")
    print("=" * 60)
    print("重点关注50M参数量左右的模型")
    print("=" * 60)
    
    # 50M参数量左右的经典流行模型
    models_around_50M = [
        # ResNet系列 (经典CNN架构)
        'resnet50',           # ~25.6M (著名的50层ResNet)
        'resnet101',          # ~44.5M (101层ResNet)
        'resnet152',          # ~60.2M (152层ResNet)
        
        # VGG系列 (经典深层CNN)
        'vgg16',              # ~138M (VGG-16，经典架构)
        'vgg19',              # ~144M (VGG-19，更深的VGG)
        
        # Inception系列 (Google提出的多尺度卷积)
        'inception_v3',       # ~23.8M (Inception v3)
        'inception_v4',       # ~42.7M (Inception v4)
        
        # DenseNet系列 (密集连接网络)
        'densenet121',        # ~8.0M (DenseNet-121)
        'densenet169',        # ~14.1M (DenseNet-169)
        'densenet201',        # ~20.0M (DenseNet-201)
        'densenet161',        # ~28.7M (DenseNet-161)
        
        # EfficientNet系列 (高效网络)
        'efficientnet_b0',    # ~5.3M (EfficientNet-B0)
        'efficientnet_b1',    # ~7.8M (EfficientNet-B1)
        'efficientnet_b2',    # ~9.2M (EfficientNet-B2)
        'efficientnet_b3',    # ~12.2M (EfficientNet-B3)
        'efficientnet_b4',    # ~19.3M (EfficientNet-B4)
        'efficientnet_b5',    # ~30.4M (EfficientNet-B5)
        'efficientnet_b6',    # ~43.0M (EfficientNet-B6)
        
        # MobileNet系列 (移动端高效网络)
        'mobilenetv2_100',    # ~3.5M (MobileNet v2)
        'mobilenetv3_large_100', # ~5.5M (MobileNet v3 Large)
        'mobilenetv3_small_100', # ~2.5M (MobileNet v3 Small)
        
        # ResNeXt系列 (分组卷积)
        'resnext50_32x4d',    # ~25.0M (ResNeXt-50)
        'resnext101_32x8d',   # ~88.8M (ResNeXt-101)
        
        # RegNet系列 (规则化网络)
        'regnetx_002',        # ~2.7M (RegNetX-200MF)
        'regnetx_004',        # ~5.2M (RegNetX-400MF)
        'regnetx_006',        # ~6.2M (RegNetX-600MF)
        'regnetx_008',        # ~7.3M (RegNetX-800MF)
        'regnetx_016',        # ~9.2M (RegNetX-1.6GF)
        'regnetx_032',        # ~15.3M (RegNetX-3.2GF)
        'regnetx_040',        # ~22.1M (RegNetX-4.0GF)
        'regnetx_064',        # ~26.2M (RegNetX-6.4GF)
        'regnetx_080',        # ~39.6M (RegNetX-8.0GF)
        'regnetx_120',        # ~46.1M (RegNetX-12GF)
        'regnetx_160',        # ~54.3M (RegNetX-16GF)
        'regnetx_320',        # ~107M (RegNetX-32GF)
        
        # 其他经典架构
        'squeezenet1_0',      # ~1.2M (SqueezeNet)
        'squeezenet1_1',      # ~1.2M (SqueezeNet v1.1)
        'shufflenet_v2_x1_0', # ~2.3M (ShuffleNet v2)
        'wide_resnet50_2',    # ~68.9M (Wide ResNet)
        
        # Vision Transformer系列 (如果可用)
        'vit_base_patch16_224', # ~86.6M (ViT Base)
        'vit_small_patch16_224', # ~22.1M (ViT Small)
        
        # Xception (深度可分离卷积)
        'xception',           # ~22.9M (Xception)
        
        # GhostNet (幽灵模块)
        'ghostnet_100',       # ~5.2M (GhostNet)
    ]
    
    # 测试不同的配置
    configs = [
        {"num_classes": 1000, "dataset": "ImageNet"},
        {"num_classes": 10, "dataset": "CIFAR-10"},
        {"num_classes": 100, "dataset": "CIFAR-100"}
    ]
    
    for config in configs:
        print(f"\n{config['dataset']} 配置 (num_classes={config['num_classes']}):")
        print("-" * 50)
        print(f"{'模型名称':<25} {'参数量(M)':<12} {'备注'}")
        print("-" * 50)
        
        # 存储结果用于排序
        results = []
        
        for model_name in models_around_50M:
            model = create_timm_model(model_name, config['num_classes'])
            if model is not None:
                param_count = count_parameters(model)
                results.append((model_name, param_count))
        
        # 按参数量排序
        results.sort(key=lambda x: x[1])
        
        # 分类显示
        print("\n🔹 轻量级模型 (< 10M):")
        for model_name, param_count in results:
            if param_count < 10:
                comment = ""
                if 'mobile' in model_name.lower():
                    comment = "(移动端优化)"
                elif 'efficient' in model_name.lower():
                    comment = "(高效架构)"
                elif 'squeeze' in model_name.lower():
                    comment = "(压缩网络)"
                elif 'ghost' in model_name.lower():
                    comment = "(幽灵模块)"
                elif 'shuffle' in model_name.lower():
                    comment = "(通道混洗)"
                print(f"{model_name:<25} {param_count:>8.2f}      {comment}")
        
        print("\n🔸 中等规模模型 (10M - 30M):")
        for model_name, param_count in results:
            if 10 <= param_count < 30:
                comment = ""
                if 'resnet' in model_name.lower():
                    comment = "(经典ResNet)"
                elif 'dense' in model_name.lower():
                    comment = "(密集连接)"
                elif 'inception' in model_name.lower():
                    comment = "(多尺度卷积)"
                elif 'efficient' in model_name.lower():
                    comment = "(高效网络)"
                elif 'vit' in model_name.lower():
                    comment = "(Vision Transformer)"
                elif 'xception' in model_name.lower():
                    comment = "(深度可分离卷积)"
                print(f"{model_name:<25} {param_count:>8.2f}      {comment}")
        
        print("\n🔺 大规模模型 (30M - 80M):")
        for model_name, param_count in results:
            if 30 <= param_count < 80:
                comment = ""
                if 'resnet' in model_name.lower():
                    comment = "(深层ResNet)"
                elif 'efficient' in model_name.lower():
                    comment = "(高效大模型)"
                elif 'wide' in model_name.lower():
                    comment = "(宽网络)"
                elif 'resnext' in model_name.lower():
                    comment = "(分组卷积)"
                elif 'regnet' in model_name.lower():
                    comment = "(规则化网络)"
                elif 'vit' in model_name.lower():
                    comment = "(Vision Transformer)"
                print(f"{model_name:<25} {param_count:>8.2f}      {comment}")
        
        print("\n🔴 超大模型 (≥ 80M):")
        for model_name, param_count in results:
            if param_count >= 80:
                comment = ""
                if 'vgg' in model_name.lower():
                    comment = "(经典VGG，参数量大)"
                elif 'resnext' in model_name.lower():
                    comment = "(深层分组卷积)"
                elif 'regnet' in model_name.lower():
                    comment = "(超大规则网络)"
                elif 'vit' in model_name.lower():
                    comment = "(大型Transformer)"
                print(f"{model_name:<25} {param_count:>8.2f}      {comment}")
    
    print("\n" + "=" * 60)
    print("📊 50M参数量左右的推荐模型:")
    print("- RegNetX-12GF: ~46M 参数 (高效规则网络)")
    print("- ResNet-152: ~60M 参数 (经典深层网络)")
    print("- EfficientNet-B6: ~43M 参数 (高效网络)")
    print("- ResNeXt-50: ~25M 参数 (分组卷积)")
    print("- Inception-v4: ~43M 参数 (多尺度卷积)")
    print("=" * 60)
    print("💡 模型选择建议:")
    print("- 移动端部署: MobileNet系列 (< 10M)")
    print("- 平衡性能: ResNet-50, EfficientNet-B4-B6")
    print("- 高精度需求: ResNet-152, Inception-v4")
    print("- 创新架构: RegNet, Vision Transformer")

if __name__ == '__main__':
    main() 