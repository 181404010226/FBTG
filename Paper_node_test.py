import torch
from Paper_Tree import SequentialDecisionTree
from Paper_DataSetCIFAR import create_valid_loader
import os
from multiprocessing import freeze_support
# ... existing code ...
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

if __name__ == '__main__':
    freeze_support()
    # 设置随机种子
    seed = 42
    torch.manual_seed(seed)
    
    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型
    model = SequentialDecisionTree().to(device)
    model_path = 'checkpoint_epoch_496_acc_0.9642.pth'
    
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state']
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module._orig_mod.'):
            new_key = key[len('module._orig_mod.'):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    # 加载修改后的状态字典
    model.load_state_dict(new_state_dict)

    # 测试模式
    model.eval()
    node_correct = [0] * len(model.nodes)
    node_total = [0] * len(model.nodes)
    class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 创建每个节点的混淆矩阵
    node_confusion = [{} for _ in range(len(model.nodes))]
    
    valid_data = create_valid_loader()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(valid_data):
            data, target = data.to(device), target.to(device)
            
            # 测试每个节点
            for node_idx, node in enumerate(model.nodes):
                outputs = node(data)
                
                # 处理每个样本
                for idx, true_label in enumerate(target):
                    # 找出真实标签所属的组
                    true_group = -1
                    for group_idx, class_indices in enumerate(node.judge):
                        if true_label.item() in class_indices:
                            true_group = group_idx
                            break
                    
                    if true_group != -1:
                        node_total[node_idx] += 1
                        predicted_group = outputs[idx].argmax().item()
                        if predicted_group == true_group:
                            node_correct[node_idx] += 1
                        
                        # 记录混淆情况
                        confusion_key = (true_group, predicted_group)
                        node_confusion[node_idx][confusion_key] = \
                            node_confusion[node_idx].get(confusion_key, 0) + 1



        # 可视化每个节点的详细信息
        print("\n=== Node-wise Analysis ===")
        plt.figure(figsize=(20, 4*len(model.nodes)))
        
        for node_idx in range(len(model.nodes)):
            # 计算准确率
            accuracy = node_correct[node_idx] / node_total[node_idx] if node_total[node_idx] > 0 else 0
            
            # 创建混淆矩阵数组
            confusion = node_confusion[node_idx]
            num_groups = len(model.nodes[node_idx].judge)
            conf_matrix = np.zeros((num_groups, num_groups))
            for (true_group, pred_group), count in confusion.items():
                conf_matrix[true_group][pred_group] = count

            # 创建子图
            plt.subplot(len(model.nodes), 2, 2*node_idx + 1)
            
            # 绘制混淆矩阵热力图
            sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
            plt.title(f'Node {node_idx + 1} Confusion Matrix\nAccuracy: {accuracy:.4f}')
            plt.xlabel('Predicted Group')
            plt.ylabel('True Group')
            
            # 在右侧添加类别分组信息
            plt.subplot(len(model.nodes), 2, 2*node_idx + 2)
            plt.axis('off')
            group_info = f"Node {node_idx + 1} Groups:\n\n"
            for group_idx, classes in enumerate(model.nodes[node_idx].judge):
                class_names = [class_labels[i] for i in classes]
                group_info += f"Group {group_idx}: {class_names}\n"
            plt.text(0, 0.5, group_info, fontsize=10, verticalalignment='center')

        plt.tight_layout()
        plt.savefig('node_analysis.png')
        plt.close()
        
        # 节点准确率条形图
        plt.figure(figsize=(12, 6))
        accuracies = [node_correct[i]/node_total[i] if node_total[i] > 0 else 0 
                     for i in range(len(model.nodes))]
        plt.bar(range(1, len(model.nodes) + 1), accuracies)
        plt.title('Node-wise Accuracy')
        plt.xlabel('Node')
        plt.ylabel('Accuracy')
        # 调整y轴的范围，给标签留出足够空间
        plt.ylim(0, 1.05)
        # 调整文本标签的位置和格式
        for i, acc in enumerate(accuracies):
            plt.text(i + 1, acc - 0.03, f'{acc:.4f}', ha='center', va='top', color='white')
        plt.savefig('node_accuracies.png', bbox_inches='tight')
        plt.close()

        # 节点分析图

        # 配置参数
        plot_config = {
            'figure_width': 8,          # 图片宽度
            'height_per_node': 2,        # 每个节点的高度
            'title_fontsize': 10,        # 标题字体大小
            'label_fontsize': 9,        # 轴标签字体大小
            'matrix_fontsize': 9,       # 混淆矩阵数字大小
            'group_info_fontsize': 9,   # 分组信息字体大小
            'dpi': 300                   # 图片分辨率
        }

        # 节点分析图
        plt.figure(figsize=(plot_config['figure_width'], 
                          plot_config['height_per_node'] * len(model.nodes)))
        
        for node_idx in range(len(model.nodes)):
            accuracy = node_correct[node_idx] / node_total[node_idx] if node_total[node_idx] > 0 else 0
            
            confusion = node_confusion[node_idx]
            num_groups = len(model.nodes[node_idx].judge)
            conf_matrix = np.zeros((num_groups, num_groups))
            for (true_group, pred_group), count in confusion.items():
                conf_matrix[true_group][pred_group] = count

            # 混淆矩阵子图
            ax1 = plt.subplot(len(model.nodes), 2, 2*node_idx + 1)
            sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues',
                       square=True, annot_kws={'size': plot_config['matrix_fontsize']})
            plt.title(f'Node {node_idx + 1} Confusion Matrix\nAccuracy: {accuracy:.4f}',
                     fontsize=plot_config['title_fontsize'])
            plt.xlabel('Predicted Group', fontsize=plot_config['label_fontsize'])
            plt.ylabel('True Group', fontsize=plot_config['label_fontsize'])
            
            # 文本区域子图
            ax2 = plt.subplot(len(model.nodes), 2, 2*node_idx + 2)
            plt.axis('off')
            
            # 准备文本内容
            group_info = f"Node {node_idx + 1} Groups:\n\n"
            group_lines = []
            for group_idx, classes in enumerate(model.nodes[node_idx].judge):
                class_names = [class_labels[i] for i in classes]
                line = f"Group {group_idx}: {class_names}"
                group_lines.append(line)
            group_info += '\n'.join(group_lines)
            
            # 添加文本并调整位置
            text = plt.text(0, 0.5, group_info,
                          fontsize=plot_config['group_info_fontsize'],
                          verticalalignment='center',
                          horizontalalignment='left')
            
            bbox = text.get_window_extent(renderer=plt.gcf().canvas.get_renderer())
            bbox_data = bbox.transformed(ax2.transData.inverted())
            
            text_width = bbox_data.width
            ax2.set_position([0.55, ax2.get_position().y0,
                            min(0.4, text_width + 0.05),
                            ax2.get_position().height])

        plt.tight_layout()
        plt.savefig('node_analysis.png',
                   bbox_inches='tight',
                   dpi=plot_config['dpi'])
        plt.close()