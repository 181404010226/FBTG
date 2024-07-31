import torch
from Paper_TreeForTest import SequentialDecisionTree
from Paper_global_vars import global_vars
from Paper_DataSet import valid_data
import os
import shutil
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import torch.nn.functional as F

# 设置随机种子
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
root = os.path.join(os.path.dirname(__file__), "CIFAR10RawData")

def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

clear_directory('/root/autodl-tmp')


# 初始化模型并移至GPU
model = SequentialDecisionTree().to(device)

# 加载模型
model_path = 'best_models/model_epoch_443_acc_0.9642.pth'
model.load_state_dict(torch.load(model_path, map_location=device))


# 测试代码
model.eval()
total_correct = 0

confusion_dict = defaultdict(int)
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

sample_count = 0
fig = plt.figure(figsize=(30, 15))
gs = fig.add_gridspec(3, 3, width_ratios=[1, 2, 2])
axes = [[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(3)]
fig.tight_layout(pad=5.0)

with torch.no_grad():
    for batch_idx, (data, target) in enumerate(valid_data):
        data, target = data.to(device), target.to(device)
        model(data)
        
        for idx, true_label in enumerate(target):
            predicted_probs = global_vars.log_image_probabilities[idx]
            predicted_label = predicted_probs.argmax().item()
            is_correct = predicted_label == true_label.item()
            total_correct += is_correct

            if not is_correct:
                confusion_pair = (class_labels[true_label.item()], class_labels[predicted_label])
                confusion_dict[confusion_pair] += 1


            if not is_correct:
                row = sample_count % 3
                
                # Raw Image
                img = data[idx].cpu().permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min())
                axes[row][0].imshow(img)
                axes[row][0].set_title('Raw Image', fontsize=20)
                axes[row][0].axis('off')
                
                # Probability Distribution
                probs_np = predicted_probs.cpu().numpy()
                axes[row][1].bar(range(len(class_labels)), probs_np)
                axes[row][1].set_title('Probability Distribution', fontsize=20)
                axes[row][1].set_ylim(0, 1)
                axes[row][1].tick_params(axis='y', labelsize=16)
                axes[row][1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # Remove x-axis ticks
                
                
                # Node Probabilities
                node_names, all_probs = [], []
                
                def traverse_tree(nodes):
                    for node in nodes:
                        if node and idx in node.node_probabilities:
                            node_names.append(node.english_name)
                            probs = node.node_probabilities[idx][0]
                            all_probs.append(probs)

                traverse_tree(model.nodes)

                x = range(len(node_names))
                num_outputs = max(len(probs) for probs in all_probs)
                width = 0.8 / num_outputs

                axes[row][2].clear()  # Clear the existing subplot
                for i in range(num_outputs):
                    probs = [node_probs[i] if i < len(node_probs) else 0 for node_probs in all_probs]
                    axes[row][2].bar([pos + i * width for pos in x], probs, width, label=f'Output {i+1}')

                axes[row][2].set_title('Node Probabilities', fontsize=20)
                axes[row][2].set_ylim(0, 1)
                axes[row][2].tick_params(axis='y', labelsize=16)
                axes[row][2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # Remove x-axis ticks
                axes[row][2].legend(fontsize=12, loc='upper right')

                axes[row][0].set_title(f'Batch {batch_idx}, Sample {idx}\nTrue: {class_labels[true_label.item()]}, Pred: {class_labels[predicted_label]}', fontsize=20)

                sample_count += 1
                
                if sample_count % 3 == 0 or batch_idx == len(valid_data) - 1:
                    plt.savefig(f'/root/autodl-tmp/combined_analysis_{sample_count//3}.png', bbox_inches='tight', dpi=300)
                    plt.close(fig)
                    if batch_idx < len(valid_data) - 1:
                        fig = plt.figure(figsize=(30, 15))
                        gs = fig.add_gridspec(3, 3, width_ratios=[1, 2, 2])
                        axes = [[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(3)]
                        fig.tight_layout(pad=5.0)

    accuracy = total_correct / len(valid_data.dataset)
    print(f"Test Accuracy: {accuracy:.4f}")


# Create and save pie chart
total_confusions = sum(confusion_dict.values())
confusion_percentages = {k: v / total_confusions * 100 for k, v in confusion_dict.items()}

# Sort confusions by percentage and combine small categories
sorted_confusions = sorted(confusion_percentages.items(), key=lambda x: x[1], reverse=True)
pie_data = []
pie_labels = []
other_percentage = 0
threshold = 3

for (true_label, pred_label), percentage in sorted_confusions:
    if percentage >= threshold:
        pie_data.append(percentage)
        pie_labels.append(f"{true_label} → {pred_label}")
    else:
        other_percentage += percentage

if other_percentage > 0:
    pie_data.append(other_percentage)
    pie_labels.append("Others")

# Create a colors list
base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
colors = (base_colors * ((len(pie_data) - 1) // len(base_colors) + 1))[:len(pie_data) - 1]
if other_percentage > 0:
    colors.append('#999999')  # Gray color for "Others"

# 设置全局字体大小
plt.rcParams.update({'font.size': 36})  # 默认字体大小的两倍

plt.figure(figsize=(24, 16))  # 增加图形大小以适应更大的字体
wedges, texts, autotexts = plt.pie(pie_data, labels=None, autopct='%1.1f%%', startangle=90, 
                                   wedgeprops=dict(width=0.6), textprops=dict(color="k"),
                                   colors=colors)

# 增加自动百分比文本的字体大小
for autotext in autotexts:
    autotext.set_fontsize(20)

# Add lines connecting wedges to labels
for i, wedge in enumerate(wedges):
    ang = (wedge.theta2 + wedge.theta1) / 2
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = f"angle,angleA=0,angleB={ang}"
    
    plt.annotate(pie_labels[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                 horizontalalignment=horizontalalignment,
                 verticalalignment="center",
                 arrowprops=dict(arrowstyle="-", color="0.5",
                                 connectionstyle=connectionstyle),
                 fontsize=48)  # 增加注释文本的字体大小

plt.title(f"Confusion Distribution (>{threshold}%)", fontsize=48)  # 增加标题字体大小
plt.axis('equal')
plt.savefig(f'/root/autodl-tmp/confusion_pie_chart_{threshold}.png', bbox_inches='tight', dpi=300)  # 增加DPI以提高图像质量
plt.close()