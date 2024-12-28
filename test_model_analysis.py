import torch
import torch.nn.functional as F
import os
from Paper_global_vars import global_vars
from Paper_DataSetCIFAR import create_valid_loader
from Paper_NeuronBundle import create_convmixer
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import seaborn as sns

def analyze_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = create_convmixer(64, 4, 32, 5, 4, 10).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    # 创建验证数据加载器
    valid_loader = create_valid_loader(global_vars.dataset, distributed=False)
    
    # 用于存储每个类别的统计信息
    class_correct = [0] * 10
    class_total = [0] * 10
    
    # 用于存储错误分类的图片
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []
    
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            
            # 修改这里：target现在已经是正确的标签格式
            correct = predicted.eq(target)  # 移除 argmax
            
            # 更新每个类别的统计信息
            for i in range(len(target)):
                label = target[i].item()  # 直接使用target，不需要argmax
                class_total[label] += 1
                if correct[i]:
                    class_correct[label] += 1
                else:
                    # 存储错误分类的图片信息
                    misclassified_images.append(data[i].cpu())
                    misclassified_labels.append(label)
                    misclassified_preds.append(predicted[i].item())

    # 计算并打印每个类别的准确率
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck']
    
    print("\nPer-class Accuracy:")
    for i in range(10):
        accuracy = 100 * class_correct[i] / class_total[i]
        print(f'{class_names[i]}: {accuracy:.2f}% ({class_correct[i]}/{class_total[i]})')
    
    # 绘制混淆矩阵
    plt.figure(figsize=(12, 8))
    accuracies = [100 * correct / total for correct, total in zip(class_correct, class_total)]
    sns.barplot(x=class_names, y=accuracies)
    plt.title('Per-class Accuracy')
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('class_accuracies.png')
    
    # 保存错误分类的图片
    if len(misclassified_images) > 0:
        os.makedirs('misclassified', exist_ok=True)
        
        # 反归一化转换
        denormalize = transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.],
                              std=[1/0.2023, 1/0.1994, 1/0.2010]),
            transforms.Normalize(mean=[-0.4914, -0.4822, -0.4465],
                              std=[1., 1., 1.]),
        ])
        
        # 保存前100个错误分类的图片
        num_images = min(100, len(misclassified_images))
        rows = int(np.sqrt(num_images))
        cols = (num_images + rows - 1) // rows
        
        plt.figure(figsize=(20, 20))
        for idx in range(num_images):
            plt.subplot(rows, cols, idx + 1)
            img = denormalize(misclassified_images[idx])
            img = img.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            
            plt.imshow(img)
            plt.title(f'True: {class_names[misclassified_labels[idx]]}\nPred: {class_names[misclassified_preds[idx]]}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('misclassified/misclassified_examples.png')

if __name__ == "__main__":
    # 设置模型路径
    model_path = "/root/autodl-tmp/checkpoint_epoch_286_acc_0.9758.pth"
    analyze_model(model_path) 