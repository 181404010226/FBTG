import gradio as gr
import torch
from Paper_Tree import SequentialDecisionTree
from torchvision import datasets, transforms
from Paper_DataSetCIFAR import create_valid_loader, data_config
import numpy as np
import random
import matplotlib.pyplot as plt

class ModelInterface:
    def __init__(self):
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")  # 添加设备信息打印
    
        # 加载模型
        self.model = SequentialDecisionTree().to(self.device)
        checkpoint = torch.load('D:/VisutalStudio/repository/FBTG/checkpoint_epoch_496_acc_0.9642.pth', 
                              map_location=self.device)
        
        # 处理状态字典
        state_dict = checkpoint['model_state']
        new_state_dict = {k.replace('module._orig_mod.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict)
        self.model.eval()

        # 加载原始数据集
        self.dataset = datasets.CIFAR10(root='CIFAR10RawData', train=False, download=True)
        self.sample_images = self._sample_dataset()
        
        # CIFAR10类别标签
        self.class_labels = ['airplane', 'automobile', 'bird', 'cat', 
                           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.save_sample_images()
    
    def _sample_dataset(self, num_samples=50):
        """从原始数据集中随机采样图片"""
        indices = random.sample(range(len(self.dataset)), num_samples)
        return [self.dataset[i][0] for i in indices]  # 只返回图像数据

    def save_sample_images(self):
        """保存采样的图片到本地文件夹"""
        import os
        
        # 创建保存图片的文件夹
        save_dir = "sample_images"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 保存采样的图片
        for i, img in enumerate(self.sample_images):
            img.save(f"{save_dir}/sample_{i}.png")  # PIL Image可以直接保存


    def _preprocess_image(self, image):
        """图像预处理函数"""
        # 应用数据集的标准化处理
        normalize = transforms.Normalize(
            mean=data_config['mean'],
            std=data_config['std']
        )
        return normalize(image).unsqueeze(0)  # 添加batch维度

    def predict(self, image, selected_nodes):
        if isinstance(image, str):
            # 从保存的原始图像加载
            image_idx = int(image.split('_')[-1].split('.')[0])
            image_tensor = transforms.ToTensor()(self.sample_images[image_idx])
        else:
            # 处理从Gradio接收的numpy数组
            image_tensor = transforms.ToTensor()(image)
    
        # 预处理图像并移到正确的设备上
        image_tensor = self._preprocess_image(image_tensor).to(self.device)

        # 获取选中节点的预测结果
        results = []
        with torch.no_grad():
            for i, node in enumerate(self.model.nodes):
                if i in selected_nodes:
                    output = node(image_tensor) 
                    probs = output[0].cpu().numpy()
                    
                    # 创建节点预测结果图
                    fig = plt.figure(figsize=(8, 4))
                    plt.bar(range(len(probs)), probs)
                    plt.title(f'Node {i+1} Predictions')
                    plt.ylim(0, 1)
                    
                    # 添加节点判别的类别标签
                    node_classes = ['+'.join([self.class_labels[idx] for idx in group]) 
                                for group in node.judge]
                    plt.xticks(range(len(probs)), node_classes, rotation=45)
                    
                    # 保存图片到临时文件
                    temp_file = f"temp_plot_{i}.png"
                    plt.savefig(temp_file, bbox_inches='tight', dpi=100)
                    plt.close()
                    
                    results.append(temp_file)

        return results

    def create_interface(self):
        """创建Gradio界面"""
        # 创建示例图片选项
        example_images = [f"sample_images/sample_{i}.png" for i in range(len(self.sample_images))]
        
        # 创建节点选择选项
        node_choices = [
            "Vehicle+Ship vs Animal",           # Node 1: [0,1,8,9] vs [2,3,4,5,6,7]
            "Airplane vs Car",                  # Node 2: [0,8] vs [1,9]
            "Airplane vs Ship",                 # Node 3: [0] vs [8]
            "Car vs Truck",                     # Node 4: [1] vs [9]
            "Bird+Frog vs Cat+Dog vs Deer+Horse", # Node 5: [2,6] vs [3,5] vs [4,7]
            "Bird vs Frog",                     # Node 6: [2] vs [6]
            "Cat vs Dog",                       # Node 7: [3] vs [5]
            "Deer vs Horse"                     # Node 8: [4] vs [7]
        ]

        # 定义界面
        interface = gr.Interface(
            fn=lambda img, nodes: self.predict(
                img, 
                [i for i, n in enumerate(node_choices) if n in nodes]
            ),
            inputs=[
                gr.Image(type="numpy", label="Upload or select an image"),
                gr.Checkboxgroup(
                    choices=node_choices,
                    label="Select nodes for prediction",
                    value=[node_choices[0]]
                )
            ],
            outputs=gr.Gallery(
                label="Prediction Results",
                show_label=True,
                elem_id="gallery",
                columns=[2],
                rows=[2],
                height="auto"
            ),
            examples=[[img, [node_choices[0]]] for img in example_images],
            title="Tree Node Visualization",
            description="Select an image and choose which tree nodes to visualize predictions from.",
            css="""
                #gallery {
                    margin: 0 auto;
                    max-width: 100%;
                    padding: 10px;
                }
                .output-image {
                    margin: 5px;
                }
            """
        )
        
        return interface

if __name__ == "__main__":
    interface = ModelInterface()
    interface.create_interface().launch(share=True)