import os
import torch
import numpy as np
import random
from timm.data import create_loader, Mixup
from torchvision import datasets, transforms
from Paper_global_vars import global_vars
from torchvision.transforms import Resize
from pprint import pprint


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


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

# 定义数据配置
data_config = {
    'input_size': (3, 224, 224),
    'interpolation': 'bicubic',
    'mean': IMAGENET_DEFAULT_MEAN,
    'std': IMAGENET_DEFAULT_STD,
    'crop_pct': 0.96,
}

mixup_args = dict(
    mixup_alpha=0.5,
    cutmix_alpha=0.5,
    prob=1.0,
    switch_prob=0.5,
    mode='batch',
    label_smoothing=0.1,
    num_classes=10
)

mixup_fn = Mixup(**mixup_args)

def collate_mixup_fn(batch):
    inputs = torch.stack([b[0] for b in batch])
    targets = torch.tensor([b[1] for b in batch])
    return mixup_fn(inputs, targets)

# 选择数据集
root = os.path.join(os.path.dirname(__file__), "CIFAR10RawData")
trainset = datasets.CIFAR10(root=root, train=True, download=True, transform=None)
testset = datasets.CIFAR10(root=root, train=False, download=True, transform=None)

# 创建训练数据加载器
loader_train = create_loader(
    trainset,
    input_size=data_config['input_size'],
    batch_size=global_vars.train_batch_size,  # -b 256
    is_training=True,
    use_prefetcher=False,
    no_aug=False,
    re_prob=0.25,  # --reprob 0.25
    re_mode='pixel',  # --remode pixel
    re_count=1,
    scale=(0.75, 1.0),  # --scale 0.75 1.0
    ratio=(3./4., 4./3.),
    hflip=0.5,
    vflip=0.,
    color_jitter=0.4,
    auto_augment='rand-m9-mstd0.5-inc1',  # --aa rand-m9-mstd0.5-inc1
    num_aug_splits=0,
    interpolation=data_config['interpolation'],
    mean=data_config['mean'],
    std=data_config['std'],
    num_workers=8,  # -j 8
    distributed=False,
    crop_pct=data_config['crop_pct'],   
    collate_fn=collate_mixup_fn,  # 使用新的 collate_fn
    use_multi_epochs_loader=False,
    worker_seeding='all',  
    pin_memory=True,
)


# 使用 create_loader 创建数据加载器
valid_data = create_loader(
    testset,
    input_size=data_config['input_size'],
    batch_size=global_vars.test_batch_size,
    is_training=False,
    use_prefetcher=False,  # 根据需要调整
    interpolation=data_config['interpolation'],
    mean=data_config['mean'],
    std=data_config['std'],
    num_workers=8,  # 根据需要调整
    distributed=False,  # 根据需要调整
    crop_pct=data_config['crop_pct'],
    pin_memory=True,  # 根据需要调整
)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    import torch
    import json

    for batch in range(3):
        # 获取一批训练数据
        # data_iter = iter(loader_train)
        data_iter = iter(valid_data)
        images, labels = next(data_iter)

        # 保存标签
        with open(f'variable_dump_batch_{batch}.json', 'w') as f:
            json.dump(labels.tolist(), f, indent=4)

        # 存储训练图片
        save_dir = f"saved_images_batch_{batch}"
        os.makedirs(save_dir, exist_ok=True)
        resize_transform = Resize((224, 224))

        for i, img in enumerate(images):
            # Resize the image
            img = resize_transform(img)
            img = img.permute(1, 2, 0)  # 将图像从 (C, H, W) 转换为 (H, W, C)
            img = img.numpy()
            # Normalize the image data to 0-1 range
            img = (img - img.min()) / (img.max() - img.min())
            
            plt.imsave(os.path.join(save_dir, f"image_{i}.png"), img)

    print("Three batches of data have been processed and saved.")