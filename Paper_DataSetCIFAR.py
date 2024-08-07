import os
import torch
import numpy as np
import random
from timm.data import create_loader, Mixup
from torchvision import datasets, transforms
from Paper_global_vars import global_vars
from torchvision.transforms import Resize
from pprint import pprint
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

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
    'input_size': (3, 224, 224), #astroformer
    #'input_size': (3, 32, 32),
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
    num_classes=100
)

# 选择数据集
root = os.path.join(os.path.dirname(__file__), "CIFAR10RawData")
trainset_cifar10 = datasets.CIFAR10(root=root, train=True, download=True, transform=None)
testset_cifar10 = datasets.CIFAR10(root=root, train=False, download=True, transform=None)
trainset_cifar100 = datasets.CIFAR100(root=root, train=True, download=True, transform=None)
testset_cifar100 = datasets.CIFAR100(root=root, train=False, download=True, transform=None)

# 修改创建训练数据加载器的部分
def create_train_loader(dataset='cifar10', distributed=False):
    global loader_train
    
    if dataset == 'cifar10':
        trainset = trainset_cifar10
        num_classes = 10
    elif dataset == 'cifar100':
        trainset = trainset_cifar100
        num_classes = 100
    else:
        raise ValueError("Invalid dataset. Choose 'cifar10' or 'cifar100'.")
    
    mixup_args['num_classes'] = num_classes
    mixup_fn = Mixup(**mixup_args)

    def collate_mixup_fn(batch):
        inputs = torch.stack([b[0] for b in batch])
        targets = torch.tensor([b[1] for b in batch])
        return mixup_fn(inputs, targets)
    
    loader_train = create_loader(
        trainset,
        input_size=data_config['input_size'],
        batch_size=global_vars.train_batch_size,
        is_training=True,
        use_prefetcher=False,
        no_aug=False,
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
        scale=(0.75, 1.0),
        ratio=(3./4., 4./3.),
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment='rand-m9-mstd0.5-inc1',
        num_aug_splits=0,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=8,
        distributed=distributed,
        crop_pct=data_config['crop_pct'],   
        collate_fn=collate_mixup_fn,
        use_multi_epochs_loader=False,
        worker_seeding='all',  
        pin_memory=True
    )
    return loader_train

# 修改创建验证数据加载器的部分
def create_valid_loader(dataset='cifar10', distributed=False):
    global valid_data
    
    if dataset == 'cifar10':
        testset = testset_cifar10
    elif dataset == 'cifar100':
        testset = testset_cifar100
    else:
        raise ValueError("Invalid dataset. Choose 'cifar10' or 'cifar100'.")
    
    valid_data = create_loader(
        testset,
        input_size=data_config['input_size'],
        batch_size=global_vars.test_batch_size,
        is_training=False,
        use_prefetcher=False,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=8,
        distributed=distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=True
    )
    return valid_data


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    import torch
    import json

    for batch in range(3):
        # 获取一批训练数据
        # data_iter = iter(loader_train)
        loader_train = create_train_loader(dataset='cifar10',distributed=False)
        # valid_data = create_valid_loader(dataset='cifar10',distributed=False)
        data_iter = iter(loader_train)
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