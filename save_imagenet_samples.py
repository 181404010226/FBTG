import os
import tarfile
import shutil
from torchvision.datasets import ImageNet
import torch

TRAIN_SRC_DIR = '/root/autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_img_train.tar'
TRAIN_DEST_DIR = '/root/autodl-tmp/imagenet/train'
VAL_SRC_DIR = '/root/autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_img_val.tar'
VAL_DEST_DIR = '/root/autodl-tmp/imagenet/val'
DEVKIT_PATH = '/root/autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_devkit_t12.tar.gz'
ROOT_DIR = '/root/autodl-tmp/imagenet'

def extract_train():
    # Training data extraction remains the same
    with open(TRAIN_SRC_DIR, 'rb') as f:
        tar = tarfile.open(fileobj=f, mode='r:')
        for i, item in enumerate(tar):
            cls_name = item.name.strip(".tar")
            a = tar.extractfile(item)
            b = tarfile.open(fileobj=a, mode="r:")
            e_path = os.path.join(TRAIN_DEST_DIR, cls_name)
            if not os.path.isdir(e_path):
                os.makedirs(e_path)
            print("#", i, "extract train dataset to >>>", e_path)
            b.extractall(e_path)

def extract_val():
    # First extract all validation images to a temporary directory
    temp_val_dir = os.path.join(VAL_DEST_DIR, 'temp')
    os.makedirs(temp_val_dir, exist_ok=True)
    
    with open(VAL_SRC_DIR, 'rb') as f:
        tar = tarfile.open(fileobj=f, mode='r:')
        tar.extractall(temp_val_dir)

    # Copy devkit to root directory
    shutil.copy2(DEVKIT_PATH, os.path.join(ROOT_DIR, 'ILSVRC2012_devkit_t12.tar.gz'))

    # Use torchvision's built-in function to organize validation data
    dataset = ImageNet(ROOT_DIR, split='val')
    
    # Clean up temporary directory
    shutil.rmtree(temp_val_dir)

if __name__ == '__main__':
    os.makedirs(ROOT_DIR, exist_ok=True)
    extract_train()
    extract_val()