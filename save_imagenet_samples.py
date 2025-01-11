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

def extract_devkit():
    print("Extracting devkit...")
    with tarfile.open(DEVKIT_PATH, 'r:gz') as tar:
        tar.extractall(ROOT_DIR)
    
    # 移动meta文件到正确位置
    meta_src = os.path.join(ROOT_DIR, 'ILSVRC2012_devkit_t12/data/meta.mat')
    meta_dst = os.path.join(ROOT_DIR, 'meta.mat')
    if os.path.exists(meta_src):
        shutil.move(meta_src, meta_dst)
    
    # 移动val标注文件到正确位置
    val_src = os.path.join(ROOT_DIR, 'ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt')
    val_dst = os.path.join(ROOT_DIR, 'val/ILSVRC2012_validation_ground_truth.txt')
    if os.path.exists(val_src):
        os.makedirs(os.path.dirname(val_dst), exist_ok=True)
        shutil.move(val_src, val_dst)

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

def organize_val_by_class():
    """将验证集图片按类别组织到子文件夹中"""
    print("Organizing validation images into class folders...")
    
    # 读取验证集标注文件
    val_anno_path = os.path.join(ROOT_DIR, 'val/ILSVRC2012_validation_ground_truth.txt')
    with open(val_anno_path, 'r') as f:
        val_labels = [int(line.strip()) for line in f.readlines()]
    
    # 读取类别映射文件
    meta_path = os.path.join(ROOT_DIR, 'meta.mat')
    import scipy.io
    meta = scipy.io.loadmat(meta_path)
    synsets = meta['synsets']
    wnids = [str(s[0][1][0]) for s in synsets]
    
    # 创建类别文件夹并移动图片
    for idx, label in enumerate(val_labels, 1):
        # ImageNet验证集图片命名格式为ILSVRC2012_val_00000001.JPEG
        src_img = os.path.join(VAL_DEST_DIR, f'ILSVRC2012_val_{idx:08d}.JPEG')
        if not os.path.exists(src_img):
            continue
        
        # 获取对应的类别文件夹
        wnid = wnids[label-1]  # label从1开始
        dst_dir = os.path.join(VAL_DEST_DIR, wnid)
        os.makedirs(dst_dir, exist_ok=True)
        
        # 移动图片到对应类别文件夹
        dst_img = os.path.join(dst_dir, f'ILSVRC2012_val_{idx:08d}.JPEG')
        shutil.move(src_img, dst_img)
        if idx % 1000 == 0:
            print(f"Processed {idx} validation images")

def extract_val():
    # 首先解压所有验证集图片到临时目录
    print("Extracting validation images...")
    with open(VAL_SRC_DIR, 'rb') as f:
        tar = tarfile.open(fileobj=f, mode='r:')
        if not os.path.isdir(VAL_DEST_DIR):
            os.makedirs(VAL_DEST_DIR)
        tar.extractall(VAL_DEST_DIR)
    
    # 然后将图片组织到类别文件夹中
    organize_val_by_class()

if __name__ == '__main__':
    os.makedirs(ROOT_DIR, exist_ok=True)
    # 首先解压devkit
    extract_devkit()
    # extract_train()
    extract_val()