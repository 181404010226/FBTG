# Sequential Decision Tree Model

This project implements a Sequential Decision Tree model for image classification tasks, with support for CIFAR-10 and CIFAR-100 datasets.

[English](#english) | [中文](#chinese)

<a name="english"></a>
## English

### Key Features

- Flexible model architecture supporting various backbone networks (ResNet, ConvMixer, RDNet, etc.)
- Support for both CIFAR-10 and CIFAR-100 datasets
- Distributed Data Parallel (DDP) training support for multi-GPU setups

### Configuration

All main parameters can be adjusted in the `Paper_global_vars.py` file. This includes settings such as:

- Number of epochs
- Input size
- Learning rate
- Batch size
- Dataset selection
- Model selection
- Optimizer choice
- Model save path

### Running the Training

#### Single GPU

For single GPU training, simply run:

```
python Paper_trainForMultiDDP.py
```

#### Multi-GPU

For multi-GPU training using DDP, use the following command:

```
python -m torch.distributed.launch --nproc_per_node=<num_gpus> Paper_trainForMultiDDP.py
```


Replace `<num_gpus>` with the number of GPUs you want to use.

### Core Components

The core of this project is the `Paper_Tree.py` file, which integrates multiple model architectures:

- SequentialDecisionTree (for CIFAR-10)
- SequentialDecisionTreeCIFAR100 (for CIFAR-100)
- SequentialDecisionTreeForRDNet (RDNet-based for CIFAR-10)
- SequentialDecisionTreeCIFAR100ForRDNet (RDNet-based for CIFAR-100)

These models incorporate various backbone networks such as ResNet, ConvMixer, and RDNet. You can easily integrate your own custom models and experiment with different architectures.

### Customization

To use your own models or datasets:

1. Add your model implementation to the `Paper_Tree.py` file.
2. Update the `global_vars` in `Paper_global_vars.py` to include your new model.
3. Modify the data loading process in `Paper_trainForMultiDDP.py` if using a custom dataset.

Feel free to experiment with different model architectures and training configurations to achieve the best results for your specific use case.

---

<a name="chinese"></a>
## 中文

### 主要特点

- 灵活的模型架构，支持各种骨干网络（ResNet、ConvMixer、RDNet等）
- 支持CIFAR-10和CIFAR-100数据集
- 支持多GPU设置的分布式数据并行（DDP）训练

### 配置

所有主要参数都可以在`Paper_global_vars.py`文件中调整。这包括以下设置：

- 训练轮数
- 输入大小
- 学习率
- 批量大小
- 数据集选择
- 模型选择
- 优化器选择
- 模型保存路径

### 运行训练

#### 单GPU

对于单GPU训练，只需运行：

```
python Paper_trainForMultiDDP.py
```

#### 多GPU

对于使用DDP的多GPU训练，使用以下命令：

```
python -m torch.distributed.launch --nproc_per_node=<num_gpus> Paper_trainForMultiDDP.py
```


将`<num_gpus>`替换为您想使用的GPU数量。

### 核心组件

本项目的核心是`Paper_Tree.py`文件，它集成了多个模型架构：

- SequentialDecisionTree（用于CIFAR-10）
- SequentialDecisionTreeCIFAR100（用于CIFAR-100）
- SequentialDecisionTreeForRDNet（基于RDNet的CIFAR-10）
- SequentialDecisionTreeCIFAR100ForRDNet（基于RDNet的CIFAR-100）

这些模型包含了各种骨干网络，如ResNet、ConvMixer和RDNet。您可以轻松集成自己的自定义模型并尝试不同的架构。

### 自定义

要使用您自己的模型或数据集：

1. 将您的模型实现添加到`Paper_Tree.py`文件中。
2. 更新`Paper_global_vars.py`中的`global_vars`以包含您的新模型。
3. 如果使用自定义数据集，请修改`Paper_trainForMultiDDP.py`中的数据加载过程。

请随意尝试不同的模型架构和训练配置，以达到您特定用例的最佳结果。