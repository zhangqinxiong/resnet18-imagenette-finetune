# ResNet18 ImageNette Fine-Tuning

使用 **timm** 库对 **ResNet18** 在 **ImageNette** 数据集上进行全参数微调。

## 项目概述

| 项目 | 说明 |
|------|------|
| **任务** | 图像分类 (10 类) |
| **模型** | ResNet18 (来自 timm, ImageNet 预训练) |
| **数据集** | [ImageNette](https://github.com/fastai/imagenette) (320x320 版本) |
| **训练方式** | 全参数微调 (Full Fine-tuning) |
| **优化器** | AdamW |
| **学习率策略** | 3 epoch 预热 + 47 epoch 余弦退火 |
| **混合精度** | AMP (Automatic Mixed Precision) FP16 |
| **监控** | TensorBoard (端口 6006) |
| **最终验证准确率** | **~98%** |

## 环境配置

### 依赖安装

```bash
pip install torch torchvision timm tensorboard tqdm
```

### 推荐环境

- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.8 (GPU 训练，强烈推荐)
- timm >= 0.9.0

## 数据集准备

### 自动下载

```bash
# 使用 fastai 提供的脚本下载 ImageNette (320px 版本)
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette-320.tgz
tar -xzf imagenette-320.tgz -C /path/to/dataset
```

### 目录结构

数据集需整理为以下结构（`train/` 和 `val/` 下各 10 个类别子目录）：

```
ImageNette/
├── train/
│   ├── n01440764/   (tench)
│   ├── n02102040/   (English springer)
│   ├── n02979186/   (cassette player)
│   ├── n03000684/   (chain saw)
│   ├── n03028079/   (church)
│   ├── n03394916/   (French horn)
│   ├── n03417042/   (garbage truck)
│   ├── n03425413/   (gas pump)
│   ├── n03445777/   (golf ball)
│   └── n03888257/   (parachute)
└── val/
    ├── n01440764/
    ├── n02102040/
    └── ...
```

## 快速开始

### 1. 配置数据集路径

编辑 `train.py` 中的 `DATA_DIR` 变量：

```python
DATA_DIR = "/path/to/your/ImageNette"  # 改为你的数据集路径
```

### 2. 启动 TensorBoard (可选)

```bash
tensorboard --logdir ./runs --port 6006 --bind_all
```

然后在浏览器访问 `http://localhost:6006` 查看实时训练曲线。

### 3. 运行训练

```bash
python train.py
```

## 代码架构

```
resnet/
├── train.py                     # 主训练脚本
├── runs/                        # TensorBoard 日志目录 (自动生成)
├── resnet18_imagenette_best.pth # 最佳模型权重 (自动保存)
├── tb.log                       # TensorBoard 日志输出
└── README.md                    # 本文件
```

## 超参数详解

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DATA_DIR` | `/home/ivi/zqx/ImageNette` | 数据集根目录 |
| `BATCH_SIZE` | 64 | 每 GPU 批次大小 |
| `EPOCHS` | 50 | 总训练轮数 |
| `WARMUP_EPOCHS` | 3 | 学习率预热轮数 |
| `LR` | 5e-5 | 最大学习率 |
| `WEIGHT_DECAY` | 0.01 | AdamW 权重衰减系数 |
| `NUM_CLASSES` | 10 | 分类类别数 |

### 学习率调度策略

```
学习率
  ^
  |         /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
  |        /
  |       /
  |      /
  |     /
  |    / 余弦退火阶段 (47 epochs)
  |   /  ↓
  |  /   → 从 5e-5 衰减至 1e-6
  | /__________________________
  +---> epoch
  0   3                        50
  ↑
  预热阶段 (3 epochs)
  从 1e-6 线性上升至 5e-5
```

## 数据增强

### 训练集

| 增强方式 | 参数 | 目的 |
|---------|------|------|
| RandomResizedCrop | 224x224 | 尺度与位置不变性 |
| RandomHorizontalFlip | p=0.5 | 镜像对称增强 |
| ColorJitter | brightness=0.3, contrast=0.3, saturation=0.3 | 色彩鲁棒性 |
| Normalize | mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] | 对齐预训练分布 |

### 验证集

| 增强方式 | 参数 | 目的 |
|---------|------|------|
| Resize | 256 | 放大后裁剪 |
| CenterCrop | 224x224 | 标准中心评估 |
| Normalize | 同上 | 与训练集保持一致 |

## 训练细节

### 前向传播

```mermaid
graph LR
    A[输入图像 3x224x224] --> B[ResNet18 Backbone]
    B --> C[Global Average Pooling]
    C --> D[FC 512->10]
    D --> E[输出 logits 10维]
```

### AMP 混合精度流程

```
前向传播 (FP16) → Loss 计算 (FP16)
         ↓
  梯度缩放 (GradScaler)
         ↓
  反向传播 (FP16)
         ↓
  梯度反缩放 → 优化器更新 (FP32)
         ↓
  更新缩放因子
```

### 最佳模型保存

每个 epoch 结束后，若验证集准确率高于历史最佳，则保存模型权重到 `resnet18_imagenette_best.pth`。

## 结果

训练 50 epoch 后的典型结果 (在 NVIDIA GPU 上)：

```
Epoch 50/50 | Train Loss: 0.1458 Acc: 0.9555 | Val Loss: 0.0611 Acc: 0.9811 | LR: 1.00e-06
Training complete. Best val acc: 0.9811
```

- **验证准确率**: ~98.11%
- **训练时间**: ~10 分钟 (NVIDIA RTX 4090)
- **显存占用**: ~2.5 GB (batch_size=64)

## TensorBoard 可视化

启动 TensorBoard 后，可监控以下指标：

| 指标 | 路径 | 说明 |
|------|------|------|
| 训练 Loss | `Loss/train` | 训练集交叉熵损失 |
| 验证 Loss | `Loss/val` | 验证集交叉熵损失 |
| 训练准确率 | `Acc/train` | 训练集 Top-1 准确率 |
| 验证准确率 | `Acc/val` | 验证集 Top-1 准确率 |
| 学习率 | `LR` | 当前学习率变化曲线 |

## 常见问题

### Q: 显存不足怎么办？

- 减小 `BATCH_SIZE` (如 32 或 16)
- 确保 AMP 已启用 (默认开启)

### Q: 如何恢复训练？

当前脚本不支持断点续训。如需该功能，可添加 checkpoint 保存与加载逻辑：
```python
# 保存
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_acc': best_acc,
}, 'checkpoint.pth')

# 加载
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

### Q: 如何更换其他模型？

修改 `train.py` 中的模型名称：
```python
model = create_model("resnet34", pretrained=True, num_classes=NUM_CLASSES)  # 或 resnet50, resnet101 等
```

## License

MIT
