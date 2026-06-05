"""
ResNet18 全参数微调训练脚本 (ImageNette 数据集)

功能:
  - 使用 timm 库加载预训练的 ResNet18 模型
  - 在 ImageNette 数据集上进行全参数微调 (Full Fine-tuning)
  - 支持 AMP (Automatic Mixed Precision) 混合精度训练
  - 使用 Cosine 余弦退火学习率调度 + Warmup 预热
  - 通过 TensorBoard 实时监控训练指标 (Loss / Accuracy / LR)
  - 自动保存验证集上最佳准确率的模型权重

依赖:
  pip install torch torchvision timm tensorboard tqdm
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from timm import create_model
from timm.scheduler.cosine_lr import CosineLRScheduler
from tqdm import tqdm
import os

# ========================== 超参数配置 ==========================

DATA_DIR = "/home/ivi/zqx/ImageNette"  # 数据集根目录 (需包含 train/ 和 val/ 子目录)
BATCH_SIZE = 64                        # 每批次样本数
EPOCHS = 50                            # 总训练轮数
WARMUP_EPOCHS = 3                      # 预热轮数 (学习率从 warmup_lr_init 线性上升到 LR)
LR = 5e-5                              # 最大学习率
WEIGHT_DECAY = 0.01                    # AdamW 权重衰减 (L2 正则化)
NUM_CLASSES = 10                       # ImageNette 类别数

# ========================== 设备初始化 ==========================

# 自动检测 GPU，若不可用则回退到 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========================== 数据预处理 ==========================

# 训练集数据增强:
#   - RandomResizedCrop: 随机裁剪并缩放至 224x224，提升尺度与位置不变性
#   - RandomHorizontalFlip: 随机水平翻转，提升泛化能力
#   - ColorJitter: 随机调整亮度/对比度/饱和度，增强色彩鲁棒性
#   - ToTensor: PIL Image -> Tensor [0,1]
#   - Normalize: 使用 ImageNet 的 mean/std 标准化，与预训练权重对齐
train_tfm = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 验证集数据增强:
#   - Resize(256) + CenterCrop(224): 标准的中心裁剪评估方式
#   - 不添加随机增强，确保评估结果稳定可复现
val_tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ========================== 数据集加载 ==========================

# ImageFolder 会自动按子目录名称解析类别标签
train_ds = ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tfm)
val_ds = ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_tfm)

print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
print(f"Classes: {train_ds.classes}")

# DataLoader 参数说明:
#   - shuffle=True:   训练集每个 epoch 打乱数据
#   - num_workers=4:  4 个子进程并行加载数据，减少 CPU 瓶颈
#   - pin_memory=True: 将 Tensor 固定在页锁定内存，加速 GPU 传输
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ========================== 模型构建 ==========================

# create_model 来自 timm 库:
#   - "resnet18": 最轻量的 ResNet 系列模型 (约 11M 参数)
#   - pretrained=True: 加载在 ImageNet-1K 上预训练的权重
#   - num_classes=10: 替换分类头，适配 ImageNette 的 10 类输出
model = create_model("resnet18", pretrained=True, num_classes=NUM_CLASSES)
model.to(device)

# 打印模型参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

# ========================== 损失函数与优化器 ==========================

# CrossEntropyLoss = LogSoftmax + NLLLoss，适合多分类任务
criterion = nn.CrossEntropyLoss()

# AdamW: 带解耦权重衰减的 Adam 优化器，比 Adam + L2 正则化效果更好
#   - lr=5e-5:        对于微调任务使用较小的学习率
#   - weight_decay=0.01: 控制正则化强度，防止过拟合
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# ========================== 学习率调度器 ==========================

# CosineLRScheduler (来自 timm):
#   - t_initial=EPOCHS-WARMUP_EPOCHS:  余弦退火的总步数 (47)
#   - lr_min=1e-6:        学习率下限
#   - warmup_lr_init=1e-6: 预热起始学习率
#   - warmup_t=3:          预热步数 (3)，学习率从 1e-6 线性上升到 LR
#   - cycle_limit=1:       只做一个余弦周期
# 调度策略: [0, 3) 预热上升, [3, 50) 余弦下降至 lr_min
scheduler = CosineLRScheduler(
    optimizer,
    t_initial=EPOCHS - WARMUP_EPOCHS,
    lr_min=1e-6,
    warmup_lr_init=1e-6,
    warmup_t=WARMUP_EPOCHS,
    cycle_limit=1,
)

# ========================== AMP 混合精度初始化 ==========================

# GradScaler: 对梯度进行动态缩放，防止 FP16 精度下的梯度下溢
scaler = torch.amp.GradScaler(device=device.type)

# ========================== TensorBoard ==========================

# SummaryWriter 将日志写入 ./runs 目录
# 启动命令: tensorboard --logdir ./runs --port 6006
writer = SummaryWriter(log_dir="./runs")

# ========================== 训练循环 ==========================

best_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    # ------------------------- 训练阶段 -------------------------
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    # tqdm 进度条显示
    pbar = tqdm(train_loader, desc=f"Epoch {epoch:2d}/{EPOCHS} [Train]")
    for images, labels in pbar:
        # 将数据移至 GPU
        images, labels = images.to(device), labels.to(device)

        # 清除上一步的梯度
        optimizer.zero_grad()

        # AMP 自动混合精度上下文:
        #   - 前向传播在 FP16 下执行，加速计算并减少显存占用
        #   - 关键操作 (如 BatchNorm) 自动回退到 FP32 保证精度
        with torch.amp.autocast(device_type=device.type):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # 反向传播 (使用 scaler 缩放梯度)
        scaler.scale(loss).backward()
        # 更新模型参数 (内部会 unscale 梯度)
        scaler.step(optimizer)
        # 更新缩放因子，为下一轮做准备
        scaler.update()

        # 统计指标
        train_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

        # 更新进度条显示的当前 loss
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # 计算训练集平均 loss 和准确率
    train_loss /= train_total
    train_acc = train_correct / train_total

    # 更新学习率调度器 (timm 的 CosineLRScheduler 按 epoch 步进)
    scheduler.step(epoch)

    # ------------------------- 验证阶段 -------------------------
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    # 验证阶段不计算梯度，节省显存和加速
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch:2d}/{EPOCHS} [Val ]"):
            images, labels = images.to(device), labels.to(device)
            # 验证时同样使用 AMP 加速
            with torch.amp.autocast(device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total

    # ------------------------- 记录与输出 -------------------------

    # 写入 TensorBoard
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("Acc/train", train_acc, epoch)
    writer.add_scalar("Acc/val", val_acc, epoch)
    writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

    # 控制台输出
    print(f"Epoch {epoch:2d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
          f"LR: {optimizer.param_groups[0]['lr']:.2e}")

    # 保存验证集最佳模型
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "resnet18_imagenette_best.pth")
        print(f"  => Saved best model (acc: {best_acc:.4f})")

# 关闭 TensorBoard writer
writer.close()
print(f"\nTraining complete. Best val acc: {best_acc:.4f}")
print(f"Best model saved to: resnet18_imagenette_best.pth")
