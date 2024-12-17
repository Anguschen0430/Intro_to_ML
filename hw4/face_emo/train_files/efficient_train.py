import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import matplotlib.pyplot as plt
from efficientnet_pytorch import EfficientNet

from approach.efficeient import EfficientEmoteNet
from get_dataset import Four4All

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# 修正数据增强顺序
transform_train = transforms.Compose([
    # 1. 首先进行空间变换（在PIL图像上进行）
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1)
    ),
    
    # 2. 然后是颜色变换（在PIL图像上进行）
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    
    # 3. 转换为灰度图（在PIL图像上进行）
    transforms.Grayscale(num_output_channels=3),
    
    # 4. 转换为张量（PIL图像 -> 张量）
    transforms.ToTensor(),
    
    # 5. 在张量上进行的变换
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    transforms.RandomErasing(p=0.1)
])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 加载数据集
train_dataset = Four4All(
    csv_file='../rafdb_large/train_labels.csv',
    img_dir='../rafdb_large/train', 
    transform=transform_train
)
train_loader = DataLoader(
    train_dataset, 
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

val_dataset = Four4All(
    csv_file='../rafdb_large/val_labels.csv',
    img_dir='../rafdb_large/val',
    transform=transform
)
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

test_dataset = Four4All(
    csv_file='../rafdb_split2/test_labels.csv',
    img_dir='../rafdb_split2/test',
    transform=transform
)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

# 打印数据集信息
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# 加载模型
model = EfficientEmoteNet().to(device)

# 打印参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')

# 训练参数设置
criterion = torch.nn.CrossEntropyLoss()
initial_lr = 1e-3
min_epochs = 30
num_epochs = 200
patience = 20
best_val_acc = 0
patience_counter = 0
epoch_counter = 0

# 确保检查点目录存在
os.makedirs('checkpoints/efficient', exist_ok=True)

# 保存结果的列表
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
test_losses = []
test_accuracies = []
saved_models = []
max_saved = 3

try:
    # 首先冻结backbone进行训练
    print("Phase 1: Training with frozen backbone")
    model.freeze_backbone(True)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=initial_lr,
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=initial_lr,
        epochs=min_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2
    )

    # 开始训练循环
    for epoch in range(num_epochs):
        if epoch == min_epochs:
            print("Phase 2: Fine-tuning entire model")
            model.freeze_backbone(False)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=initial_lr/10,
                weight_decay=1e-5
            )
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=initial_lr/10,
                epochs=num_epochs-min_epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.2
            )

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # 训练循环
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct/total:.2f}%'
            })

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # 验证集评估
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # 测试集评估
        test_running_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_loss = test_running_loss / len(test_loader)
        test_acc = test_correct / test_total
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        # 打印进度
        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        epoch_counter += 1

        # 保存最好的模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            model_path = f'checkpoints/efficient/efficient_{epoch}_{val_acc:.4f}.pth'
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
            }
            torch.save(checkpoint, model_path)
            
            saved_models.append((val_acc, model_path))
            saved_models.sort(reverse=True)
            if len(saved_models) > max_saved:
                _, old_path = saved_models.pop()
                if os.path.exists(old_path):
                    os.remove(old_path)
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs.")

        # 早停
        if epoch >= min_epochs and patience_counter > patience:
            print("Early stopping triggered.")
            break

    # 保存训练结果
    df = pd.DataFrame({
        'Epoch': range(1, epoch_counter+1),
        'Train Loss': train_losses,
        'Test Loss': test_losses,
        'Validation Loss': val_losses,
        'Train Accuracy': train_accuracies,
        'Test Accuracy': test_accuracies,
        'Validation Accuracy': val_accuracies
    })
    df.to_csv('results_efficient.csv', index=False)

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

    print(f"Best validation accuracy: {best_val_acc:.4f}")

except Exception as e:
    print(f"Training interrupted: {str(e)}")
    import traceback
    print(traceback.format_exc())
    
    # 保存已有的训练结果
    if epoch_counter > 0:
        df = pd.DataFrame({
            'Epoch': range(1, epoch_counter+1),
            'Train Loss': train_losses,
            'Test Loss': test_losses,
            'Validation Loss': val_losses,
            'Train Accuracy': train_accuracies,
            'Test Accuracy': test_accuracies,
            'Validation Accuracy': val_accuracies
        })
        df.to_csv('results_efficient_interrupted.csv', index=False)