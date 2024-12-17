import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import pandas as pd
import timm
from get_dataset import Four4All

# 定義 FaceViT 模型
class FaceViT(nn.Module):
    def __init__(self, num_classes=7):
        super(FaceViT, self).__init__()
        # 加載預訓練的 ViT 模型
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        # 替換分類器頭部
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, x):
        return self.vit(x)


def train_model():
    # ================== 配置參數 ==================
    # 數據相關參數
    batch_size = 64  # 調整批次大小
    image_size = 224  # 調整圖片尺寸
    num_epochs = 100  # 調整訓練輪數
    patience = 15  # 調整提前停止耐心值
    checkpoint_interval = 4  # 調整檢查點間隔

    # 優化器參數
    learning_rate = 1e-4  # 調整學習率
    weight_decay = 0.01  # 調整權重衰減python -m train_files.facevit_train

    # 數據增強參數
    use_augmentation = True
    flip_prob = 0.5  # 保持不變
    rotation_degrees = 20  # 調整旋轉角度
    brightness = 0.1  # 調整亮度
    contrast = 0.1  # 調整對比度
    saturation = 0.1  # 調整飽和度

    # 路徑設置
    checkpoint_dir = "checkpoints"
    data_dir = "../rafdb_split_v1"

    # 設備設置
    device = torch.device("cuda" )
    print(f"Using device: {device}")
    num_workers = 4

    # ================== 數據增強和加載 ==================
    base_transforms = [
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),  # 如果是灰階圖片，轉換為3通道
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]

    if use_augmentation:
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=flip_prob),
            transforms.RandomRotation(rotation_degrees),
            transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation)
        ] + base_transforms)
    else:
        train_transforms = transforms.Compose(base_transforms)

    val_test_transforms = transforms.Compose(base_transforms)

    # 加載數據集
    train_dataset = Four4All(
        csv_file=os.path.join(data_dir, 'train_labels.csv'),
        img_dir=os.path.join(data_dir, 'train'),
        transform=train_transforms
    )

    val_dataset = Four4All(
        csv_file=os.path.join(data_dir, 'val_labels.csv'),
        img_dir=os.path.join(data_dir, 'val'),
        transform=val_test_transforms
    )

    test_dataset = Four4All(
        csv_file=os.path.join(data_dir, 'test_labels.csv'),
        img_dir=os.path.join(data_dir, 'test'),
        transform=val_test_transforms
    )

    train_loader = DataLoader(
                        train_dataset, 
                        batch_size=batch_size, 
                        shuffle=True, 
                        num_workers=8,  # 增加 worker 數量
                        pin_memory=True,  # 使用固定內存
                        persistent_workers=True,  # 保持 worker 進程存活
                        prefetch_factor=2  # 預讀取因子
                    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # ================== 模型初始化 ==================
    try:
        model = FaceViT(num_classes=len(train_dataset.labels.iloc[:, 1].unique())).to(device)
        
        print(f"Model initialized successfully! Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # ================== 訓練和驗證 ==================
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_acc = 0
    patience_counter = 0

    history = {
        "train_losses": [], "val_losses": [], "test_losses": [],
        "train_accuracies": [], "val_accuracies": [], "test_accuracies": []
    }

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}")
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            try:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            except Exception as e:
                print(f"Error during training loop: {e}")
                return

        train_loss /= len(train_loader)
        train_acc = correct / total
        history["train_losses"].append(train_loss)
        history["train_accuracies"].append(train_acc)

        # 驗證階段
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = correct / total
        history["val_losses"].append(val_loss)
        history["val_accuracies"].append(val_acc)

        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'vit_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), os.path.join(checkpoint_dir,f'vit_epoch_{epoch+1}.pth'))
            print(f"Saved checkpoint at epoch {epoch+1}")
            
        # Early Stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
        else:
            patience_counter += 1
            print(f"---------patience_counter:{patience_counter}")
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    return history


if __name__ == "__main__":
    history = train_model()

    # 保存訓練歷史
    if history:
        df = pd.DataFrame({
            "Epoch": range(1, len(history["train_losses"]) + 1),
            "Train Loss": history["train_losses"],
            "Validation Loss": history["val_losses"],
            "Train Accuracy": history["train_accuracies"],
            "Validation Accuracy": history["val_accuracies"]
        })
        df.to_csv("training_results.csv", index=False)