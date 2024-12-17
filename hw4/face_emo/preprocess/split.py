import os
import shutil
import random
from tqdm import tqdm

def split_dataset(source_dir, target_dir, val_ratio=0.1):
    # 創建目標目錄
    os.makedirs(target_dir, exist_ok=True)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(target_dir, split), exist_ok=True)
    
    # 讀取所有表情目錄
    emotion_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    for emotion in emotion_dirs:
        print(f"Processing {emotion}...")
        emotion_path = os.path.join(source_dir, emotion)
        print(f"Emotion path: {emotion_path}")
        # 獲取該表情的所有圖片
        images = [img for img in os.listdir(emotion_path) if img.endswith('.jpg')]
        
        random.shuffle(images)  # 隨機打亂
        
        # 計算分割點
        total = len(images)
        val_split = int(total * val_ratio)
        
        # 分割數據集
        val_images = images[:val_split]  # 10% 給 validation
        train_images = images[val_split:]  # 90% 給 training
        
        # 複製文件到對應目錄
        # Training 數據
        for i, img in enumerate(train_images):
            src = os.path.join(emotion_path, img)
            new_name = f"train_{i+1}_{emotion.lower()}.jpg"
            dst = os.path.join(target_dir, 'train', new_name)
            shutil.copy2(src, dst)
        
        # Validation 和 Test 數據 (使用相同的圖片)
        for i, img in enumerate(val_images):
            src = os.path.join(emotion_path, img)
            # 複製到 validation 目錄
            val_name = f"val_{i+1}_{emotion.lower()}.jpg"
            val_dst = os.path.join(target_dir, 'val', val_name)
            shutil.copy2(src, val_dst)
            
            # 複製到 test 目錄
            test_name = f"test_{i+1}_{emotion.lower()}.jpg"
            test_dst = os.path.join(target_dir, 'test', test_name)
            shutil.copy2(src, test_dst)
        
        print(f"{emotion}: Total={total}, Train={len(train_images)}, Val/Test={len(val_images)}")

# 設置隨機種子以確保可重複性
random.seed(42)

# 使用示例
source_directory = "../data/Images copy/train"  # 原始train目錄路徑
target_directory = "../rafdb_large"  # 新目錄路徑

# 執行分割
split_dataset(source_directory, target_directory)