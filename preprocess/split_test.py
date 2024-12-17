import os
import shutil
import random
from tqdm import tqdm
from PIL import Image, ImageEnhance, ImageOps, ImageFilter

def augment_image(image_path, output_dir, count):
    """
    對單張圖片進行數據增強，生成指定數量的變化圖片
    """
    try:
        image = Image.open(image_path)
        for i in range(count):
            # 隨機旋轉
            angle = random.choice([0, 15, 30, 45])
            augmented_image = image.rotate(angle)
            
            # 隨機調整亮度
            enhancer = ImageEnhance.Brightness(augmented_image)
            brightness = random.uniform(0.9, 1.1)
            augmented_image = enhancer.enhance(brightness)
            
            # 隨機水平翻轉
            if random.choice([True, False]):
                augmented_image = ImageOps.mirror(augmented_image)
            
            # 隨機添加適度的高斯模糊
            if random.choice([True, False]):
                blur_radius = random.uniform(0.3, 0.8)
                augmented_image = augmented_image.filter(ImageFilter.GaussianBlur(blur_radius))
            
            # 保存增強後的圖片
            new_name = f"aug_{i+1}_{os.path.basename(image_path)}"
            augmented_image.save(os.path.join(output_dir, new_name))
    except Exception as e:
        print(f"Error augmenting image {image_path}: {e}")

def balance_and_split_dataset(source_dir, target_dir, target_count=3000, val_ratio=0.1):
    """
    平衡數據集並分割成訓練、驗證和測試集
    - 先從原始數據取10%作為驗證集
    - 複製驗證集到測試集
    - 剩下90%做增強到目標數量作為訓練集
    """
    os.makedirs(target_dir, exist_ok=True)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(target_dir, split), exist_ok=True)
    
    emotion_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    for emotion in emotion_dirs:
        print(f"Processing {emotion}...")
        emotion_path = os.path.join(source_dir, emotion)
        
        # 獲取原始圖片
        original_images = [img for img in os.listdir(emotion_path) if img.endswith('.jpg')]
        total = len(original_images)
        
        # 分配驗證集（10%原始圖片）
        val_size = int(total * val_ratio)
        random.shuffle(original_images)
        val_imgs = original_images[:val_size]
        train_imgs = original_images[val_size:]  # 剩下90%用於訓練
        
        # 計算需要增強的數量
        augment_needed = max(0, target_count - len(train_imgs))
        
        if augment_needed > 0:
            # 計算每張圖片需要增強的數量
            augment_per_image = augment_needed // len(train_imgs) + 1
            
            # 對訓練集圖片進行增強
            for img in tqdm(train_imgs, desc=f"Augmenting {emotion}"):
                augment_image(
                    os.path.join(emotion_path, img),
                    emotion_path,
                    augment_per_image
                )
            
            # 獲取所有訓練圖片（原始 + 增強）
            all_train_images = train_imgs + [
                img for img in os.listdir(emotion_path) 
                if img.startswith('aug_') and img.endswith('.jpg')
            ]
            
            # 如果數量超過目標，隨機選擇
            if len(all_train_images) > target_count:
                train_imgs = random.sample(all_train_images, target_count)
            else:
                train_imgs = all_train_images
        
        # 複製圖片到對應目錄
        for split, imgs in [('train', train_imgs), ('val', val_imgs), ('test', val_imgs)]:  # 注意test使用val_imgs
            for i, img in enumerate(imgs):
                src = os.path.join(emotion_path, img)
                dst = os.path.join(target_dir, split, f"{split}_{i+1}_{emotion.lower()}.jpg")
                shutil.copy2(src, dst)
        
        print(f"{emotion}: Original={total}, Train={len(train_imgs)} (Augmented={len(train_imgs)-len(original_images[val_size:])}), Val/Test={len(val_imgs)}")
        
        # 清理增強的圖片
        for img in os.listdir(emotion_path):
            if img.startswith('aug_'):
                os.remove(os.path.join(emotion_path, img))

# 設置隨機種子
random.seed(42)

# 執行分割和平衡
source_directory = "../data/Images/train"
target_directory = "../rafdb_split_balanced"
balance_and_split_dataset(source_directory, target_directory, target_count=3000)