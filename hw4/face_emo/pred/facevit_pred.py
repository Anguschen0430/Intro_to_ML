import torch
from torchvision import transforms
import os
from PIL import Image
import pandas as pd
import timm
from torch import nn

class FaceViTPredictor:
    def __init__(self, model_path, config=None):
        if config is None:
            self.config = self._get_default_config()
        else:
            self.config = config
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.transform = self._get_transforms()
        
        # 情緒標籤映射
        self.emotion_labels = {
            0: 'Angry',
            1: 'Disgust',
            2: 'Fear',
            3: 'Happy',
            4: 'Neutral',
            5: 'Sad',
            6: 'Surprise'
        }
        # 反向映射
        self.label_to_idx = {v.lower(): k for k, v in self.emotion_labels.items()}
    
    def _get_default_config(self):
        class DefaultConfig:
            def __init__(self):
                self.image_size = 224  # ViT 默認輸入大小
                self.num_classes = 7
        return DefaultConfig()
    
    def _load_model(self, model_path):
        # 創建模型結構
        class FaceViT(nn.Module):
            def __init__(self, num_classes):
                super(FaceViT, self).__init__()
                self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
                self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

            def forward(self, x):
                return self.vit(x)
        
        # 初始化模型
        model = FaceViT(num_classes=self.config.num_classes).to(self.device)
        # 載入權重
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        model.eval()
        return model
    
    def _get_transforms(self):
        return transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.Grayscale(num_output_channels=3),  # 確保輸入為 3 通道
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def predict_image(self, image_path):
        """預測單張圖片的情緒"""
        # 載入並預處理圖片
        image = Image.open(image_path)
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 進行預測
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy().flatten()
            predicted_class = probabilities.argmax()
            
        return {
            'class_id': predicted_class,
            'emotion': self.emotion_labels[predicted_class],
            'probabilities': probabilities
        }
    
    def predict_batch(self, image_paths):
        """批量預測多張圖片"""
        predictions = []
        for image_path in image_paths:
            try:
                result = self.predict_image(image_path)
                probabilities = result['probabilities']
                predictions.append({
                    'filename': os.path.basename(image_path),
                    'label': result['class_id'],
                    'emotion': result['emotion'],
                    'Angry_prob': probabilities[0],
                    'Disgust_prob': probabilities[1],
                    'Fear_prob': probabilities[2],
                    'Happy_prob': probabilities[3],
                    'Neutral_prob': probabilities[4],
                    'Sad_prob': probabilities[5],
                    'Surprise_prob': probabilities[6]
                })
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        return predictions

def predict_directory(model_path, image_dir, output_file=None):
    # 檢查文件和目錄
    if not os.path.exists(model_path):
        print(f"錯誤: 找不到模型文件 {model_path}")
        return []
        
    if not os.path.exists(image_dir):
        print(f"錯誤: 找不到圖片目錄 {image_dir}")
        return []
    
    predictor = FaceViTPredictor(model_path)
    
    # 支援的圖片格式
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # 取得所有支援的圖片文件
    image_paths = [
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in supported_formats
    ]
    
    if not image_paths:
        print(f"警告: 在 {image_dir} 中沒有找到支援的圖片文件")
        return []
    
    results = predictor.predict_batch(image_paths)
    
    # 保存結果到 CSV
    if output_file and results:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
    
    return results

def main():
    # 使用相對路徑
    model_path = "checkpoints/vit_best.pth"
    image_dir = os.path.join('..', 'data', 'Images', 'test')  # 調整為您的圖片目錄路徑
    output_file = 'predictions.csv'
    
    print(f"模型路徑: {os.path.abspath(model_path)}")
    print(f"圖片目錄: {os.path.abspath(image_dir)}")
    print(f"輸出文件: {os.path.abspath(output_file)}")
    
    # 執行預測
    results = predict_directory(
        model_path=model_path,
        image_dir=image_dir,
        output_file=output_file
    )
    
    # 輸出結果
    print(f"\nTotal Predictions: {len(results)}")
    if results:
        print("\nSample predictions:")
        for i, result in enumerate(results[:3]):  # 顯示前3個預測結果
            print(f"Prediction {i+1}:", result)
    else:
        print("\n沒有生成預測結果，請檢查:")
        print("1. 模型文件是否存在")
        print("2. 圖片目錄是否包含支援的圖片文件")
        print("3. 圖片文件是否可以正常打開")

if __name__ == "__main__":
    main()