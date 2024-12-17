import torch
from torchvision import transforms
import os
from PIL import Image
import pandas as pd
from approach.efficeient import EfficientEmoteNet

class EfficientEmotePredictor:
    def __init__(self, model_path, config=None):
        if config is None:
            self.config = self._get_default_config()
        else:
            self.config = config
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.transform = self._get_transforms()
        
        # 情绪标签映射
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
                self.image_size = 224  # EfficientNet-B0 默认输入大小
                self.num_classes = 7
        return DefaultConfig()
    
    def _load_model(self, model_path):
        # 创建模型
        model = EfficientEmoteNet(num_classes=self.config.num_classes).to(self.device)
        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
    def _get_transforms(self):
        return transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.Grayscale(num_output_channels=3),  # 确保输入为3通道
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def predict_image(self, image_path):
        """预测单张图片的情绪"""
        # 加载并预处理图片
        image = Image.open(image_path)
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 进行预测
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
        """批量预测多张图片"""
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
    # 检查文件和目录
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        return []
        
    if not os.path.exists(image_dir):
        print(f"错误: 找不到图片目录 {image_dir}")
        return []
    
    predictor = EfficientEmotePredictor(model_path)
    
    # 支持的图片格式
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # 获取所有支持的图片文件
    image_paths = [
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in supported_formats
    ]
    
    if not image_paths:
        print(f"警告: 在 {image_dir} 中没有找到支持的图片文件")
        return []
    
    results = predictor.predict_batch(image_paths)
    
    # 保存结果到 CSV
    if output_file and results:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    
    return results

def main():
    model_path = "checkpoints/efficient/efficient_62_0.6934.pth"  # 使用你训练好的最佳模型
    image_dir = "../data/Images/test"  # 调整为你的测试图片目录
    output_file = 'efficient_predictions.csv'
    
    print(f"Model path: {os.path.abspath(model_path)}")
    print(f"Image directory: {os.path.abspath(image_dir)}")
    print(f"Output file: {os.path.abspath(output_file)}")
    
    # 执行预测
    results = predict_directory(
        model_path=model_path,
        image_dir=image_dir,
        output_file=output_file
    )
    
    # 输出结果
    print(f"\nTotal Predictions: {len(results)}")
    if results:
        print("\nSample predictions:")
        for i, result in enumerate(results[:3]):  # 显示前3个预测结果
            print(f"Prediction {i+1}:", result)
    else:
        print("\nNo predictions generated, please check:")
        print("1. Model file exists")
        print("2. Image directory contains supported image files")
        print("3. Image files can be opened properly")

if __name__ == "__main__":
    main()