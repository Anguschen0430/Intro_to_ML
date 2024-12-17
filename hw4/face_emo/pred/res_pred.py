import torch
from torchvision import transforms
import os
from PIL import Image
import pandas as pd
from approach.ResEmoteNet import ResEmoteNet

class EmotionPredictor:
    def __init__(self, model_path, config=None):
        if config is None:
            self.config = self._get_default_config()
        else:
            self.config = config
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.transform = self._get_transforms()
        
        self.emotion_labels = {
            0: 'Angry',
            1: 'Disgust',
            2: 'Fear',
            3: 'Happy',
            4: 'Neutral',
            5: 'Sad',
            6: 'Surprise'
        }
        self.label_to_idx = {v.lower(): k for k, v in self.emotion_labels.items()}
    
    def _get_default_config(self):
        class DefaultConfig:
            def __init__(self):
                self.image_size = 64
                self.num_channels = 3
        return DefaultConfig()
    
    def _load_model(self, model_path):
        model = ResEmoteNet().to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)  # 直接加载 state_dict
        model.eval()
        return model
    # def _load_model(self, model_path):
    #     model = ResEmoteNet().to(self.device)
    #     checkpoint = torch.load(model_path, map_location=self.device)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     model.eval()
    #     return model
    def _get_transforms(self):
        return transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.Grayscale(num_output_channels=self.config.num_channels),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def predict_image(self, image_path):
        # 載入並預處理圖片
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 進行預測
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            all_probs = probabilities[0].tolist()  # 轉換所有機率為列表
            
        return predicted_class, all_probs
    
    def predict_batch(self, image_paths):
        predictions = []
        total = len(image_paths)
        for i, image_path in enumerate(image_paths, 1):
            try:
                # print(f"Processing image {i}/{total}: {os.path.basename(image_path)}")
                predicted_class, all_probs = self.predict_image(image_path)
                result = {
                    'filename': os.path.basename(image_path),
                    'label': predicted_class,
                    'emotion': self.emotion_labels[predicted_class],
                }
                for emotion_idx, prob in enumerate(all_probs):
                    result[f'{self.emotion_labels[emotion_idx]}_prob'] = round(prob , 10)
                
                predictions.append(result)
                
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                print(f"Error type: {type(e)}")
        return predictions

def predict_directory(model_path, image_dir, output_file=None):
    """
    預測整個目錄中的圖片並保存結果為 CSV 文件
    
    Args:
        model_path (str): 模型路徑
        image_dir (str): 圖片目錄
        output_file (str, optional): 輸出文件路徑
    """
    predictor = EmotionPredictor(model_path)
    
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = [
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in supported_formats
    ]
    
    results = predictor.predict_batch(image_paths)
    
    if output_file and results:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
    
    return results

def main():
    results = predict_directory(
        model_path='checkpoints/res/v3_nottry/res_56_0.63.pth',
        image_dir='../data/Images/test',
        output_file='../result/predictions.csv'
    )
    print(f"Total Predictions: {len(results)}")
    if results:
        print("Sample Output:")
        print(results[0])

if __name__ == "__main__":
    main()