import torch
from torchvision import transforms
import os
from PIL import Image
import pandas as pd
from torch import nn

class BaselinePredictor:
    def __init__(self, model_path, config=None):
        if config is None:
            self.config = self._get_default_config()
        else:
            self.config = config
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.transform = self._get_transforms()
        
        # Emotion label mapping
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
                self.image_size = 64  # Based on your training transform
                self.num_classes = 7
        return DefaultConfig()
    
    def _load_model(self, model_path):
        # Create model structure (import your BaselineModel class here)
        from approach.baseline import BaselineModel
        
        # Initialize model
        model = BaselineModel().to(self.device)
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        model.eval()
        return model
    
    def _get_transforms(self):
        return transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def predict_image(self, image_path):
        """Predict emotion for a single image"""
        # Load and preprocess image
        image = Image.open(image_path)
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
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
        """Predict emotions for multiple images"""
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
    # Check files and directories
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return []
        
    if not os.path.exists(image_dir):
        print(f"Error: Image directory not found at {image_dir}")
        return []
    
    predictor = BaselinePredictor(model_path)
    
    # Supported image formats
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # Get all supported image files
    image_paths = [
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in supported_formats
    ]
    
    if not image_paths:
        print(f"Warning: No supported image files found in {image_dir}")
        return []
    
    results = predictor.predict_batch(image_paths)
    
    # Save results to CSV
    if output_file and results:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
    
    return results

def main():
    # Use relative paths
    model_path = "checkpoints/baseline/base_198_1.00.pth"  # Adjust to your model path
    image_dir = "../data/Images/test"  # Adjust to your image directory
    output_file = "../result/output_base.csv"
    
    print(f"Model path: {os.path.abspath(model_path)}")
    print(f"Image directory: {os.path.abspath(image_dir)}")
    print(f"Output file: {os.path.abspath(output_file)}")
    
    # Run predictions
    results = predict_directory(
        model_path=model_path,
        image_dir=image_dir,
        output_file=output_file
    )
    
    # Output results
    print(f"\nTotal Predictions: {len(results)}")
    if results:
        print("\nSample predictions:")
        for i, result in enumerate(results[:3]):  # Show first 3 predictions
            print(f"Prediction {i+1}:", result)
    else:
        print("\nNo predictions generated. Please check:")
        print("1. Model file exists")
        print("2. Image directory contains supported image files")
        print("3. Image files can be opened properly")

if __name__ == "__main__":
    main()