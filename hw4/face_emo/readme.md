
A comprehensive facial emotion recognition system that leverages deep learning architectures for emotion analysis. This system supports multiple model architectures including VGG, ResNet, EfficientNet, and custom architectures like ResEmoteNet and FaceViT. Features include automated data preprocessing, flexible model training, and straightforward emotion prediction.

## Table of Contents

- [Facial Emotion Recognition Pipeline](#facial-emotion-recognition-pipeline)
- [Requirements](#requirements)
- [Run](#run)
- [Table of Contents](#table-of-contents)



## Requirements

- Python 3.7+
- PyTorch >= 1.8.0
- OpenCV
- Pandas
- NumPy
- Scikit-learn
- CUDA (optional, for GPU acceleration)

## Run
# 1. Data Preprocessing
Split your dataset into train/validation/test sets:
```bash
python preprocess/split.py
```
Generate CSV files with path information:
```bash
python preprocess/train2csv.py
```

# 2. Model Training
Ensure the CSV paths are properly set for:

Training set (train)
Validation set (val)
Test set (test)

Train a model (e.g., FaceViT):
```bash
python -m train_files.facevit_train
```
Available training scripts:



Training configuration and hyperparameters can be modified in the respective training scripts.
# 3. Prediction
Run emotion prediction on new images:
bashCopypython checkpoints/pred/efficient_pred.py
For different model architectures, use the corresponding prediction script:

base_pred.py - For baseline models
efficient_pred.py - For EfficientNet
facevit_pred.py - For FaceViT
res_pred.py - For ResNet models

```bash
python pred/efficient_pred.py
```
# 4. Bagging

```bash
python bagging.py
```
output
```bash
python emo_label.py
```

## Table of Contents

```bash
cd face-emo


Project Structure
Copyface_emo/
├── approach/          # Model architecture implementations
│   ├── ResEmoteNet.py
│   ├── baseline.py
│   ├── baseline_se.py
│   ├── efficient.py
│   ├── resnet.py
│   └── vgg.py
├── checkpoints/      # Model checkpoints and prediction scripts
│   └── pred/
│       ├── base_pred.py
│       ├── efficient_pred.py
│       ├── facevit_pred.py
│       └── res_pred.py
├── preprocess/       # Data preprocessing utilities
│   ├── split.py
│   ├── split_test.py
│   └── train2csv.py
└── train_files/      # Training implementation files
    ├── ResEmoteNet_train.py
    ├── baseline_SE_train.py
    ├── baseline_train.py
    ├── efficient_train.py
    ├── facevit_train.py
    ├── resnet18_train.py
    ├── resnet34_train.py
    └── vgg_train.py
