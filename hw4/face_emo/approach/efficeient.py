import torch
import torch.nn as nn
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet

class EfficientEmoteNet(nn.Module):
    def __init__(self, num_classes=7):
        super(EfficientEmoteNet, self).__init__()
        # 加载预训练的 EfficientNet-B0
        self.efficient_net = EfficientNet.from_pretrained('efficientnet-b0')
        
        # 修改第一层卷积以接受灰度图像（3通道，因为我们用 Grayscale(num_output_channels=3)）
        self.efficient_net._conv_stem = nn.Conv2d(
            3, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        
        # 获取特征提取器的输出维度
        num_features = self.efficient_net._fc.in_features
        
        # 替换分类头
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )
        
        self.efficient_net._fc = nn.Identity()

    def forward(self, x):
        # 特征提取
        features = self.efficient_net(x)
        # 分类
        out = self.classifier(features)
        return out

    def freeze_backbone(self, freeze=True):
        # 冻结/解冻主干网络
        for param in self.efficient_net.parameters():
            param.requires_grad = not freeze