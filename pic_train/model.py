import torch.nn as nn
from torchvision import models


def build_model(num_classes, device):
    """使用预训练ResNet18进行迁移学习"""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # 冻结特征提取层
    for param in model.parameters():
        param.requires_grad = False
    # 替换全连接层
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5), nn.Linear(num_features, 256), nn.ReLU(),
        nn.Dropout(0.3), nn.Linear(256, num_classes),
    )
    # 解冻layer4用于微调
    for param in model.layer4.parameters():
        param.requires_grad = True
    return model.to(device)
