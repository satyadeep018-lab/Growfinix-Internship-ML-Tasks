import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image

CLASS_NAMES = ["Defective", "Perfect"]

class DefectResNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(DefectResNet, self).__init__()
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
        except Exception:
            self.backbone = models.resnet18(weights=None)
            
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        self.gradients = None
        self.activations = None
        self.hook_layers()
