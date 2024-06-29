import torch
import torch.nn as nn
from torchvision.models import resnet18


class classifier_lidar(nn.Module):
    def __init__(self, hp=None, in_dim=None):
        super().__init__()
        self.hp = hp

        self.classifier_head = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # output: (N, 256, 12, 39)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # output: (N, 512, 6, 20)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),  # output: (N, 1024, 3, 10)
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), # output: (N, 1024, 1, 1)
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
            )

    def forward(self, x):
        return self.classifier_head(x)