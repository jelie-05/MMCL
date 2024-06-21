import torch
import torch.nn as nn
from torchvision.models import resnet18
import torch.nn.functional as F


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False # Freeze
            

class Image_Backbone(nn.Module):

    def __init__(self, hp=None):
        super().__init__()
        self.hp = hp

        model = resnet18(pretrained=True)
        set_parameter_requires_grad(model, feature_extracting=False) # True: freeze

        # Image Backbone: First 3 Block of ResNet
        backbone_im = torch.nn.Sequential(*(list(model.children())[0:6]))
        self.encoder_im = backbone_im

    def forward(self, input):
        output_im = self.encoder_im(input)

        return output_im


class Lidar_Backbone(nn.Module):

    def __init__(self, hp=None):
        super().__init__()
        self.hp = hp

        model = resnet18(pretrained=True)
        set_parameter_requires_grad(model, feature_extracting=False) # True: freeze

        # Change input channel for Lidar
        backbone_lid = torch.nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            *(list(model.children())[1:6]))

        self.encoder_lid = backbone_lid

    def forward(self, input):
        output_lid = self.encoder_lid(input)

        return output_lid
