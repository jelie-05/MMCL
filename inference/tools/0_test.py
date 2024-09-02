import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import src.models.resnet as resnet

import torch
import torchvision.models as models
import torch.nn as nn
from src.models.mm_siamese import resnet18_2B_lid, resnet18_2B_im



if __name__ == "__main__":
    model_name = 'resnet18_small_lid'
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    model = resnet.__dict__[model_name]().to(device)

    print(model)
