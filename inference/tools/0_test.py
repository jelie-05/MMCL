import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import src.models.resnet as resnet

if __name__ == "__main__":
    model_name = 'resnet18_all_im'
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    model = resnet.__dict__[model_name]().to(device)

    print(model)