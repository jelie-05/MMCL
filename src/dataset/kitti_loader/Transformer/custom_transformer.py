import torch
import numpy as np
import torchvision.transforms as transforms
import src.dataset.kitti_loader.Transformer.custom_methods as augmethods
from .base_transformer import BaseTransformer


class CustTransformer(BaseTransformer):
    """
    An example of Custom Transformer.
    This class should work with custom transform methods which defined in custom_methods.py
    """
    def __init__(self, phase):
        BaseTransformer.__init__(self, phase)
        if not self.phase in ["train", "test", "val", "check", 'checkval']:
            raise ValueError("Panic::Invalid phase parameter")
        else:
            pass

    def get_joint_transform(self):
        if self.phase == "train" or self.phase == "check":
            return transforms.Compose([augmethods.TransToPIL(),
                                       augmethods.RandomHorizontalFlip()])
        else:
            return transforms.Compose([augmethods.TransToPIL()])

    def get_img_transform(self):
        if self.phase == "train" or self.phase == "check":
            return transforms.Compose([augmethods.ImgAug(),
                                       augmethods.RandomHorizontalFlip(),
                                       augmethods.ToTensor("Img"),
                                       augmethods.Scale("Img", [188, 621])])

        else:
            return transforms.Compose([augmethods.ImgAug(),
                                       augmethods.ToTensor("Img"),
                                       augmethods.Scale("Img", [188, 621])])

    def get_depth_transform(self):
        return transforms.Compose([augmethods.ToTensor("depth"),
                                   augmethods.Scale("depth", [188, 621]),
                                   augmethods.RandCrop(),
                                   augmethods.InputNormalize()])
