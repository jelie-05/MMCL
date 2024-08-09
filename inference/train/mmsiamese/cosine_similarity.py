import torch
import torch.nn as nn
from inference.train.mmsiamese.calc_receptive_field import PixelwiseFeatureMaps


class CosineSim(nn.Module):
    def __init__(self, margin=4.0):
        super(CosineSim, self).__init__()
        self.margin = margin