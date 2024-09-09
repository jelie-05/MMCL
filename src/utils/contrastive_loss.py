import torch
import torch.nn as nn
from src.utils.calc_receptive_field import PixelwiseFeatureMaps
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=4.0, patch_size=16):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin


    def forward(self, output_im, output_lid, labels, model_im, H, W, vit, mask):
        # L2 Distances of feature embeddings

        if vit:
            # Input (B, N, D). Comparing each embedding (1, D)-vectors
            distances = F.pairwise_distance(output_im, output_lid, p=2)  # Shape (B, N)
            distances_mean = distances.mean(dim=1)  # Shape (B,)

            positive_loss = torch.pow(distances_mean, 2) * labels
            negative_loss = torch.pow(torch.clamp(self.margin - distances_mean, min=0.0), 2) * (1 - labels)

            loss_contrastive = (positive_loss + negative_loss).mean()

        else:
            # Input (B, C, H, W). Comparing each pixel (across C)
            distance = torch.sqrt(torch.sum((output_im - output_lid) ** 2, dim=1))
            N, H_dist, W_dist = distance.shape
            labels_broadcasted = labels.view(N, 1, 1).expand(N, H_dist, W_dist)

            positive_loss = torch.pow(distance, 2) * labels_broadcasted
            negative_loss = torch.pow(torch.clamp(self.margin - distance, min=0.0), 2) * (1 - labels_broadcasted)
            loss_contrastive = (positive_loss + negative_loss).mean()

        return loss_contrastive
