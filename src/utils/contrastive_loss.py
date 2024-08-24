import torch
import torch.nn as nn
from src.utils.calc_receptive_field import PixelwiseFeatureMaps
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=4.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output_im, output_lid, labels, model_im, H, W, pixel_wise, mask):

        # L2 Distances of feature embeddings
        distance = torch.sqrt(torch.sum((output_im - output_lid) ** 2, dim=1))

        # Map back each distance back into original size of image/lidar
        if pixel_wise:
           
            distance = PixelwiseFeatureMaps(model=model_im, embeddings_value=distance.unsqueeze(1),
                                                input_image_size=(H, W))
            distance = distance.assign_embedding_value().squeeze(1)
            N, H_dist, W_dist = distance.shape
            labels_broadcasted = labels.view(N, 1, 1).expand(N, H_dist, W_dist)

            mask = mask.float()

            patch_size = 16
            
            # Reshape the mask into patches of size 4x4
            mask_reshaped = mask.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
            
            # Check if each patch has at least one non-zero value
            patch_has_nonzero = mask_reshaped.sum(dim=(-1, -2)) > 0
            
            # Expand the result back to the original size
            patch_result = patch_has_nonzero.float().repeat_interleave(patch_size, dim=-1).repeat_interleave(patch_size, dim=-2)
            
            # Reshape back to the original mask size
            mask_analyzed = patch_result.view(N, 1, H, W)
            
            # lidar_mask_downsampled = F.interpolate(mask_analyzed, size=(H_dist, W_dist), mode='nearest').squeeze(1)

            # torch.set_printoptions(profile='full')
            # print(lidar_mask_downsampled[1,:,:])

            # M_norm = lidar_mask_downsampled.sum(dim=(1, 2), keepdim=True)  # Shape: (N, 1, 1)
            # epsilon = 1e-8
            # M_norm = M_norm + epsilon

            distance_final = distance * mask_analyzed

            positive_loss = torch.pow(distance_final, 2) * labels_broadcasted
            negative_loss = torch.pow(torch.clamp(self.margin - distance_final, min=0.0), 2) * (1 - labels_broadcasted)
            loss_contrastive = (positive_loss + negative_loss).mean()

        else:
            N, H_dist, W_dist = distance.shape
            labels_broadcasted = labels.view(N, 1, 1).expand(N, H_dist, W_dist)

            positive_loss = torch.pow(distance, 2) * labels_broadcasted
            negative_loss = torch.pow(torch.clamp(self.margin - distance, min=0.0), 2) * (1 - labels_broadcasted)
            loss_contrastive = (positive_loss + negative_loss).mean()

        return loss_contrastive
