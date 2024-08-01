import torch
import torch.nn as nn
from inference.train.mmsiamese.calc_receptive_field import PixelwiseFeatureMaps


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=4.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output_im, output_lid, labels, model_im, H, W, pixel_wise, mask):
        
        # L2 Distances of feature embeddings
        dist_squared = (output_im - output_lid) ** 2
        summed = torch.sum(dist_squared, dim=1)
        distance = torch.sqrt(summed)

        # Map back each distance back into original size of image/lidar
        if pixel_wise:
            distance_map = distance.unsqueeze(1)
            distance_map = PixelwiseFeatureMaps(model=model_im, embeddings_value=distance_map, input_image_size=(H, W))
            distance_map = distance_map.assign_embedding_value().squeeze(1)

            N, H_dist, W_dist = distance_map.shape
            labels_broadcasted = labels.view(N, 1, 1).expand(N, H_dist, W_dist)  # Combine reshape and expand

            if mask is not None and mask.any():
                distance_map = distance_map * mask
                non_zero_counts = torch.tensor([torch.count_nonzero(mask[i]).item() for i in range(mask.size(0))])

                positive_loss = torch.pow(distance_map, 2) * labels_broadcasted
                negative_loss = torch.pow(torch.clamp(self.margin - distance_map, min=0.0), 2) * (1 - labels_broadcasted)
                loss_map = (positive_loss + negative_loss)

                # Summing over the correct dimensions (H and W)
                sum_loss_map = loss_map.sum(dim=(1, 2))  # Summing over H and W dimensions
                non_zero_counts = non_zero_counts.to(distance_map.device)  # Move counts tensor to the same device

                # Averaging for non-zero counts only
                loss_contrastive = sum_loss_map / non_zero_counts  # Broadcasting to match dimensions
            else:
                positive_loss = torch.pow(distance_map, 2) * labels_broadcasted
                negative_loss = torch.pow(torch.clamp(self.margin - distance_map, min=0.0), 2) * (1 - labels_broadcasted)
                loss_contrastive = (positive_loss + negative_loss)
                loss_contrastive = torch.mean(loss_contrastive)

        else:
            N, H_dist, W_dist = distance.shape
            labels_broadcasted = labels.view(N, 1, 1).expand(N, H_dist, W_dist)  # Combine reshape and expand

            positive_loss = torch.pow(distance, 2) * labels_broadcasted
            negative_loss = torch.pow(torch.clamp(self.margin - distance, min=0.0), 2) * (1 - labels_broadcasted)
            loss_contrastive = (positive_loss + negative_loss)
            loss_contrastive = torch.mean(loss_contrastive)

        return loss_contrastive
