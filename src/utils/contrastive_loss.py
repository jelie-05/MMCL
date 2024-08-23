import torch
import torch.nn as nn
from inference.train.mmsiamese.calc_receptive_field import PixelwiseFeatureMaps


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=4.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output_im, output_lid, labels, model_im, H, W, pixel_wise, mask):

        # L2 Distances of feature embeddings
        distance = torch.sqrt(torch.sum((output_im - output_lid) ** 2, dim=1))

        # Map back each distance back into original size of image/lidar
        if pixel_wise:
            distance_map = PixelwiseFeatureMaps(model=model_im, embeddings_value=distance.unsqueeze(1),
                                                input_image_size=(H, W))
            distance_map = distance_map.assign_embedding_value().squeeze(1)

            N, H_dist, W_dist = distance_map.shape
            labels_broadcasted = labels.view(N, 1, 1).expand(N, H_dist, W_dist)

            if mask is not None and mask.any():
                distance_map = distance_map * mask.squeeze(1)
                non_zero_counts = mask.flatten(1).sum(dim=1)

                positive_loss = torch.pow(distance_map, 2) * labels_broadcasted
                negative_loss = torch.pow(torch.clamp(self.margin - distance_map, min=0.0), 2) * (1 - labels_broadcasted)
                loss_map = positive_loss + negative_loss

                # sum_loss_map = loss_map.sum(dim=(1, 2))

                # Avoid division by zero
                # non_zero_counts[non_zero_counts == 0] = 1
                # non_zero_counts = non_zero_counts.to(distance_map.device)

                # loss_contrastive = sum_loss_map / non_zero_counts
                # loss_contrastive = loss_contrastive.mean()  # Mean over batch
                loss_contrastive = loss_map.mean()
            else:
                positive_loss = torch.pow(distance_map, 2) * labels_broadcasted
                negative_loss = torch.pow(torch.clamp(self.margin - distance_map, min=0.0), 2) * (1 - labels_broadcasted)
                loss_contrastive = (positive_loss + negative_loss).mean()
        else:
            N, H_dist, W_dist = distance.shape
            labels_broadcasted = labels.view(N, 1, 1).expand(N, H_dist, W_dist)

            positive_loss = torch.pow(distance, 2) * labels_broadcasted
            negative_loss = torch.pow(torch.clamp(self.margin - distance, min=0.0), 2) * (1 - labels_broadcasted)
            loss_contrastive = (positive_loss + negative_loss).mean()

        return loss_contrastive
