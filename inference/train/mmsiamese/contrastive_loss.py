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

        # Map back the each distance back into original size of image/lidar
        if pixel_wise:
            distance = distance.unsqueeze(1)
            distance = PixelwiseFeatureMaps(model=model_im, embeddings_value=distance, input_image_size=(H, W))
            distance = distance.assign_embedding_value()
            distance = distance.squeeze(1)

        # Step 3: Broadcast labels to match the dimensions of the tensor
        N, H_dist, W_dist = distance.shape
        labels_broadcasted = labels.view(N, 1, 1).expand(N, H_dist, W_dist)  # Combine reshape and expand

        # Calculate the contrastive loss
        positive_loss = torch.pow(distance, 2) * labels_broadcasted
        negative_loss = torch.pow(torch.clamp(self.margin - distance, min=0.0), 2) * (1 - labels_broadcasted)

        loss_contrastive = (positive_loss + negative_loss)
        
        # Apply Masking to the loss function
        if mask is not None and mask.any():
            mask = mask.squeeze(1)
            loss_contrastive = loss_contrastive[mask.bool()]
        
        loss_contrastive = torch.mean(loss_contrastive)
        
        return loss_contrastive
