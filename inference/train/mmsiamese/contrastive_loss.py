import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=4.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, labels):
        # Calculate the Euclidean distance and calculate the contrastive loss
        # distance = F.pairwise_distance(output1_flat, output2_flat, keepdim=True)

        # L2
        dist_squared = (output1-output2)**2
        summed = torch.sum(dist_squared, dim=1)

        distance = torch.sqrt(summed)
        N, H, W = distance.shape

        # Step 3: Broadcast labels to match the dimensions of the tensor
        labels_broadcasted = labels.view(N, 1, 1).expand(N, H, W)  # Combine reshape and expand

        # Calculate the contrastive loss
        positive_loss = torch.pow(distance, 2) * labels_broadcasted
        negative_loss = torch.pow(torch.clamp(self.margin - distance, min=0.0), 2) * (1 - labels_broadcasted)

        loss_contrastive = torch.mean(positive_loss + negative_loss)

        return loss_contrastive
