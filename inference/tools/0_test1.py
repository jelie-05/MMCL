import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=4.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output_im, output_lid, labels, H, W):

        # L2 Distances of feature embeddings
        distance = torch.sqrt(torch.sum((output_im - output_lid) ** 2, dim=1))

        N, H_dist, W_dist = distance.shape
        labels_broadcasted = labels.view(N, 1, 1).expand(N, H_dist, W_dist)
        print(labels_broadcasted)

        positive_loss = torch.pow(distance, 2) * labels_broadcasted
        print(positive_loss)
        negative_loss = torch.pow(torch.clamp(self.margin - distance, min=0.0), 2) * (1 - labels_broadcasted)
        print(negative_loss)
        loss_contrastive = (positive_loss + negative_loss).mean()

        return loss_contrastive

batch_size, channels, height, width = 3, 4, 5, 5
tensor1 = torch.randn(batch_size, channels, height, width)
tensor2 = torch.randn(batch_size, channels, height, width)
labels = torch.randint(0, 2, (batch_size,)).float()  # 0 for dissimilar, 1 for similar pairs

CL = ContrastiveLoss(margin=4.0)
loss = CL(tensor1, tensor2, labels, height, width)

print("Contrastive Loss:", loss.item())
print(labels)