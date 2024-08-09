import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, tensor1, tensor2, label):
        # Compute the Euclidean distance between the two tensors
        distances = F.pairwise_distance(tensor1.view(tensor1.size(0), -1), tensor2.view(tensor2.size(0), -1))
        print(distances.shape)

        diff = tensor1 - tensor2  # Step 1: Element-wise difference
        squared_diff = diff ** 2  # Step 2: Square the differences
        #sum_squared_diff = torch.sum(squared_diff, dim=1)  # Step 3: Sum across the channel dimension C
        sum_squared_diff = torch.sum(squared_diff, dim=1)
        sum_squared_diff = torch.sum(sum_squared_diff, dim=1)
        sum_squared_diff = torch.sum(sum_squared_diff, dim=3)
        print(sum_squared_diff.shape)
        distance = torch.sqrt(sum_squared_diff)
        distance = distance.mean()
        print(distance.shape)
        # Compute the contrastive loss
        loss = 0.5 * (label * distances.pow(2) + (1 - label) * F.relu(self.margin - distances).pow(2))
        loss2 = 0.5 * (label * distance.pow(2) + (1 - label) * F.relu(self.margin - distance).pow(2))

        return loss.mean(), loss2.mean()


# Example usage
batch_size, channels, height, width = 16, 3, 32, 32
tensor1 = torch.randn(batch_size, channels, height, width)
tensor2 = torch.randn(batch_size, channels, height, width)
labels = torch.randint(0, 2, (batch_size,)).float()  # 0 for dissimilar, 1 for similar pairs

contrastive_loss = ContrastiveLoss(margin=1.0)
loss, loss2 = contrastive_loss(tensor1, tensor2, labels)

print("Contrastive Loss:", loss.item())
print("Contrastive Loss:", loss2.item())
