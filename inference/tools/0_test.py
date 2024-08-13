import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, tensor1, tensor2, label):

        # Ver 1
        distances = F.pairwise_distance(tensor1.view(tensor1.size(0), -1), tensor2.view(tensor2.size(0), -1))
        loss1 = 0.5 * (label * distances.pow(2) + (1 - label) * F.relu(self.margin - distances).pow(2))
        print(loss1.mean())

        # diff = tensor1 - tensor2  # Step 1: Element-wise difference
        # squared_diff = diff ** 2  # Step 2: Square the differences
        # sum_squared_diff = torch.sum(squared_diff, dim=1)
        # sum_squared_diff = torch.sum(sum_squared_diff, dim=1)
        # sum_squared_diff = torch.sum(sum_squared_diff, dim=1)
        # distance = torch.sqrt(sum_squared_diff)

        n,c,h,w = tensor1.shape
        flattened1 = tensor1.view(n,-1)
        flattened2 = tensor2.view(n, -1)
        diff = (flattened1-flattened2) ** 2
        distance = torch.sqrt(torch.sum(diff, dim=1))
        print(distance.shape)

        loss = 0.5 * (label * distance.pow(2) + (1 - label) * F.relu(self.margin - distance).pow(2))

        distance2 = torch.sqrt(torch.sum((tensor1 - tensor2) ** 2, dim=1))
        N, H_dist, W_dist = distance2.shape
        labels_broadcasted = label.view(N, 1, 1).expand(N, H_dist, W_dist)

        positive_loss = torch.pow(distance2, 2) * labels_broadcasted
        negative_loss = torch.pow(torch.clamp(self.margin - distance2, min=0.0), 2) * (1 - labels_broadcasted)
        loss2 = positive_loss + negative_loss

        distance2 = torch.sqrt(torch.sum((tensor1 - tensor2) , dim=1) ** 2)
        N, H_dist, W_dist = distance2.shape
        labels_broadcasted = label.view(N, 1, 1).expand(N, H_dist, W_dist)

        positive_loss = torch.pow(distance2, 2) * labels_broadcasted
        negative_loss = torch.pow(torch.clamp(self.margin - distance2, min=0.0), 2) * (1 - labels_broadcasted)
        loss3 = positive_loss + negative_loss

        return loss.mean(), loss2.mean(), loss3.mean()


# Example usage
batch_size, channels, height, width = 3, 4, 5, 5
tensor1 = torch.randn(batch_size, channels, height, width)
tensor2 = torch.randn(batch_size, channels, height, width)
labels = torch.randint(0, 2, (batch_size,)).float()  # 0 for dissimilar, 1 for similar pairs

contrastive_loss = ContrastiveLoss(margin=1.0)
loss, loss2, loss3 = contrastive_loss(tensor1, tensor2, labels)

print("Contrastive Loss:", loss.item())
print("Contrastive Loss:", loss2.item())
print("Contrastive Loss:", loss3.item())
