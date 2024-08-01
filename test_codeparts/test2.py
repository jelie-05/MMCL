import torch

torch.set_printoptions(profile="full")

distance_map = torch.randn(2,5,5)
print(distance_map)
mask = torch.randint(0, 2, distance_map.shape)
print(mask)
labels = torch.cat([torch.zeros(1), torch.ones(1)])

N, H_dist, W_dist = distance_map.shape
labels_broadcasted = labels.view(N, 1, 1).expand(N, H_dist, W_dist)

distance_map = distance_map * mask

# Calculate the contrastive loss
positive_loss = torch.pow(distance_map, 2) * labels_broadcasted
negative_loss = torch.pow(torch.clamp(4 - distance_map, min=0.0), 2) * (1 - labels_broadcasted)

non_zero_counts = torch.tensor([torch.count_nonzero(mask[i]).item() for i in range(mask.size(0))])
print(non_zero_counts.shape)

# Summing over the correct dimensions (H and W)
sum_distance_map = distance_map.sum(dim=(1,2))  # Summing over H and W dimensions
print(sum_distance_map.shape)
print(sum_distance_map)
non_zero_counts = non_zero_counts.to(distance_map.device)  # Move counts tensor to the same device

# Averaging for non-zero counts only
loss_contrastive = sum_distance_map / non_zero_counts # Broadcasting to match dimensions

print(loss_contrastive.shape)
print(loss_contrastive)