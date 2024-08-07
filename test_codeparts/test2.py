import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset.kitti_loader.dataset_2D import DataGenerator


root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
kitti_path = os.path.join(root, 'data', 'kitti')
print(kitti_path)

train_gen = DataGenerator(kitti_path, 'train', perturb_filenames="perturbation_neg.csv")
train_loader = train_gen.create_data(64, shuffle=True)

device = torch.device("cuda:0")

for batch in train_loader:
    left_img_batch = batch['left_img'].to(device)
    depth_batch = batch['depth'].to(device)
    depth_neg = batch['depth_neg'].to(device)

    mask = (depth_neg > 0.0).int().squeeze(1)

    depth_neg = depth_neg.squeeze(1)
    print(depth_neg.shape)


    print(mask.shape)
    non_zero_counts = mask.flatten(1).sum(dim=1)
    print(non_zero_counts)

    a = depth_neg * mask
    print(a.shape)

    sum_loss_map = a.sum(dim=(1, 2))
    print(sum_loss_map)
    b = sum_loss_map/non_zero_counts
    print(b)
    print(b.mean())

