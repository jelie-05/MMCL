from torchvision.models import resnet18
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from inference.train.mmsiamese.calc_receptive_field import PixelwiseFeatureMaps
from src.dataset.kitti_loader.dataset_2D import DataGenerator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from src.utils.save_load_model import load_model_lidar, load_model_img

device = torch.device("cuda:0")
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
kitti_path = os.path.join(root, 'data', 'kitti')

# Load pretrained model
# im_pretrained_path = os.path.join(root, 'outputs/models', "image_240719_full_1")
# lid_pretrained_path = os.path.join(root, 'outputs/models', "lidar_240719_full_1")
im_pretrained_path = os.path.join(root, 'outputs/models', "240725_full_1_image")
lid_pretrained_path = os.path.join(root, 'outputs/models', "240725_full_1_lid")
im_pretrained = load_model_img(im_pretrained_path)
lid_pretrained = load_model_lidar(lid_pretrained_path)

model_im = im_pretrained.to(device)
model_lid = lid_pretrained.to(device)

eval_gen = DataGenerator(kitti_path, 'check')
eval_dataloader = eval_gen.create_data(8)

masking = True
pixel_wise = True

def image_lidar_visualization(image, lid_pos, lid_neg):
    # Permute
    lidar = lid_pos.permute(1, 2, 0).cpu().numpy()  # position (H, W, 1) = (ca. 100x600x1)
    lidar_neg = lid_neg.permute(1, 2, 0).cpu().numpy()
    img_np1 = image.permute(1, 2, 0).cpu().numpy()

    values_store = []
    values_store_neg = []

    lidar = np.squeeze(lidar)
    lidar_neg = np.squeeze(lidar_neg)

    for (i, j), value in np.ndenumerate(lidar):
        values_store.append([j, i, value])

    for (i, j), value in np.ndenumerate(lidar_neg):
        values_store_neg.append([j, i, value])

    values_store = np.array(values_store)
    values_store = np.delete(values_store, np.where(values_store[:, 2] == 0), axis=0)

    values_store_neg = np.array(values_store_neg)
    values_store_neg = np.delete(values_store_neg, np.where(values_store_neg[:, 2] == 0), axis=0)

    plt.figure(figsize=(15, 4.8))
    plt.imshow(img_np1, alpha=1.0)
    plt.scatter(values_store[:, 0], values_store[:, 1], c=values_store[:, 2], cmap='rainbow_r', alpha=0.5, s=3)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()

    # bool_store = (values_store[:,2] > 0.0).astype(int)
    #
    # plt.figure(figsize=(15, 4.8))
    # plt.imshow(img_np1, alpha=0.0)
    # plt.scatter(values_store[:, 0], values_store[:, 1], c=bool_store, cmap='gray_r', alpha=1, s=3)
    # plt.gca().set_facecolor('black')
    # plt.xticks([])
    # plt.yticks([])
    # plt.tight_layout()
    # plt.show()

    plt.figure(figsize=(15, 4.8))
    plt.imshow(img_np1)
    plt.scatter(values_store_neg[:, 0], values_store_neg[:, 1], c=values_store_neg[:, 2], cmap='rainbow_r', alpha=0.5,
                s=5)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()

def loss_map(output_im, output_lid, model_im, H, W, pixel_wise, mask):
    # L2 Distances of feature embeddings
    dist_squared = (output_im - output_lid) ** 2
    summed = torch.sum(dist_squared, dim=1)
    distance = torch.sqrt(summed)

    if pixel_wise:
        distance_map = distance.unsqueeze(1)
        distance_map = PixelwiseFeatureMaps(model=model_im, embeddings_value=distance_map, input_image_size=(H, W))
        distance_map = distance_map.assign_embedding_value().squeeze(1)
        distance_map = torch.pow(distance_map, 2)

        if mask is not None and mask.any():
            loss_contrastive_nomask = torch.mean(distance_map)
            distance_map = distance_map * mask
            loss_contrastive = torch.sum(distance_map) / torch.count_nonzero(mask)
        else:
            loss_contrastive = loss_contrastive_nomask = torch.mean(distance_map)
    else:
        distance_map = torch.pow(distance, 2)
        loss_contrastive = loss_contrastive_nomask = torch.mean(distance_map)

    return loss_contrastive, loss_contrastive_nomask, distance_map

model_im.eval()
model_lid.eval()

with torch.no_grad():
    for batch in eval_dataloader:
        left_img_batch = batch['left_img'].to(device)
        depth_batch = batch['depth'].to(device)
        depth_neg = batch['depth_neg'].to(device)

        i = 0

        image_sample = left_img_batch[i]
        lid_pos_sample = depth_batch[i]
        lid_neg_sample = depth_neg[i]

        # Prediction & Backpropagation
        pred_im = model_im.forward(image_sample.unsqueeze(0))
        pred_lid = model_lid.forward(lid_pos_sample.unsqueeze(0))
        pred_neg = model_lid.forward(lid_neg_sample.unsqueeze(0))

        N, C, H, W = left_img_batch.size()

        if masking:
            mask = (lid_pos_sample > 0.0).int()
            mask_neg = (lid_neg_sample > 0.0).int()
            # print(mask.shape)
        else:
            mask = None

        cl_loss, cl_loss_nomask, loss = loss_map(output_im=pred_im, output_lid=pred_lid, model_im=model_im, H=H, W=W,
                        pixel_wise=pixel_wise, mask=mask)
        print(f"cl_loss:{cl_loss}")
        print(f"cl_nomask:{cl_loss_nomask}")

        cl_neg, cl_nomask_neg, loss_neg = loss_map(output_im=pred_im, output_lid=pred_neg, model_im=model_im, H=H, W=W,
                            pixel_wise=pixel_wise, mask=mask_neg)
        print(f"cl_neg: {cl_neg}")
        print(f"cl_neg_nomask:{cl_nomask_neg}")

        # Convert the tensor to a NumPy array
        array = loss.cpu().numpy().squeeze(0)
        array_neg = loss_neg.cpu().numpy().squeeze(0)
        max_value = int(np.max(array_neg))
        image_lidar_visualization(image=image_sample, lid_pos=lid_pos_sample, lid_neg=lid_neg_sample)

        text = f"CL loss: {cl_loss:.3f}"
        text_neg = f"CL loss: {cl_neg:.3f}"
        # Plot the array
        plt.figure(figsize=(15, 6))  # Adjust figsize to match your image dimensions
        plt.imshow(array, cmap='viridis', vmin=0, vmax=max_value)  # You can change 'viridis' to any other colormap you prefer
        plt.axis('off')  # Optional: Turn off the axis
        plt.text(0.5, -0.1, text, ha='center', va='center', transform=plt.gca().transAxes, fontsize=25)
        plt.tight_layout()  # Adjusts the plot to fit the figure area
        plt.show()

        plt.figure(figsize=(15, 6))  # Adjust figsize to match your image dimensions
        plt.imshow(array_neg, cmap='viridis', vmin=0, vmax=max_value)  # You can change 'viridis' to any other colormap you prefer
        plt.axis('off')  # Optional: Turn off the axis
        plt.text(0.5, -0.1, text_neg, ha='center', va='center', transform=plt.gca().transAxes, fontsize=25)
        plt.tight_layout()  # Adjusts the plot to fit the figure area
        plt.show()
