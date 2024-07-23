from torchvision.models import resnet18
import torch
from inference.train.mmsiamese.calc_receptive_field import PixelwiseFeatureMaps
from src.models.mm_siamese import resnet18_2B_lid, resnet18_2B_im
from src.dataset.kitti_loader_2D.dataset_2D import DataGenerator
from inference.train.mmsiamese.contrastive_loss import ContrastiveLoss as CL
import numpy as np
import matplotlib.pyplot as plt
import os

device = torch.device("cuda:0")
model_im = resnet18_2B_im().to(device)
model_lid = resnet18_2B_lid().to(device)

root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
kitti_path = os.path.join(root, '../data', 'kitti')
eval_gen = DataGenerator(kitti_path, 'val')
eval_dataloader = eval_gen.create_data(64)

loss_func = CL(margin=4)

with torch.no_grad():
    for batch in eval_dataloader:
        left_img_batch = batch['left_img'].to(device)
        depth_batch = batch['depth'].to(device)

        # Prediction & Backpropagation
        pred_im = model_im.forward(left_img_batch)
        pred_lid = model_lid.forward(depth_batch)
        label_list = torch.ones(len(depth_batch), device=device)

        # For pixel-wise comparison
        N, C, H, W = left_img_batch.size()
        pixel_im = PixelwiseFeatureMaps(model=model_im, embeddings_value=pred_im,
                                        input_image_size=(H, W))
        pred_im = pixel_im.assign_embedding_value()
        print(pred_im[0,0,:,:])
        pixel_lid = PixelwiseFeatureMaps(model=model_lid, embeddings_value=pred_lid,
                                         input_image_size=(H, W))
        pred_lid = pixel_lid.assign_embedding_value()

        loss = loss_func(output_im=pred_im, output_lid=pred_lid, labels=label_list)

        # Convert the tensor to a NumPy array
        array = loss[1,:,:].cpu().numpy()

        # Plot the array
        # plt.figure(figsize=(2, 1))  # Adjust figsize to match your image dimensions
        plt.imshow(array, cmap='viridis')  # You can change 'viridis' to any other colormap you prefer
        plt.axis('off')  # Optional: Turn off the axis
        plt.tight_layout()  # Adjusts the plot to fit the figure area
        plt.show()

