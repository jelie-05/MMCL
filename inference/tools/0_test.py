from src.datasets.kitti_loader.dataset_2D import DataGenerator
import os
import multiprocessing
import torch


num_cores = min(multiprocessing.cpu_count(), 64)
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
eval_gen = DataGenerator(root, 'check', perturb_filenames='perturbation_neg_master.csv', augmentation=False)
eval_dataloader = eval_gen.create_data(batch_size=64, shuffle=False, nthreads=num_cores)
device = torch.device('cuda:0')

for batch in eval_dataloader:
    left_img_batch = batch['left_img'].to(device)  # batch of left image, id 02
    depth_batch = batch['depth'].to(device)  # the corresponding depth ground truth of given id
    depth_neg = batch['depth_neg'].to(device)
    depth_name = batch['name']