import os
import torch
import numpy as np
from PIL import Image
from src.datasets.kitti_loader.Dataloader import Kittiloader
from src.datasets.kitti_loader.Transformer.custom_transformer import CustTransformer
from torch.utils.data import Dataset, DataLoader


class KittiDataset(Dataset):
    def __init__(self,
                 kittiDir,
                 mode,
                 perturb_filenames,
                 transform=None,
                 augmentation=False):
        self.mode = mode
        self.kitti_root = kittiDir
        self.perturb_filenames = perturb_filenames
        self.transform = transform
        self.augmentation = augmentation

        # use left image by default
        if self.augmentation:
            print(f"Create augmentation for correct input. {self.augmentation}")
        else:
            print(f"No augmentation for correct input. {self.augmentation}")
            
        self.kittiloader = Kittiloader(kittiDir, mode, perturb_filenames, cam=2, augmentation=self.augmentation)

    def __getitem__(self, idx):
        # load an item according to the given index
        data_item = self.kittiloader.load_item(idx)     # get a set of data
        data_transed = self.transform(data_item)        # transform the set of data with CustTransformer.get_transform()

        return data_transed

    def __len__(self):
        return self.kittiloader.data_length()


class DataGenerator(object):
    def __init__(self,
                 KittiDir,
                 phase,
                 perturb_filenames,
                 high_gpu=True,
                 augmentation=False):
        self.phase = phase
        self.high_gpu = high_gpu
        self.perturb_filenames = perturb_filenames
        self.augmentation = augmentation

        # if not self.phase in ['train', 'test', 'val', 'check', 'checkval']:
        #     raise ValueError("Panic::Invalid phase parameter")
        # else:
        #     pass

        transformer = CustTransformer(self.phase)
        self.dataset = KittiDataset(KittiDir,
                                    self.phase,
                                    self.perturb_filenames,
                                    transformer.get_transform(),
                                    augmentation=self.augmentation)

    def create_data(self, batch_size, nthreads=0, shuffle=False):
        print(f'num_workers: {nthreads}')
        # use page locked gpu memory by default
        return DataLoader(self.dataset,
                          batch_size,
                          shuffle=shuffle,
                          num_workers=nthreads,
                          drop_last=True,  # Ensures all batches are the same size
                          pin_memory=self.high_gpu)


class KittiLeftImageDataset(KittiDataset):
    def __init__(self, kittiDir, mode, perturb_filenames, transform=None, augmentation=False):
        super().__init__(kittiDir, mode, perturb_filenames, transform=transform,
                         augmentation=augmentation)  # Pass the transform

    def __getitem__(self, idx):
        data_item = super().__getitem__(idx)  # Get the full data item, already transformed
        left_img = data_item['left_img']  # Just extract the left image
        return left_img


class KittiDepthDataset(KittiDataset):
    def __init__(self, kittiDir, mode, perturb_filenames, transform=None, augmentation=False):
        super().__init__(kittiDir, mode, perturb_filenames, transform=transform,
                         augmentation=augmentation)  # Pass the transform

    def __getitem__(self, idx):
        data_item = super().__getitem__(idx)  # Get the full data item, already transformed
        depth = data_item['depth']  # Just extract the depth
        return depth


def create_dataloaders(root, perturb_filenames, mode, batch_size, num_cores):
    # Initialize the transformer based on mode
    transformer = CustTransformer(mode)
    transform = transformer.get_transform()

    # Dataset for left images with the transform
    left_img_dataset = KittiLeftImageDataset(root, mode, perturb_filenames=perturb_filenames, transform=transform,
                                             augmentation=False)
    dataloader_img = DataLoader(left_img_dataset, batch_size=batch_size, shuffle=False, num_workers=num_cores,
                                drop_last=True, pin_memory=True)

    # Dataset for depth with the transform
    depth_dataset = KittiDepthDataset(root, mode, perturb_filenames=perturb_filenames, transform=transform,
                                      augmentation=False)
    dataloader_lid = DataLoader(depth_dataset, batch_size=batch_size, shuffle=False, num_workers=num_cores,
                                drop_last=True, pin_memory=True)
    print("data is loaded")
    return dataloader_img, dataloader_lid
