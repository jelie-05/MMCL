import os
import torch
import numpy as np
from PIL import Image
from src.dataset.kitti_loader.Dataloader import Kittiloader
from src.dataset.kitti_loader.Transformer.custom_transformer import CustTransformer
from torch.utils.data import Dataset, DataLoader


class KittiDataset(Dataset):
    def __init__(self,
                 kittiDir,
                 mode,
                 perturb_filenames,
                 transform=None):
        self.mode = mode
        self.kitti_root = kittiDir
        self.perturb_filenames = perturb_filenames
        self.transform = transform

        # use left image by default
        self.kittiloader = Kittiloader(kittiDir, mode, perturb_filenames, cam=2)

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
                 high_gpu=True):
        self.phase = phase
        self.high_gpu = high_gpu
        self.perturb_filenames = perturb_filenames

        if not self.phase in ['train', 'test', 'val', 'check', 'checkval']:
            raise ValueError("Panic::Invalid phase parameter")
        else:
            pass

        transformer = CustTransformer(self.phase)
        self.dataset = KittiDataset(KittiDir,
                                    phase,
                                    perturb_filenames,
                                    transformer.get_transform())

    def create_data(self, batch_size, nthreads=0, shuffle=False):
        # use page locked gpu memory by default
        return DataLoader(self.dataset,
                          batch_size,
                          shuffle=shuffle,
                          num_workers=nthreads,
                          pin_memory=self.high_gpu)
