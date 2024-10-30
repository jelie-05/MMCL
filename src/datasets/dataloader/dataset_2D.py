import torch
import numpy as np
from src.datasets.dataloader.kitti_dataloader import Kittiloader
from src.datasets.dataloader.Transformer.custom_transformer import CustTransformer
from torch.utils.data import Dataset, DataLoader
import pykitti
from src.datasets.dataloader.kitti_odom_dataloader.bin2depth import *


class KittiDataset(Dataset):
    def __init__(self,
                 kittiDir,
                 mode,
                 perturb_filenames,
                 transform=None,
                 augmentation=None):
        self.mode = mode
        self.kitti_root = kittiDir
        self.perturb_filenames = perturb_filenames
        self.transform = transform
        self.augmentation = augmentation

        # use left image by default
        if self.augmentation is not None:
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


class KITTIOdometryDataset(Dataset):
    def __init__(self, datadir, phase, perturb_filenames, cam_index=2, im_shape=(1242, 375), transform=None, augmentation=None):
        """
        Args:
            datadir (str): Path to the KITTI odometry dataset.
            sequence_list_file (str): Path to the text file containing sequence numbers (e.g., 'sequence_list.txt').
            cam_index (int): Camera index to project points onto (0, 1, 2, or 3).
            im_shape (tuple): Image shape (width, height) for filtering projected points.
            transform (callable, optional): Transformation to apply to images and points.
        """
        self.basedir = datadir
        self.cam_index = cam_index
        self.im_shape = im_shape
        self.transform = transform
        self.augmentation = augmentation


        sequence_list_file = os.path.join(datadir, f'sequence_list_{phase}.txt')
        self.perturb_path = os.path.join(datadir, f'{perturb_filenames}.csv')

        # Read sequences from the sequence list file
        with open(sequence_list_file, 'r') as f:
            self.sequences = [line.strip() for line in f if line.strip()]

        # Initialize pykitti datasets and accumulate all image and point indices
        self.data_indices = []
        for sequence in self.sequences:
            dataset = pykitti.odometry(datadir, sequence)
            num_frames = len(list(getattr(dataset, f"cam{cam_index}")))
            self.data_indices.extend([(sequence, idx) for idx in range(num_frames)])

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        # Retrieve the sequence and frame index
        sequence, frame_idx = self.data_indices[idx]

        # Load the specific sequence dataset and calibration parameters
        dataset = pykitti.odometry(self.basedir, sequence)
        T_cam_velo = getattr(dataset.calib, f"T_cam{self.cam_index:02d}_velo")
        R_rect = dataset.calib.R_rect_00
        P_rect = getattr(dataset.calib, f"P_rect_{self.cam_index:02d}")

        # Load the image and Velodyne points for the given frame
        rgb_image = list(getattr(dataset, f"cam{self.cam_index}"))[frame_idx]
        velodyne_points = list(dataset.velo)[frame_idx]
        name = f"{sequence}_{frame_idx:06d}"

        # Project Velodyne points onto the image
        depth, depth_neg = project_velodyne_to_camera(velodyne_points, self.im_shape, T_cam_velo, R_rect, P_rect, self.perturb_path,
                                                      name, augmentation=self.augmentation)

        # Create data item with a zero-padded name
        data_item = {
            'left_img': rgb_image.convert('RGB'),  # Ensure RGB format if necessary
            'depth': depth.astype(np.float32),
            'depth_neg': depth_neg.astype(np.float32),
            'name': name  # Sequence and zero-padded frame index, e.g., 02_000013
        }

        # Apply transformations if specified
        if self.transform:
            data_item = self.transform(data_item)

        # Convert image to tensor
        data_item['left_img'] = torch.tensor(np.array(data_item['left_img']), dtype=torch.float32).permute(2, 0, 1) / 255.0
        data_item['depth'] = torch.tensor(data_item['depth'], dtype=torch.float32)
        data_item['depth_neg'] = torch.tensor(data_item['depth_neg'], dtype=torch.float32)

        return data_item

class DataGenerator(object):
    def __init__(self,
                 datadir,
                 phase,
                 perturb_filenames,
                 high_gpu=True,
                 augmentation=None,
                 loader='kitti_raw'):
        self.phase = phase
        self.high_gpu = high_gpu
        self.perturb_filenames = perturb_filenames
        self.augmentation = augmentation
        self.loader = loader

        # if not self.phase in ['train', 'test', 'val', 'check', 'checkval']:
        #     raise ValueError("Panic::Invalid phase parameter")
        # else:
        #     pass

        transformer = CustTransformer(self.phase)

        if self.loader == 'kitti_raw':
            self.dataset = KittiDataset(datadir,
                                        self.phase,
                                        self.perturb_filenames,
                                        transformer.get_transform(),
                                        augmentation=self.augmentation)
        elif self.loader == 'kitti_odom':
            self.dataset = KITTIOdometryDataset(datadir, )

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
    def __init__(self, kittiDir, mode, perturb_filenames, transform=None, augmentation=None):
        super().__init__(kittiDir, mode, perturb_filenames, transform=transform,
                         augmentation=augmentation)  # Pass the transform

    def __getitem__(self, idx):
        data_item = super().__getitem__(idx)  # Get the full data item, already transformed
        left_img = data_item['left_img']  # Just extract the left image
        return left_img


class KittiDepthDataset(KittiDataset):
    def __init__(self, kittiDir, mode, perturb_filenames, transform=None, augmentation=None):
        super().__init__(kittiDir, mode, perturb_filenames, transform=transform,
                         augmentation=augmentation)  # Pass the transform

    def __getitem__(self, idx):
        data_item = super().__getitem__(idx)  # Get the full data item, already transformed
        depth = data_item['depth']  # Just extract the depth
        return depth

class KittiNegDataset(KittiDataset):
    def __init__(self, kittiDir, mode, perturb_filenames, transform=None, augmentation=None):
        super().__init__(kittiDir, mode, perturb_filenames, transform=transform,
                         augmentation=augmentation)  # Pass the transform

    def __getitem__(self, idx):
        data_item = super().__getitem__(idx)  # Get the full data item, already transformed
        depth_neg = data_item['depth_neg']  # Just extract the depth
        return depth_neg


def create_dataloaders(root, perturb_filenames, mode, batch_size, num_cores):
    # Initialize the transformer based on mode
    transformer = CustTransformer(mode)
    transform = transformer.get_transform()

    # Dataset for left images with the transform
    left_img_dataset = KittiLeftImageDataset(root, mode, perturb_filenames=perturb_filenames, transform=transform)
    dataloader_img = DataLoader(left_img_dataset, batch_size=batch_size, shuffle=False, num_workers=num_cores,
                                drop_last=True, pin_memory=True)

    # Dataset for depth with the transform
    depth_dataset = KittiDepthDataset(root, mode, perturb_filenames=perturb_filenames, transform=transform)
    dataloader_lid = DataLoader(depth_dataset, batch_size=batch_size, shuffle=False, num_workers=num_cores,
                                drop_last=True, pin_memory=True)

    # Dataset for depth with the transform
    depth_dataset = KittiNegDataset(root, mode, perturb_filenames=perturb_filenames, transform=transform)
    dataloader_neg = DataLoader(depth_dataset, batch_size=batch_size, shuffle=False, num_workers=num_cores,
                                drop_last=True, pin_memory=True)

    print("data is loaded")
    return dataloader_img, dataloader_lid, dataloader_neg
