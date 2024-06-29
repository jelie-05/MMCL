from .Dataloader import Kittiloader
from torch.utils.data import Dataset, DataLoader


class KittiDataset(Dataset):
    def __init__(self,
                 kittiDir,
                 mode,
                 transform=None):
        self.mode = mode
        self.kitti_root = kittiDir

        # use left image by default
        self.kittiloader = Kittiloader(kittiDir, mode, cam=2)

    def __getitem__(self, idx):
        # load an item according to the given index
        data_item = self.kittiloader.load_item(idx)     # get a set of data

        return data_item

    def __len__(self):
        return self.kittiloader.data_length()


class DataGenerator(object):
    def __init__(self,
                 KittiDir,
                 phase,
                 high_gpu=True):
        self.phase = phase
        self.high_gpu = high_gpu

        if not self.phase in ['train', 'test', 'val', 'check', 'checkval']:
            raise ValueError("Panic::Invalid phase parameter")
        else:
            pass

        self.dataset = KittiDataset(KittiDir,
                                    phase)

    def create_data(self, batch_size, nthreads=0, shuffle=False):
        # use page locked gpu memory by default
        return DataLoader(self.dataset,
                          batch_size,
                          shuffle=shuffle,
                          num_workers=nthreads,
                          pin_memory=self.high_gpu)
