import os
from torch.utils.tensorboard import SummaryWriter

def tb_logger(args, root, name):
    path = args['logging']['rel_path']
    tag = args['logging']['tag']
    path_siamese = os.path.join(root, path, f'run_{name}_{tag}')
    logger = SummaryWriter(path_siamese)

    return logger
