import argparse
from .mmsiamese.train import main as train_main
from torch.utils.tensorboard import SummaryWriter
import os
import yaml

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs/configs.yaml')


if __name__ == "__main__":
    args = parser.parse_args()

    path = "logs"
    num_of_runs = len(os.listdir(path)) if os.path.exists(path) else 0
    path = os.path.join(path, f'run_{num_of_runs + 1}')
    tb_logger = SummaryWriter(path)

    with open(args.fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        # logger.info('loaded params...')

    train_main(params=params['train'], tb_logger=tb_logger)