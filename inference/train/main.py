import argparse
from inference.train.mmsiamese.train_from_3D import main as train_3D
from inference.train.mmsiamese.train_from_2D import main as train_2D
from torch.utils.tensorboard import SummaryWriter
import os
import yaml
from tensorboard import program

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default=r'C:\Users\jerem\OneDrive\Me\StudiumMaster\00_Semesterarbeit\Project\MMSiamese\configs\configs.yaml')
parser.add_argument(
    '--root', type=str,
    help='name of config file to load',
    default=r'C:\Users\jerem\OneDrive\Me\StudiumMaster\00_Semesterarbeit\Project\MMSiamese')
parser.add_argument(
    '--model_lid', type=str,
    help='name of config file to load',
    default='lidar_backbone')
parser.add_argument(
    '--model_im', type=str,
    help='name of config file to load',
    default='image_backbone')


if __name__ == "__main__":
    args = parser.parse_args()

    data_root = os.path.join(args.root, 'data', 'kitti')

    with open(args.fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        # logger.info('loaded params...')

    # Tensorboard Setup
    path = "logs"
    num_of_runs = len(os.listdir(path)) if os.path.exists(path) else 0
    path = os.path.join(path, f'run_{num_of_runs + 1}')
    tb_logger = SummaryWriter(path)

    port = 6006
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', path, '--port', str(port)])
    url = tb.launch()
    print(f"TensorBoard started at {url}")

    # load_ext tensorboard
    # tensorboard - -logdir logs - -port 6006

    lidar_raw = True
    if lidar_raw:
        train_3D(params=params['train'], tb_logger=tb_logger, data_root=data_root)
    else:
        train_2D(params=params['train'], tb_logger=tb_logger, data_root=data_root, save_model_lid=args.model_lid, save_model_im=args.model_im)

