import argparse
from mmsiamese.train_from_3D import main as train_3D
from mmsiamese.train_from_2D import main as train_2D
from mmsiamese.classifier_train import main as train_cls
from torch.utils.tensorboard import SummaryWriter
import os
import yaml
from tensorboard import program
from src.utils.save_load_model import load_model_lidar, load_model_img

import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs.yaml')
parser.add_argument(
    '--model_lid', type=str,
    help='name of lidar model to save',
    default='lidar_backbone')
parser.add_argument(
    '--model_im', type=str,
    help='name of image model file to save',
    default='image_backbone')
parser.add_argument(
    '--name_cls', type=str,
    help='name of clasfifier model file to save',
    default='classifier')
parser.add_argument(
    '--lidar_3D', action='store_true', help='train with 3D data as input')


if __name__ == "__main__":
    args = parser.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    configs_path = os.path.join(root, 'configs', args.fname)
    kitti_path = os.path.join(root, 'data', 'kitti')

    with open(configs_path, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        # logger.info('loaded params...')

    # Tensorboard Setup
    path = "logs"
    num_of_runs = len(os.listdir(path)) if os.path.exists(path) else 0
    path_siamese = os.path.join(path, f'run_{num_of_runs + 1}')
    tb_logger = SummaryWriter(path_siamese)

    port = 6006
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', path, '--port', str(port)])
    url = tb.launch()
    print(f"TensorBoard started at {url}")

    # load_ext tensorboard
    # tensorboard --logdir logs --port 6006
    
    if args.lidar_3D:
        train_3D(params=params['train'], tb_logger=tb_logger, data_root=kitti_path, save_model_lid=args.model_lid, save_model_im=args.model_im)
    else:
        train_2D(params=params['train'], tb_logger=tb_logger, data_root=kitti_path, save_model_lid=args.model_lid, save_model_im=args.model_im)

    # Load pretrained model
    im_pretrained_path = os.path.join(root, 'models', args.model_im)
    lid_pretrained_path = os.path.join(root, 'models', args.model_lid)
    im_pretrained = load_model_img(im_pretrained_path)
    lid_pretrained = load_model_lidar(lid_pretrained_path)

    # Tensorboard Setup
    #path = "logs"
    num_of_runs = len(os.listdir(path)) if os.path.exists(path) else 0
    path_cls = os.path.join(path, f'run_cls{num_of_runs + 1}')
    tb_logger_cls = SummaryWriter(path_cls)

    train_cls(params=params['train_cls'], data_root=kitti_path, tb_logger=tb_logger_cls, 
              pretrained_im=im_pretrained, pretrained_lid=lid_pretrained, name_cls=args.name_cls)


