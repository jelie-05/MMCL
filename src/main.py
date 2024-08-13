import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from train_contrastive import main as train_contrastive
from train_classifier import main as train_cls
from torch.utils.tensorboard import SummaryWriter
import yaml
from tensorboard import program
from src.utils.save_load_model import load_model_lidar, load_model_img
from src.models.mm_siamese import resnet18_2B_lid, resnet18_2B_im
from src.models.classifier_head import classifier_head
from src.utils.helper import load_checkpoint


parser = argparse.ArgumentParser()
parser.add_argument(
    '--config', type=str,
    help='name of config file to load',
    default='configs_contrastive.yaml')
parser.add_argument(
    '--save_name', type=str,
    help='name of model for saving',
    default='dates')
parser.add_argument(
    '--pixel_wise', action='store_true', help='comparing pixel-wise distance')
parser.add_argument(
    '--masking', action='store_true', help='enable masking')
parser.add_argument(
    '--augmentation', action='store_true', help='enable augmentation for correct calibration')
parser.add_argument(
    '--model', type=str,
    help='type of model',
    default='mmsiamese')

if __name__ == "__main__":
    args = parser.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    print(f'project_root: {root}')
    configs_path = os.path.join(root, 'configs', args.config)
    kitti_path = os.path.join(root, 'data', 'kitti')

    # Load configs
    with open(configs_path, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)

    # Train Model
    train_contrastive(args=params, project_root=root, save_name=args.save_name, pixel_wise=args.pixel_wise, masking=args.masking, logger_launch='True', augmentation=args.augmentation)

    # Load pretrained model
    name_im = args.save_name + '_im'
    name_lid = args.save_name + '_lid'
    im_pretrained_path = os.path.join(root, 'outputs/models', name_im)
    lid_pretrained_path = os.path.join(root, 'outputs/models', name_lid)
    im_pretrained = load_model_img(im_pretrained_path).eval()
    lid_pretrained = load_model_lidar(lid_pretrained_path).eval()

    path = os.path.join(root, 'outputs/models', 'test_contrastive-latest.pth.tar')

    model_im, model_lid, model_cls, epoch = load_checkpoint(r_path=path, model_im=resnet18_2B_im(), model_lid=resnet18_2B_lid(), 
                                                            model_cls=classifier_head(model_im=resnet18_2B_im(), model_lid=resnet18_2B_lid()))

    # train_cls(args=params, project_root=root, pretrained_im=im_pretrained, pretrained_lid=lid_pretrained, save_name=args.save_name, 
    #           pixel_wise=args.pixel_wise, masking=args.masking, augmentation=args.augmentation)


