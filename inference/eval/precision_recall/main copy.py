import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import torch
import yaml
from src.utils.save_load_model import load_model_lidar, load_model_img
from src.models.mm_siamese import resnet18_2B_lid, resnet18_2B_im
from src.models.classifier_head import classifier_head
from src.utils.helper import load_checkpoint, load_checkpoint_cls, init_model

from precision_recall2 import evaluation
# from src.utils.save_load_model import load_model_lidar, load_model_img, load_model_cls
# import torch

parser = argparse.ArgumentParser()

parser.add_argument(
    '--config', type=str,
    help='name of config file to load',
    default='configs_resnet18_small_base.yaml')
parser.add_argument(
    '--save_name', type=str,
    help='name of lidar model to save',
    default='lidar_backbone')
parser.add_argument(
    '--pixel_wise', action='store_true', help='comparing pixel-wise distance')
parser.add_argument(
    '--perturbation', type=str,
    help='filename for perturbation',
    default='perturbation_neg_all.csv')
parser.add_argument(
    '--failure_mode', type=str,
    help='type of failure',
    default='labeled')

if __name__ == "__main__":
    args = parser.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
    kitti_path = os.path.join(root, 'data', 'kitti')
    configs_path = os.path.join(root, 'configs', args.config)

    with open(configs_path, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)

    save_name = args.save_name

    if not torch.cuda.is_available():
        device = torch.device('cpu')
        print('cuda is not available')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    encoder_im, encoder_lid = init_model(device=device, mode=params['meta']['backbone'], model_name=params['meta']['model_name'])

    classifier = classifier_head(model_im=encoder_im, model_lid=encoder_lid)
    classifier, epoch = load_checkpoint_cls(r_path=path, classifier= classifier)

    # Load pretrained model
    im_pretrained_path = os.path.join(root, 'outputs/models', args.name_im)
    lid_pretrained_path = os.path.join(root, 'outputs/models', args.name_lid)
    cls_pretrained_path = os.path.join(root, 'outputs/models', args.name_cls)

    im_pretrained = load_model_img(im_pretrained_path).eval()
    lid_pretrained = load_model_lidar(lid_pretrained_path).eval()
    cls_pretrained = load_model_cls(cls_pretrained_path, model_im=im_pretrained, model_lid=lid_pretrained, pixel_wise=args.pixel_wise).eval()

    device = torch.device("cuda:0")
    PR = evaluation(device=device, data_root=kitti_path, model_cls=cls_pretrained, perturb_file=args.perturbation, mode=args.failure_mode)
    print(PR)

# /home/ubuntu/Documents/students/Jeremialie/MMSiamese/.venv/bin/python /home/ubuntu/Documents/students/Jeremialie/MMSiamese/inference/eval/precision_recall/main.py