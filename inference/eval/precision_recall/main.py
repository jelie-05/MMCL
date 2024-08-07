import argparse
import os
from precision_recall import evaluation
from src.utils.save_load_model import load_model_lidar, load_model_img, load_model_cls
import torch

parser = argparse.ArgumentParser()

parser.add_argument(
    '--name_lid', type=str,
    help='name of lidar model to save',
    default='lidar_backbone')
parser.add_argument(
    '--name_im', type=str,
    help='name of image model file to save',
    default='image_backbone')
parser.add_argument(
    '--name_cls', type=str,
    help='name of clasfifier model file to save',
    default='classifier')
parser.add_argument(
    '--lidar_3D', action='store_true', help='train with 3D data as input')
parser.add_argument(
    '--pixel_wise', action='store_true', help='comparing pixel-wise distance')
parser.add_argument(
    '--perturbation', type=str,
    help='filename for perturbation',
    default='perturbation_neg.csv')

if __name__ == "__main__":
    args = parser.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
    kitti_path = os.path.join(root, 'data', 'kitti')

    # Load pretrained model
    im_pretrained_path = os.path.join(root, 'outputs/models', args.name_im)
    lid_pretrained_path = os.path.join(root, 'outputs/models', args.name_lid)
    cls_pretrained_path = os.path.join(root, 'outputs/models', args.name_cls)

    im_pretrained = load_model_img(im_pretrained_path)
    lid_pretrained = load_model_lidar(lid_pretrained_path)
    cls_pretrained = load_model_cls(cls_pretrained_path, model_im=im_pretrained, model_lid=lid_pretrained, pixel_wise=args.pixel_wise)

    device = torch.device("cuda:0")
    PR = evaluation(device=device, data_root=kitti_path, model_cls=cls_pretrained, perturb_file=args.perturbation)
    print(PR)

# python3 inference/eval/precision_recall/main.py --name_lid lidar_240719_full_1 --name_im image_240719_full_1 --name_cls cls_240719_full_1