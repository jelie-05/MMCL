import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from train_resnet import main as train_resnet
import yaml
import torch
from src.utils.save_load_model import load_model_lidar, load_model_img
from src.models.mm_siamese import resnet18_2B_lid, resnet18_2B_im
from src.models.classifier_head import classifier_head
from src.utils.helper import load_checkpoint, load_checkpoint_cls, init_model


parser = argparse.ArgumentParser()
parser.add_argument(
    '--config', type=str,
    help='name of config file to load',
    default='configs_resnet18_small_base.yaml')
parser.add_argument(
    '--save_name', type=str,
    help='name of model for saving',
    default='dates')
parser.add_argument(
    '--pixel_wise', action='store_true', help='comparing pixel-wise distance')
parser.add_argument(
    '--masking', action='store_true', help='enable masking')
parser.add_argument(
    '--classifier', action='store_true', help='training directly classifier')


if __name__ == "__main__":
    args = parser.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    print(f'project_root: {root}')
    configs_path = os.path.join(root, 'configs', args.config)
    kitti_path = os.path.join(root, 'data', 'kitti')

    # Load configs
    with open(configs_path, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)

    if args.classifier:
        print('Training the classifier')
    else:
        print('Not training the classifier')

    curr_model_name = params['meta']['model_name']
    print(f'Training model with {curr_model_name}, model name: {args.save_name}')

    # Train Model
    mode = params['meta']['backbone']

    if mode == 'resnet':
        train_resnet(args=params, project_root=root, save_name=args.save_name, pixel_wise=args.pixel_wise, masking=args.masking, logger_launch='True',
                     train_classifier=args.classifier)
    elif mode == 'vit':
        print('vit')
    else:
        assert mode in ['resnet', 'vit'], 'backbone is not covered'


    # Load pretrained model
    # name_im = args.save_name + '_im'
    # name_lid = args.save_name + '_lid'
    # im_pretrained_path = os.path.join(root, 'outputs/models', name_im)
    # lid_pretrained_path = os.path.join(root, 'outputs/models', name_lid)
    # im_pretrained = load_model_img(im_pretrained_path).eval()
    # lid_pretrained = load_model_lidar(lid_pretrained_path).eval()

    path = os.path.join(root, 'outputs/models', f'{args.save_name}_contrastive-latest.pth.tar')


    if not torch.cuda.is_available():
        device = torch.device('cpu')
        print('cuda is not available')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
    encoder_im, encoder_lid = init_model(device=device, mode=params['meta']['backbone'], model_name=params['meta']['model_name'])

    classifier = classifier_head(model_im=encoder_im, model_lid=encoder_lid)
    classifier, epoch = load_checkpoint_cls(r_path=path, classifier= classifier)

    #python ./src/main.py --save_name 240829_test --config configs_resnet18_small.yaml --classifier
    # train_cls(args=params, project_root=root, pretrained_im=im_pretrained, pretrained_lid=lid_pretrained, save_name=args.save_name, 
    #           pixel_wise=args.pixel_wise, masking=args.masking, augmentation=args.augmentation)


