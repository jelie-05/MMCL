import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import yaml
from src.models.classifier_head import classifier_head
from src.utils.helper import load_checkpoint, load_checkpoint_cls, init_model, init_opt
from precision_recall import pr_evaluation
from cka_analysis import cka_analysis


parser = argparse.ArgumentParser()
parser.add_argument(
    '--save_name', type=str,
    help='name of lidar model to save',
    default='lidar_backbone')
parser.add_argument(
    '--failure_mode', type=str,
    help='type of failure',
    default='labeled')
parser.add_argument(
    '--perturbation', type=str,
    help='type of evaluation',
    default='neg_master')
parser.add_argument(
    '--show_plot', action='store_true', help='enable printing or showing plots')
parser.add_argument(
    '--eval_metrics', action='store_true', help='enable evaluation metrics analysis')
parser.add_argument(
    '--cka', action='store_true', help='enable cka analysis')
parser.add_argument(
    '--other_epoch', action='store_true', help='enable analysis of other epochs')
parser.add_argument(
    '--epoch', type=int,
    help='the model to analyze',
    default=40)

if __name__ == "__main__":

    args = parser.parse_args()

    # Directory
    save_name = args.save_name
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    kitti_path = os.path.join(root, 'data', 'kitti')
    config_name = 'configs_' + save_name + '.yaml'
    configs_path = os.path.join(root, 'configs', config_name)
    perturbation_file = 'perturbation_' + args.perturbation + '.csv'
    # -

    # Saving directories
    output_dir = os.path.join(root, 'inference/eval', f"{args.save_name}_{args.perturbation}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory {output_dir} created.")

    if os.path.exists(output_dir) and not os.listdir(output_dir):
        save_dir = os.path.join(output_dir, 'run_1')
        print(f"Directory {save_dir} created.")
    else:
        num_folders = len([name for name in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, name))])
        save_dir = os.path.join(output_dir, f'run_{num_folders + 1}')
        print(f"Directory {save_dir} created.")
    # -

    # Configs loader, CUDA
    with open(configs_path, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
    if not torch.cuda.is_available():
        device = torch.device('cpu')
        print('cuda is not available')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
    # -

    # Initialized and Load Encoders
    encoder_im, encoder_lid = init_model(device=device, mode=params['meta']['backbone'],
                                         model_name=params['meta']['model_name'])
    opt_im, scheduler_im = init_opt(encoder_im, params['optimization'])
    opt_lid, scheduler_lid = init_opt(encoder_lid, params['optimization'])

    # Load pretrained model
    pretrained = params['meta']['pretrained_encoder']
    if pretrained:
        pretrained_name = params['meta']['pretrained_name']
        path_encoders = os.path.join(root, 'outputs_gpu', pretrained_name, 'models',
                                     f'{pretrained_name}_contrastive-latest.pth.tar')
        print(f"Use pretrained encoder from: {pretrained_name}")
    else:
        path_encoders = os.path.join(root, 'outputs_gpu', args.save_name, 'models',
                                     f'{args.save_name}_contrastive-latest.pth.tar')
        print("Not using pretrained encoder")

    encoder_im, encoder_lid, opt_im, opt_lid, epoch = load_checkpoint(r_path=path_encoders,
                                                                  encoder_im=encoder_im,
                                                                  encoder_lid=encoder_lid,
                                                                  opt_im=opt_im, opt_lid=opt_lid)
    encoder_im.eval()
    encoder_lid.eval()
    # -

    # Initialized and Load Classifier
    path_cls = os.path.join(root, 'outputs_gpu', args.save_name, 'models',
                            f'{args.save_name}_classifier-latest.pth.tar')
    classifier = classifier_head(model_im=encoder_im, model_lid=encoder_lid, model_name=params['meta']['model_name'])
    classifier, epoch_cls = load_checkpoint_cls(r_path=path_cls, classifier=classifier)
    classifier.to(device)
    classifier.eval()
    # -

    # Starting Evaluation
    if args.eval_metrics:
        PR = pr_evaluation(device=device, data_root=kitti_path, model_cls=classifier, mode=args.failure_mode,
                           perturbation_eval=perturbation_file, output_dir=save_dir, show_plot=args.show_plot)
    else:
        print("No Evalualtion Metrics Analysis")

    model_name = params['meta']['model_name']
    if '_aug' in save_name:
        title = f'{model_name} (Calibrated II)'
    elif '_noaug' in save_name:
        title = f'{model_name} (Calibrated I)'
    else:
        assert False, "Error: save_path must contain '_aug' or '_noaug'"

    if args.cka:
        cka_analysis(data_root=kitti_path, output_dir=save_dir, model_1=encoder_im, model_2=encoder_lid,
                     tag_1='Encoder Image', tag_2="Encoder LiDAR", title=title,
                     perturbation_eval=perturbation_file, show_plot=args.show_plot)
        cka_analysis(data_root=kitti_path, output_dir=save_dir, model_1=encoder_im, model_2=encoder_im,
                     tag_1='Encoder Image', tag_2="Encoder Image", title=title,
                     perturbation_eval=perturbation_file, show_plot=args.show_plot)
        cka_analysis(data_root=kitti_path, output_dir=save_dir, model_1=encoder_lid, model_2=encoder_lid,
                     tag_1='Encoder LiDAR', tag_2="Encoder LiDAR", title=title,
                     perturbation_eval=perturbation_file, show_plot=args.show_plot)
    else:
        print("No CKA Analysis")



    if args.other_epoch:
        num_epoch = args.epoch
        save_dir_epoch = output_dir = os.path.join(root, 'inference/eval', args.save_name, f'model_ep_{num_epoch}')
        path_encoders = os.path.join(root, 'outputs_gpu', args.save_name, 'models',
                                     f'{args.save_name}_contrastive-ep{num_epoch}.pth.tar')
        path_cls = os.path.join(root, 'outputs_gpu', args.save_name, 'models',
                                f'{args.save_name}_classifier-ep{num_epoch}.pth.tar')

        encoder_im, encoder_lid = init_model(device=device, mode=params['meta']['backbone'],
                                             model_name=params['meta']['model_name'])
        opt_im, scheduler_im = init_opt(encoder_im, params['optimization'])
        opt_lid, scheduler_lid = init_opt(encoder_lid, params['optimization'])
        encoder_im, encoder_lid, opt_im, opt_lid, epoch = load_checkpoint(r_path=path_encoders,
                                                                          encoder_im=encoder_im,
                                                                          encoder_lid=encoder_lid,
                                                                          opt_im=opt_im, opt_lid=opt_lid)
        encoder_im.eval()
        encoder_lid.eval()
        classifier = classifier_head(model_im=encoder_im, model_lid=encoder_lid, model_name=params['meta']['model_name'])
        classifier, epoch_cls = load_checkpoint_cls(r_path=path_cls, classifier=classifier)
        classifier.to(device)
        classifier.eval()

        PR_epoch = pr_evaluation(device=device, data_root=kitti_path, model_cls=classifier, mode=args.failure_mode,
                                 perturbation_eval=perturbation_file, output_dir=save_dir_epoch, show_plot=args.show_plot)

        cka_analysis(data_root=kitti_path, output_dir=save_dir_epoch, model_1=encoder_im, model_2=encoder_lid,
                     perturbation_eval=perturbation_file, show_plot=args.show_plot)