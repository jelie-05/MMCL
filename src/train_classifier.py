import os

import torch
from tqdm import tqdm
from src.models.classifier_head import classifier_head
from src.datasets.kitti_loader.dataset_2D import DataGenerator
from src.utils.logger import tb_logger
import torch.nn as nn
import torch.optim as optim
from tensorboard import program
from src.utils.save_load_model import save_model
from src.utils.helper import gen_mixed_data


def create_tqdm_bar(iterable, desc):
    return tqdm(enumerate(iterable), total=len(iterable), ncols=150, desc=desc)


def main(args, project_root, pretrained_im, pretrained_lid, save_name, pixel_wise, masking, logger_launch='True'):

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    logger = tb_logger(project_root, args, save_name)

    """ Loader """
    data_root = os.path.join(project_root, args['data']['dataset_path'])
    train_gen = DataGenerator(data_root, 'train', args['data']['perturbation_file'])
    val_gen = DataGenerator(data_root, 'val', args['data']['perturbation_file'])
    train_loader = train_gen.create_data(args['data']['batch_size'], shuffle=True)
    val_loader = val_gen.create_data(args['data']['batch_size'], shuffle=False)

    model_im = pretrained_im.to(device)
    model_lid = pretrained_lid.to(device)
    model_lid.eval()
    model_im.eval()
    model_cls = classifier_head(model_lid=model_lid, model_im=model_im, pixel_wise=pixel_wise).to(device)

    """ Optimization """
    learning_rate = float(args['optimization']['lr'])
    epochs = int(args['optimization']['epochs'])

    loss_func = nn.BCELoss()

    optimizer = torch.optim.Adam(model_cls.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args['optimization']['scheduler_step'],
                                          gamma=args['optimization']['scheduler_gamma'])
    
    logger = tb_logger(args, project_root, save_name)
    if logger_launch:
        port = 6006
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', args['logging']['rel_path'], '--port', str(port)])
        url = tb.launch()
        print(f"TensorBoard started at {url}")

    for epoch in range(epochs):
        training_loss = 0
        validation_loss = 0

        # Training stage: set the model to training mode
        model_cls.train()

        # Create a progress bar for the training loop.
        training_loop = create_tqdm_bar(train_loader, desc=f'Training Epoch [{epoch + 1}/{epochs}]')

        for train_iteration, batch in training_loop:
            optimizer.zero_grad()  # Reset the gradients

            left_img_batch = batch['left_img'].to(device)
            depth_batch = batch['depth'].to(device)
            depth_neg = batch['depth_neg'].to(device)
            
            stacked_depth_batch, label_list, stacked_mask = gen_mixed_data(depth_batch, depth_neg, device, masking)

            N, C, H, W = left_img_batch.size()

            pred_cls = model_cls.forward(image=left_img_batch, lidar=stacked_depth_batch, H=H, W=W)
            pred_cls = pred_cls.squeeze(dim=1)

            loss = loss_func(pred_cls, label_list)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

            # Update the progress bar.
            training_loop.set_postfix(curr_train_loss="{:.8f}".format(training_loss / (train_iteration + 1)),
                                      val_loss="{:.8f}".format(validation_loss))

            # Update the tensorboard logger.
            tb_logger.add_scalar(f'classifier_{save_name}/train_loss', loss.item(),
                                 epoch * len(train_loader) + train_iteration)

        # Validation stage
        model_cls.eval()
        val_loop = create_tqdm_bar(val_loader, desc=f'Validation Epoch [{epoch + 1}/{epochs}]')

        with torch.no_grad():
            for val_iteration, val_batch in val_loop:
                left_img_batch = val_batch['left_img'].to(device)
                depth_batch = val_batch['depth'].to(device)
                depth_neg = val_batch['depth_neg'].to(device)
                
                stacked_depth_batch, label_val, stacked_mask = gen_mixed_data(depth_batch, depth_neg, device, masking)

                N, C, H, W = left_img_batch.size()
                pred_cls = model_cls.forward(image=left_img_batch, lidar=stacked_depth_batch, H=H, W=W).squeeze(dim=1)

                loss = loss_func(pred_cls, label_val)
                validation_loss += loss.item()

                # Update the progress bar.
                val_loop.set_postfix(val_loss="{:.8f}".format(validation_loss / (val_iteration + 1)))

                # Update the tensorboard logger.
                tb_logger.add_scalar(f'classifier_{save_name}/val_loss', loss.item(),
                                     epoch * len(val_loader) + val_iteration)

        scheduler.step()

        training_loss /= len(train_loader)
        validation_loss /= len(val_loader)
        tb_logger.add_scalar('training_cls_loss_epoch', training_loss,
                             epoch)
        tb_logger.add_scalar('validation_cls_loss_epoch', validation_loss,
                             epoch)
        # torch.cuda.empty_cache()

    name_cls = save_name + 'cls'
    save_model(model_cls, file_name=name_cls)
