import torch
from tqdm import tqdm
import os
from src.models.mm_siamese import resnet18_2B_lid, resnet18_2B_im
from src.models.classifier_head import classifier_head
from src.datasets.kitti_loader.dataset_2D import DataGenerator
from src.utils.contrastive_loss import ContrastiveLoss as CL
from src.utils.helper import gen_mixed_data
from src.utils.logger import tb_logger
from src.utils.save_load_model import save_model
import torch.optim as optim
import torch.nn as nn
from tensorboard import program


def create_tqdm_bar(iterable, desc):
    return tqdm(enumerate(iterable), total=len(iterable), ncols=150, desc=desc)


def main(args, project_root, save_name, pixel_wise, masking, logger_launch='True', augmentation='False', train_classifier='True'):

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    logger = tb_logger(args['logging'], project_root, save_name)
    
    if logger_launch:
        port = 6006
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', args['logging']['rel_path'], '--port', str(port)])
        url = tb.launch()
        print(f"TensorBoard started at {url}")

    checkpoint_freq = 2
    tag = args['logging']['tag']
    tag_cls = args['logging_cls']['tag']

    batch_size = args['data']['batch_size']
    dataset_path = args['data']['dataset_path']
    perturbation_file = args['data']['perturbation_file']

    """ Optimization """
    learning_rate = float(args['optimization']['lr'])
    epochs = int(args['optimization']['epochs'])
    loss_func = CL(margin=args['optimization']['margin'])
    scheduler_step = args['optimization']['scheduler_step']
    scheduler_gamma = args['optimization']['scheduler_gamma']

    data_root = os.path.join(project_root, dataset_path)
    train_gen = DataGenerator(data_root, 'train', perturb_filenames=perturbation_file, augmentation=augmentation)
    val_gen = DataGenerator(data_root, 'val', perturb_filenames=perturbation_file, augmentation=augmentation)
    # train_gen = DataGenerator(data_root, 'check', perturb_filenames=perturbation_file, augmentation=augmentation)
    # val_gen = DataGenerator(data_root, 'check', perturb_filenames=perturbation_file, augmentation=augmentation)
    train_loader = train_gen.create_data(batch_size, shuffle=True)
    val_loader = val_gen.create_data(batch_size, shuffle=False)

    model_im = resnet18_2B_im().to(device)
    model_lid = resnet18_2B_lid().to(device)
    model_cls = classifier_head(model_lid=model_lid, model_im=model_im, pixel_wise=pixel_wise).to(device)

    optimizer_im = torch.optim.Adam(model_im.parameters(), learning_rate)
    scheduler_im = optim.lr_scheduler.StepLR(optimizer_im, step_size=scheduler_step,
                                             gamma=scheduler_gamma)

    optimizer_lid = torch.optim.Adam(model_lid.parameters(), learning_rate)
    scheduler_lid = optim.lr_scheduler.StepLR(optimizer_lid, step_size=scheduler_step,
                                              gamma=scheduler_gamma)
    
    optimizer = torch.optim.Adam(model_cls.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args['optimization_cls']['scheduler_step'],
                                        gamma=args['optimization_cls']['scheduler_gamma'])

    save_path = os.path.join(project_root, 'outputs/models', f'{save_name}_{tag}' + '-ep{epoch}.pth.tar')
    save_path_cls = os.path.join(project_root, 'outputs/models', f'{save_name}_{tag_cls}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(project_root, 'outputs/models', f'{save_name}_{tag}-latest.pth.tar')

    def save_checkpoint(epoch, curr_loss, tag='contrastive'):
        save_dict = {
            'model_im': model_im.state_dict(),
            'model_lid': model_lid.state_dict(),
            'model_cls': model_cls.state_dict(),
            'optimizer_im': optimizer_im.state_dict(),
            'optimizer_lid': optimizer_lid.state_dict(),
            'optimizer_cls': optimizer.state_dict(),
            'epoch': epoch,
            'train_loss': curr_loss,
            'batch_size': batch_size,
            'lr': learning_rate,
            'train_cls': train_classifier
        }
        torch.save(save_dict, latest_path)
        if (epoch + 1) % checkpoint_freq == 0:
            if tag == 'contrastive':
                torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))
            else:
                torch.save(save_dict, save_path_cls.format(epoch=f'{epoch + 1}'))

    for epoch in range(epochs):
        training_loss = 0
        validation_loss = 0

        # Training stage: set the model to training mode
        model_im.train()
        model_lid.train()

        training_loop = create_tqdm_bar(train_loader, desc=f'Training Epoch [{epoch + 1}/{epochs}]')

        for train_iteration, batch in training_loop:
            optimizer_im.zero_grad()
            optimizer_lid.zero_grad()

            left_img_batch = batch['left_img'].to(device)
            depth_batch = batch['depth'].to(device)
            depth_neg = batch['depth_neg'].to(device)

            stacked_depth_batch, label_list, stacked_mask = gen_mixed_data(depth_batch, depth_neg, device, masking)

            # Prediction & Backpropagation
            pred_im = model_im.forward(left_img_batch)
            pred_lid = model_lid.forward(stacked_depth_batch)

            # For pixel-wise comparison
            N, C, H, W = left_img_batch.size()

            # Calculating the loss
            loss = loss_func(output_im=pred_im, output_lid=pred_lid, labels=label_list, model_im=model_im, H=H, W=W,
                             pixel_wise=pixel_wise, mask=stacked_mask)
            loss.backward()
            optimizer_im.step()
            optimizer_lid.step()

            training_loss += loss.item()

            # Update the progress bar.
            training_loop.set_postfix(curr_train_loss="{:.8f}".format(training_loss / (train_iteration + 1)),
                                      val_loss="{:.8f}".format(validation_loss))

            # Update the tensorboard logger.
            logger.add_scalar(f'siamese_{save_name}/train_loss', loss.item(),
                                 epoch * len(train_loader) + train_iteration)
            
        training_loss /= len(train_loader)
        logger.add_scalar('training_loss_epoch', training_loss,
                        epoch)
        save_checkpoint(epoch, training_loss)

        # Validation stage
        model_im.eval()
        model_lid.eval()
        val_loop = create_tqdm_bar(val_loader, desc=f'Validation Epoch [{epoch + 1}/{epochs}]')

        with torch.no_grad():
            for val_iteration, val_batch in val_loop:

                left_img_batch = val_batch['left_img'].to(device)
                depth_batch = val_batch['depth'].to(device)
                depth_neg = val_batch['depth_neg'].to(device)

                stacked_depth_val, label_val, stacked_mask = gen_mixed_data(depth_batch, depth_neg, device, masking)

                pred_im = model_im.forward(left_img_batch)
                pred_lid = model_lid.forward(stacked_depth_val)

                N, C, H, W = left_img_batch.size()
                loss_val = loss_func(output_im=pred_im, output_lid=pred_lid, labels=label_val, model_im=model_im,
                                     H=H, W=W, pixel_wise=pixel_wise, mask=stacked_mask)
                validation_loss += loss_val.item()

                # Update the progress bar.
                val_loop.set_postfix(val_loss="{:.8f}".format(validation_loss / (val_iteration + 1)))

                # Update the tensorboard logger.
                logger.add_scalar(f'siamese_{save_name}/val_loss', loss_val.item(),
                                     epoch * len(val_loader) + val_iteration)

        # Epoch-wise calculation
        validation_loss /= len(val_loader)
        logger.add_scalar('validation_loss_epoch', validation_loss,
                             epoch)
        # Step the scheduler after each epoch
        scheduler_im.step()
        scheduler_lid.step()
        
    save_name_im = save_name + '_im'
    save_name_lid = save_name + '_lid'
    save_model(model_im, file_name=save_name_im)
    save_model(model_lid, file_name=save_name_lid)


    if train_classifier:
        model_im.to(device).eval()
        model_lid.to(device).eval()
        logger = tb_logger(root=project_root, args=args['logging_cls'], name=save_name)

        """ Optimization """
        learning_rate = float(args['optimization_cls']['lr'])
        epochs = int(args['optimization_cls']['epochs'])

        loss_func = nn.BCELoss()

        for epoch in range(epochs):
            cls_training_loss = 0
            cls_validation_loss = 0

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

                cls_training_loss += loss.item()

                # Update the progress bar.
                training_loop.set_postfix(curr_train_loss="{:.8f}".format(cls_training_loss / (train_iteration + 1)),
                                        val_loss="{:.8f}".format(cls_validation_loss))

                # Update the tensorboard logger.
                logger.add_scalar(f'classifier_{save_name}/train_loss', loss.item(),
                                    epoch * len(train_loader) + train_iteration)

            cls_training_loss /= len(train_loader)
            logger.add_scalar('training_cls_loss_epoch', cls_training_loss,
                                epoch)
            save_checkpoint(epoch, cls_training_loss, tag=tag_cls)
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
                    cls_validation_loss += loss.item()

                    # Update the progress bar.
                    val_loop.set_postfix(val_loss="{:.8f}".format(cls_validation_loss / (val_iteration + 1)))

                    # Update the tensorboard logger.
                    logger.add_scalar(f'classifier_{save_name}/val_loss', loss.item(),
                                        epoch * len(val_loader) + val_iteration)

            scheduler.step()

            
            cls_validation_loss /= len(val_loader)
            logger.add_scalar('validation_cls_loss_epoch', cls_validation_loss,
                                epoch)
            # torch.cuda.empty_cache()

        name_cls = save_name + '_cls'
        save_model(model_cls, file_name=name_cls)

        
