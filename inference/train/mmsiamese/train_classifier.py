import torch
from tqdm import tqdm
from src.models.classifier_head import classifier_head
from src.dataset.kitti_loader.dataset_2D import DataGenerator
import torch.nn as nn
import torch.optim as optim

from src.utils.save_load_model import save_model


def create_tqdm_bar(iterable, desc):
    return tqdm(enumerate(iterable), total=len(iterable), ncols=150, desc=desc)

def main(params, data_root, tb_logger, pretrained_im, pretrained_lid, name_cls, pixel_wise, perturb_filename, name="default"):

    """ Data Loader """
    train_gen = DataGenerator(data_root, 'train', perturb_filename)
    train_loader = train_gen.create_data(int(params.get('batch_size')), shuffle=True)
    val_gen = DataGenerator(data_root, 'val', perturb_filename)
    val_loader = val_gen.create_data(int(params.get('batch_size')), shuffle=False) 

    """ Other hyperparams """
    learning_rate = float(params.get('lr'))
    epochs = int(params.get('epoch'))

    """ Initialize """
    loss_func = nn.BCELoss()
    device = torch.device(params.get('device'))

    model_im = pretrained_im.to(device)
    model_lid = pretrained_lid.to(device)
    model_lid.eval()
    model_im.eval()

    model_cls = classifier_head(model_lid=model_lid, model_im=model_im, pixel_wise=pixel_wise).to(device)
    optimizer = torch.optim.Adam(model_cls.parameters(), learning_rate)

    # Define the scheduler to decrease the learning rate by a factor of 0.1 every 30 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

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

            batch_length = len(depth_batch)
            half_length = batch_length // 2

            # Create shuffled label tensor
            label_tensor = torch.cat(
                [torch.zeros(half_length, device=device), torch.ones(half_length, device=device)])
            label_list = label_tensor[torch.randperm(label_tensor.size(0))]

            # Stack depth batches according to labels
            stacked_depth_batch = torch.where(label_list.unsqueeze(1).unsqueeze(2).unsqueeze(3).bool(), depth_batch,
                                              depth_neg)

            N, C, H, W = left_img_batch.size()

            pred_cls = model_cls.forward(image=left_img_batch, lidar=stacked_depth_batch,  H=H, W=W)
            pred_cls = pred_cls.squeeze(dim=1)

            loss = loss_func(pred_cls, label_list)
            loss.backward() 
            optimizer.step()
        
            training_loss += loss.item()

            # Update the progress bar.
            training_loop.set_postfix(curr_train_loss="{:.8f}".format(training_loss / (train_iteration + 1)),
                                      val_loss="{:.8f}".format(validation_loss))

            # Update the tensorboard logger.
            tb_logger.add_scalar(f'classifier_{name}/train_loss', loss.item(),
                                 epoch * len(train_loader) + train_iteration)

        # Validation stage
        model_cls.eval()
        val_loop = create_tqdm_bar(val_loader, desc=f'Validation Epoch [{epoch + 1}/{epochs}]')

        with torch.no_grad():
            for val_iteration, val_batch in val_loop:

                left_img_batch = val_batch['left_img'].to(device) 
                depth_batch = val_batch['depth'].to(device)
                depth_neg = val_batch['depth_neg'].to(device)

                batch_length = len(depth_batch)
                half_length = batch_length // 2

                # Create shuffled label tensor
                label_tensor = torch.cat(
                    [torch.zeros(half_length, device=device), torch.ones(half_length, device=device)])
                label_val = label_tensor[torch.randperm(label_tensor.size(0))]

                # Stack depth batches according to labels
                stacked_depth_batch = torch.where(label_val.unsqueeze(1).unsqueeze(2).unsqueeze(3).bool(), depth_batch,
                                                  depth_neg)

                N, C, H, W = left_img_batch.size()
                pred_cls = model_cls.forward(image=left_img_batch, lidar=stacked_depth_batch,  H=H, W=W).squeeze(dim=1)

                loss = loss_func(pred_cls, label_val)
                validation_loss += loss.item()

                # Update the progress bar.
                val_loop.set_postfix(val_loss="{:.8f}".format(validation_loss / (val_iteration + 1)))

                # Update the tensorboard logger.
                tb_logger.add_scalar(f'classifier_{name}/val_loss', loss.item(),
                                     epoch * len(val_loader) + val_iteration)
                
        scheduler.step()

        training_loss /= len(train_loader)
        validation_loss /= len(val_loader)
        tb_logger.add_scalar('training_cls_loss_epoch', training_loss,
                             epoch)
        tb_logger.add_scalar('validation_cls_loss_epoch', validation_loss,
                             epoch)
        # torch.cuda.empty_cache()

    save_model(model_cls, file_name=name_cls)
