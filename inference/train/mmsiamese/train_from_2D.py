import torch
from tqdm import tqdm
from src.models.mm_siamese import lidar_backbone, image_backbone
from src.dataset.kitti_loader_2D.dataset_2D import DataGenerator
from .contrastive_loss import ContrastiveLoss as CL
from inference.train.mmsiamese.calc_receptive_field import PixelwiseFeatureMaps
from src.utils.save_load_model import save_model
import torch.optim as optim


def create_tqdm_bar(iterable, desc):
    return tqdm(enumerate(iterable), total=len(iterable), ncols=150, desc=desc)


def main(params, data_root, tb_logger, save_model_im, save_model_lid, pixel_wise, masking, name="default"):

    """ Data Loader """
    train_gen = DataGenerator(data_root, 'train')
    train_loader = train_gen.create_data(int(params.get('batch_size')), shuffle=True)
    val_gen = DataGenerator(data_root, 'val')
    val_loader = val_gen.create_data(int(params.get('batch_size')), shuffle=False)

    """ Other hyperparams """
    learning_rate = float(params.get('lr'))
    epochs = int(params.get('epoch'))

    """ Initializing """
    loss_func = CL(margin=(params.get('margin')))
    device = torch.device(params.get('device'))

    model_im = image_backbone().to(device)
    model_lid = lidar_backbone().to(device)

    optimizer_im = torch.optim.Adam(model_im.parameters(), learning_rate)
    optimizer_lid = torch.optim.Adam(model_lid.parameters(), learning_rate)

    # Define the scheduler to decrease the learning rate by a factor of 0.1 every 30 epochs
    scheduler_im = optim.lr_scheduler.StepLR(optimizer_im, step_size=30, gamma=0.1)
    scheduler_lid = optim.lr_scheduler.StepLR(optimizer_lid, step_size=30, gamma=0.1)

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

            # Calculate Mask
            if masking:
                mask = depth_batch != 0
                mask = torch.tensor(mask.clone().detach().bool(), dtype=torch.bool)
            else:
                mask = None

            # Assign label randomly to each component of the batch
            batch_length = len(depth_batch)
            half_length = batch_length // 2
            label_tensor = torch.cat(
                [torch.zeros(half_length, device=device), torch.ones(half_length, device=device)])
            label_list = label_tensor[torch.randperm(label_tensor.size(0))]

            # Stack depth batches according to labels (depth_batch or depth_neg)
            stacked_depth_batch = torch.where(label_list.unsqueeze(1).unsqueeze(2).unsqueeze(3).bool(), depth_batch,
                                              depth_neg)

            # Prediction & Backpropagation
            pred_im = model_im.forward(left_img_batch)
            pred_lid = model_lid.forward(stacked_depth_batch)

            # For pixel-wise comparison
            N, C, H, W = left_img_batch.size()

            # Calculating the loss
            loss = loss_func(output_im=pred_im, output_lid=pred_lid, labels=label_list, model_im=model_im, H=H, W=W, pixel_wise=pixel_wise, mask=mask)
            print(loss)
            loss.backward()
            optimizer_im.step()
            optimizer_lid.step()
            
            training_loss += loss.item()

            # Update the progress bar.
            training_loop.set_postfix(curr_train_loss="{:.8f}".format(training_loss / (train_iteration + 1)),
                                      val_loss="{:.8f}".format(validation_loss))

            # Update the tensorboard logger.
            tb_logger.add_scalar(f'siamese_{name}/train_loss', loss.item(),
                                 epoch * len(train_loader) + train_iteration)
        
        # Step the scheduler after each epoch
        scheduler_im.step()
        scheduler_lid.step()

        # Validation stage
        model_im.eval()
        model_lid.eval()
        val_loop = create_tqdm_bar(val_loader, desc=f'Validation Epoch [{epoch + 1}/{epochs}]')

        with torch.no_grad():
            for val_iteration, val_batch in val_loop:

                left_img_batch = val_batch['left_img'].to(device)
                depth_batch = val_batch['depth'].to(device)
                depth_neg = val_batch['depth_neg'].to(device)

                # Calculate Mask
                if masking:
                    mask = depth_batch != 0
                    mask = torch.tensor(mask.clone().detach().bool(), dtype=torch.bool)
                else:
                    mask = None

                batch_length = len(depth_batch)
                half_length = batch_length // 2

                # Create shuffled label tensor directly on the specified device
                label_tensor_val = torch.cat(
                    [torch.zeros(half_length, device=device), torch.ones(half_length, device=device)])
                label_val = label_tensor_val[torch.randperm(label_tensor_val.size(0))]

                # Stack depth batches according to labels
                stacked_depth_val = torch.where(label_val.unsqueeze(1).unsqueeze(2).unsqueeze(3).bool(), depth_batch,
                                                  depth_neg)

                pred_im = model_im.forward(left_img_batch)
                pred_lid = model_lid.forward(stacked_depth_val)

                N, C, H, W = left_img_batch.size()
                loss_val = loss_func(output_im=pred_im, output_lid=pred_lid, labels=label_val, model_im=model_im, H=H, W=W, pixel_wise=pixel_wise, mask=mask)
                validation_loss += loss_val.item()

                # Update the progress bar.
                val_loop.set_postfix(val_loss="{:.8f}".format(validation_loss / (val_iteration + 1)))

                # Update the tensorboard logger.
                tb_logger.add_scalar(f'siamese_{name}/val_loss', loss_val.item(),
                                     epoch * len(val_loader) + val_iteration)

        # Epoch-wise calculation
        training_loss /= len(train_loader)
        validation_loss /= len(val_loader)
        tb_logger.add_scalar('training_loss_epoch', training_loss,
                             epoch)
        tb_logger.add_scalar('validation_loss_epoch', validation_loss,
                             epoch)
        torch.cuda.empty_cache()

    save_model(model_im, file_name=save_model_im)
    save_model(model_lid, file_name=save_model_lid)
