import torch
from tqdm import tqdm
import sys

# Add the src directory to the system path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.models.mm_siamese import lidar_backbone, image_backbone
from src.dataset.kitti_loader_3D.dataset_3D import DataGenerator
from inference.train.mmsiamese.contrastive_loss import ContrastiveLoss as CL
from inference.train.mmsiamese.utils import lidar_batch_transform
from src.dataset.kitti_loader_3D.Dataloader.bin2depth import get_calibration_files
from src.utils.save_load_model import save_model


def create_tqdm_bar(iterable, desc):
    return tqdm(enumerate(iterable), total=len(iterable), ncols=150, desc=desc)

def main(params, data_root, tb_logger, save_model_im, save_model_lid, name="default"):

    """
    Train the classifier for a number of epochs.
    """

    """ Data Loader """
    train_gen = DataGenerator(data_root, 'train')
    train_loader = train_gen.create_data(int(params.get('batch_size')), shuffle=True)
    val_gen = DataGenerator(data_root, 'val')
    val_loader = val_gen.create_data(int(params.get('batch_size')), shuffle=False)

    """ Loss Function """
    loss_func = CL(float(params.get('margin')))

    """ Device """
    device = torch.device(params.get('device'))

    """ Other hyperparams """
    learning_rate = float(params.get('lr'))
    epochs = int(params.get('epoch'))

    model_im = image_backbone().to(device)
    model_lid = lidar_backbone().to(device)

    optimizer_im = torch.optim.Adam(model_im.parameters(), learning_rate)
    optimizer_lid = torch.optim.Adam(model_lid.parameters(), learning_rate)

    for epoch in range(epochs):

        training_loss = 0
        validation_loss = 0

        # Training stage, where we want to update the parameters.
        model_im.train()  # Set the model to training mode
        model_lid.train()

        # Create a progress bar for the training loop.
        training_loop = create_tqdm_bar(train_loader, desc=f'Training Epoch [{epoch + 1}/{epochs}]')
        for train_iteration, batch in training_loop:
            optimizer_im.zero_grad()  # Reset the gradients - VERY important! Otherwise they accumulate.
            optimizer_lid.zero_grad()

            # Not yet as torch tensor
            left_img_batch = batch['left_img'].to(device)  # batch of left image, id 02
            velo_points = batch['velo']
            path = batch['cam_path'][0] # TODO: Still assuming that all calibration file same!!!
            cam2cam, velo2cam = get_calibration_files(calib_dir=path)

            size = left_img_batch[0].size()   # ASSUME: same image size for all batch

            depth_batch, depth_neg = lidar_batch_transform(cam2cam=cam2cam, velo2cam=velo2cam,lidar_batch=velo_points, im_shape=[size[2],size[1]]) # height width
            print(type(depth_batch))

            # Image to torch tensor
            # image lidar transforms

            depth_batch, depth_neg = depth_batch.float().to(device), depth_neg.float().to(device)

            batch_length = len(depth_batch)
            half_length = batch_length // 2

            # Create shuffled label tensor directly on the specified device
            label_tensor = torch.cat([torch.zeros(half_length, device=device), torch.ones(half_length, device=device)])

            # Shuffle the tensor on the GPU
            label_list = label_tensor[torch.randperm(label_tensor.size(0))]

            # Stack depth batches according to labels
            stacked_depth_batch = torch.where(label_list.unsqueeze(1).unsqueeze(2).unsqueeze(3).bool(), depth_batch,
                                              depth_neg)

            pred_im = model_im.forward(left_img_batch)
            pred_lid = model_lid.forward(stacked_depth_batch)

            loss = loss_func(pred_im, pred_lid, label_list)
            loss.backward()  # Stage 2: Backward().
            optimizer_im.step()  # Stage 3: Update the parameters.
            optimizer_lid.step()

            training_loss += loss.item()

            # Update the progress bar.
            training_loop.set_postfix(curr_train_loss="{:.8f}".format(training_loss / (train_iteration + 1)),
                                      val_loss="{:.8f}".format(validation_loss))

            # Update the tensorboard logger.
            tb_logger.add_scalar(f'classifier_{name}/train_loss', loss.item(),
                                 epoch * len(train_loader) + train_iteration)

        # Validation stage, where we don't want to update the parameters. Pay attention to the classifier.eval() line
        # and "with torch.no_grad()" wrapper.
        model_im.eval()
        model_lid.eval()
        val_loop = create_tqdm_bar(val_loader, desc=f'Validation Epoch [{epoch + 1}/{epochs}]')

        with torch.no_grad():
            for val_iteration, val_batch in val_loop:

                left_img_batch = val_batch['left_img'].to(device)  # batch of left image, id 02
                velo_points = val_batch['velo']
                path = val_batch['cam_path'][0] # TODO: Still assuming that all calibration file same!!!
                cam2cam, velo2cam = get_calibration_files(calib_dir=path)

                size = left_img_batch[0].size()   # ASSUME: same image size for all batch

                depth_batch, depth_neg = lidar_batch_transform(cam2cam=cam2cam, velo2cam=velo2cam,lidar_batch=velo_points, im_shape=[size[2],size[1]]) # height width
                # Image to torch tensor
                # image lidar transforms

                depth_batch, depth_neg = depth_batch.to(device), depth_neg.to(device)

                batch_length = len(depth_batch)
                half_length = batch_length // 2

                # Create shuffled label tensor directly on the specified device
                label_tensor = torch.cat(
                    [torch.zeros(half_length, device=device), torch.ones(half_length, device=device)])

                # Shuffle the tensor on the GPU
                label_val = label_tensor[torch.randperm(label_tensor.size(0))]

                # Stack depth batches according to labels
                stacked_depth_batch = torch.where(label_val.unsqueeze(1).unsqueeze(2).unsqueeze(3).bool(), depth_batch,
                                                  depth_neg)

                pred_im = model_im.forward(left_img_batch)
                pred_lid = model_lid.forward(stacked_depth_batch)

                loss = loss_func(pred_im, pred_lid, label_val)
                validation_loss += loss.item()

                # Update the progress bar.
                val_loop.set_postfix(val_loss="{:.8f}".format(validation_loss / (val_iteration + 1)))

                # Update the tensorboard logger.
                tb_logger.add_scalar(f'classifier_{name}/val_loss', loss.item(),
                                     epoch * len(val_loader) + val_iteration)

        # This value is used for the progress bar of the training loop.
        validation_loss /= len(val_loader)

    save_model(model_im, file_name=save_model_im)
    save_model(model_lid, file_name=save_model_lid)