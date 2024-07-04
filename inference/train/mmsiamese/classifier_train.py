import torch
from tqdm import tqdm
from src.models.classifier_head import classifier_head
from src.dataset.kitti_loader_2D.dataset_2D import DataGenerator
import torch.nn as nn

from src.utils.save_load_model import save_model


def create_tqdm_bar(iterable, desc):
    return tqdm(enumerate(iterable), total=len(iterable), ncols=150, desc=desc)

def main(params, data_root, tb_logger, pretrained_im, pretrained_lid, name_cls, name="default"):
    """
    Train the classifier for a number of epochs.
    """

    """ Data Loader """
    train_gen = DataGenerator(data_root, 'train')
    train_loader = train_gen.create_data(int(params.get('batch_size')), shuffle=True)
    val_gen = DataGenerator(data_root, 'val')
    val_loader = val_gen.create_data(int(params.get('batch_size')), shuffle=False)

    """ Loss Function """
    loss_func = nn.BCELoss()

    """ Device """
    device = torch.device(params.get('device'))

    """ Other hyperparams """
    learning_rate = float(params.get('lr'))
    epochs = int(params.get('epoch'))

    model_im = pretrained_im.to(device)
    model_lid = pretrained_lid.to(device)
    model_lid.eval()
    model_im.eval()
    model_cls = classifier_head(model_lid=model_lid, model_im=model_im).to(device)

    optimizer = torch.optim.Adam(model_cls.parameters(), learning_rate)

    for epoch in range(epochs):

        training_loss = 0
        validation_loss = 0

        # Training stage, where we want to update the parameters.
        model_cls.train()  # Set the model to training mode

        # Create a progress bar for the training loop.
        training_loop = create_tqdm_bar(train_loader, desc=f'Training Epoch [{epoch + 1}/{epochs}]')
        for train_iteration, batch in training_loop:
            optimizer.zero_grad()  # Reset the gradients - VERY important! Otherwise they accumulate.

            # Not yet as torch tensor
            left_img_batch = batch['left_img'].to(device)  # batch of left image, id 02
            depth_batch = batch['depth'].to(device)  # the corresponding depth ground truth of given id
            depth_neg = batch['depth_neg'].to(device)

            batch_length = len(depth_batch)
            half_length = batch_length // 2

            # Create shuffled label tensor directly on the specified device
            label_tensor = torch.cat([torch.zeros(half_length, device=device), torch.ones(half_length, device=device)])

            # Shuffle the tensor on the GPU
            label_list = label_tensor[torch.randperm(label_tensor.size(0))]

            # Stack depth batches according to labels
            stacked_depth_batch = torch.where(label_list.unsqueeze(1).unsqueeze(2).unsqueeze(3).bool(), depth_batch,
                                              depth_neg)

            pred_cls = model_cls.forward(image=left_img_batch, lidar=stacked_depth_batch)
            pred_cls = pred_cls.squeeze(dim=1)

            loss = loss_func(pred_cls, label_list)
            loss.backward()  # Stage 2: Backward().
            optimizer.step()  # Stage 3: Update the parameters.

            training_loss += loss.item()

            # Update the progress bar.
            training_loop.set_postfix(curr_train_loss="{:.8f}".format(training_loss / (train_iteration + 1)),
                                      val_loss="{:.8f}".format(validation_loss))

            # Update the tensorboard logger.
            tb_logger.add_scalar(f'siamese_{name}/train_loss', loss.item(),
                                 epoch * len(train_loader) + train_iteration)

        # Validation stage, where we don't want to update the parameters. Pay attention to the classifier.eval() line
        # and "with torch.no_grad()" wrapper.
        model_cls.eval()
        val_loop = create_tqdm_bar(val_loader, desc=f'Validation Epoch [{epoch + 1}/{epochs}]')

        with torch.no_grad():
            for val_iteration, val_batch in val_loop:

                left_img_batch = val_batch['left_img'].to(device)  # batch of left image, id 02
                depth_batch = val_batch['depth'].to(device)  # the corresponding depth ground truth of given id
                depth_neg = val_batch['depth_neg'].to(device)

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

                pred_cls = model_cls.forward(image=left_img_batch, lidar=stacked_depth_batch).squeeze(dim=1)

                loss = loss_func(pred_cls, label_val)
                validation_loss += loss.item()

                # Update the progress bar.
                val_loop.set_postfix(val_loss="{:.8f}".format(validation_loss / (val_iteration + 1)))

                # Update the tensorboard logger.
                tb_logger.add_scalar(f'siamese_{name}/val_loss', loss.item(),
                                     epoch * len(val_loader) + val_iteration)

        # This value is used for the progress bar of the training loop.
        validation_loss /= len(val_loader)

    save_model(model_cls, file_name=name_cls)
