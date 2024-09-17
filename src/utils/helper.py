import torch
import src.models.resnet as resnet
import src.models.vision_transformer as vit
import torch.optim as optim
from src.utils.tensors import trunc_normal_
import os
from src.models.classifier_head import classifier_head as classifier

def load_checkpoint(
    r_path,
    encoder_im,
    encoder_lid,
    opt_im,
    opt_lid
):
    # Check if the checkpoint file exists
    if not os.path.exists(r_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {r_path}")

    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']

        pretrained_im = checkpoint['encoder_im']
        msg_im = encoder_im.load_state_dict(pretrained_im)
        print(f'Loaded pretrained image encoder from epoch {epoch} with msg: {msg_im}')

        pretrained_lid = checkpoint['encoder_lid']
        msg_lid = encoder_lid.load_state_dict(pretrained_lid)
        print(f'Loaded pretrained lidar encoder from epoch {epoch} with msg: {msg_lid}')

        opt_im.load_state_dict(checkpoint['optimizer_im'])
        opt_lid.load_state_dict(checkpoint['optimizer_lid'])

        print(f'Read path: {r_path}')
        del checkpoint

    except Exception as e:
        print(f"Encountered exception when loading checkpoint: {e}")
        epoch = 0

    return encoder_im, encoder_lid, opt_im, opt_lid, epoch


def load_checkpoint_cls(
    r_path,
    classifier
):
    checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
    epoch = checkpoint['epoch']

    pretrained_dict = checkpoint['classifier']
    msg = classifier.load_state_dict(pretrained_dict)
    print(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

    # opt.load_state_dict(checkpoint['opt'])

    print(f'read-path: {r_path}')
    del checkpoint

    # except Exception as e:
    #     print(f'Encountered exception when loading checkpoint {e}')
    #     epoch = 0

    return classifier, epoch


def gen_mixed_data(img_batch, depth_batch, depth_batch_neg, device, masking):
    stacked_mask = None
    half_length = len(depth_batch)
    stacked_img = torch.cat([img_batch, img_batch])

    label_tensor = torch.cat([torch.ones(half_length, device=device), torch.zeros(half_length, device=device)])

    stacked_depth_batch = torch.cat([depth_batch, depth_batch_neg])

    indices = torch.randperm(stacked_depth_batch.size(0))

    stacked_depth_batch = stacked_depth_batch[indices]
    label_list = label_tensor[indices]
    stacked_img = stacked_img[indices]

    # label_list = label_tensor[torch.randperm(label_tensor.size(0))]
    # Stack depth batches according to labels (depth_batch or depth_neg)
    # stacked_depth_batch = torch.where(label_list.unsqueeze(1).unsqueeze(2).unsqueeze(3).bool(), depth_batch,
    #                                   depth_batch_neg)

    return stacked_depth_batch, stacked_img, label_list, stacked_mask


def init_model(
        device,
        mode='resnet',
        model_name='resnet18_small',
        patch_size=16,
):
    if mode == 'resnet':
        model_name_im = model_name+'_im'
        model_name_lid = model_name + '_lid'
        encoder_im = resnet.__dict__[model_name_im]().to(device)
        encoder_lid = resnet.__dict__[model_name_lid]().to(device)
    else:
        model_name_im = model_name+'_im'
        model_name_lid = model_name + '_lid'
        encoder_im = vit.__dict__[model_name_im](patch_size=patch_size).to(device)
        encoder_lid = vit.__dict__[model_name_lid](patch_size=patch_size).to(device)

        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.LayerNorm):
                torch.nn.init.constant_(m.bias, 0)
                torch.nn.init.constant_(m.weight, 1.0)

        for m in encoder_im.modules():
            init_weights(m)

        for m in encoder_lid.modules():
            init_weights(m)

    return encoder_im, encoder_lid

def init_opt(
        model,
        args
):
    opt_name = args['optimizer']
    learning_rate = float(args['lr'])
    weight_decay = float(args['weight_decay'])
    scheduler_step = args['scheduler_step']
    scheduler_gamma = args['scheduler_gamma']

    if opt_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        print("adam initialized")
    elif opt_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        print("adamw initialized")
    else:
        raise NotImplementedError(f"Optimizer '{opt_name}' not implemented yet")

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    return optimizer, scheduler

