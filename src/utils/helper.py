import torch
import src.models.resnet as resnet
import src.models.vision_transformer as vit
import torch.optim as optim
from src.utils.tensors import trunc_normal_
from src.models.classifier_head import classifier_head as classifier

def load_checkpoint(
    r_path,
    encoder_im,
    encoder_lid,
    opt_im,
    opt_lid
):
    checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
    epoch = checkpoint['epoch']

    pretrained_im = checkpoint['encoder_im']
    msg = encoder_im.load_state_dict(pretrained_im)
    print(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

    pretrained_lid = checkpoint['encoder_lid']
    msg = encoder_lid.load_state_dict(pretrained_lid)
    print(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

    opt_im.load_state_dict(checkpoint['optimizer_im'])
    opt_lid.load_state_dict(checkpoint['optimizer_lid'])

    print(f'read-path: {r_path}')
    del checkpoint

    # except Exception as e:
    #     print(f'Encountered exception when loading checkpoint {e}')
    #     epoch = 0

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


def gen_mixed_data(depth_batch, depth_batch_neg, device, masking):
    # Assign label randomly to each component of the batch (50/50)
    batch_length = len(depth_batch)
    half_length = batch_length // 2
    label_tensor = torch.cat(
        [torch.zeros(half_length, device=device), torch.ones(half_length, device=device)])
    label_list = label_tensor[torch.randperm(label_tensor.size(0))]

    # Stack depth batches according to labels (depth_batch or depth_neg)
    stacked_depth_batch = torch.where(label_list.unsqueeze(1).unsqueeze(2).unsqueeze(3).bool(), depth_batch,
                                      depth_batch_neg)
    # Calculate Mask
    if masking:
        mask = (depth_batch != 0.0).int()
        mask_neg = (depth_batch_neg != 0.0).int()
        stacked_mask = torch.where(label_list.unsqueeze(1).unsqueeze(2).unsqueeze(3).bool(), mask,
                                   mask_neg)
    else:
        stacked_mask = None

    return stacked_depth_batch, label_list, stacked_mask


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

