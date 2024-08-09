import torch

def save_ckpt(model, file_name, epoch):
    return

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