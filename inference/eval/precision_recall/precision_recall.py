from sklearn.metrics import auc
import torch
import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.dataset.kitti_loader_2D.dataset_2D import DataGenerator
from sklearn.metrics import precision_recall_curve, auc


def pr_auc(label, prediction):
    # Convert tensors to numpy arrays
    label = np.concatenate([tensor.cpu().numpy() for tensor in label])
    prediction = np.concatenate([tensor.cpu().numpy() for tensor in prediction])

    # Calculate precision-recall pairs for different thresholds
    precision, recall, thresholds = precision_recall_curve(label, prediction)

    # Calculate PR AUC
    pr_auc = auc(recall, precision)

    return {
        "Precision": precision,
        "Recall": recall,
        "thresholds": thresholds,
        "pr_auc": pr_auc
    }


def evaluation(device, data_root, model_cls):
    model_cls.to(device)
    model_cls.eval()

    eval_gen = DataGenerator(data_root, 'test')
    eval_dataloader = eval_gen.create_data(64)

    label = []
    prediction = []

    with torch.no_grad():
        for batch in eval_dataloader:
            left_img_batch = batch['left_img'].to(device)  # batch of left image, id 02
            depth_batch = batch['depth'].to(device)  # the corresponding depth ground truth of given id
            depth_neg = batch['depth_neg'].to(device)

            batch_length = len(depth_batch)
            half_length = batch_length // 2

            # Create shuffled label tensor directly on the specified device
            label_tensor = torch.cat(
                [torch.zeros(half_length, device=device), torch.ones(half_length, device=device)])

            # Shuffle the tensor on the GPU
            label_val = label_tensor[torch.randperm(label_tensor.size(0))].unsqueeze(1)

            # Stack depth batches according to labels
            stacked_depth_batch = torch.where(label_val.unsqueeze(2).unsqueeze(3).bool(), depth_batch,
                                              depth_neg)

            N, C, H, W = left_img_batch.size()
            pred_cls = model_cls.forward(image=left_img_batch, lidar=stacked_depth_batch, H=H, W=W)

            label = label.append(label_val)
            prediction = prediction.append(pred_cls)

    PR = pr_auc(label=label, prediction=prediction)

    return PR