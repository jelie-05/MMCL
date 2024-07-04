from sklearn.metrics import auc
import torch
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.dataset.kitti_loader_2D.dataset_2D import DataGenerator


def PR_AuC(TP,TN,FP,FN):
    # Calculate Precision
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    # Calculate Recall
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    pr_auc = auc(recall, precision)

    return {
        "Precision": precision,
        "Recall": recall,
        "P-R AUC": pr_auc
    }


def evaluation(device, data_root, model_cls):

    device = torch.device(device)

    model_cls.to(device)
    model_cls.eval()

    eval_gen = DataGenerator(data_root, 'test')
    # eval_gen = DataGenerator(data_root,'val')
    eval_dataloader = eval_gen.create_data(64)

    # Initialize counters for confusion matrix components
    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0
    iteration = 0

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

            pred_cls = model_cls.forward(image=left_img_batch, lidar=stacked_depth_batch)

            # pred_cls = torch.sigmoid(pred_cls)
            classified_pred = (pred_cls >= 0.5).int()

            # Calculate confusion matrix components on GPU: 1 = positive, 0 = negative
            tp = torch.sum((label_val == 1) & (classified_pred == 1)).item()
            tn = torch.sum((label_val == 0) & (classified_pred == 0)).item()
            fp = torch.sum((label_val == 0) & (classified_pred == 1)).item()
            fn = torch.sum((label_val == 1) & (classified_pred == 0)).item()

            # Accumulate the results
            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn
            print("batch {} done".format(iteration))
            iteration += 1

    # Display accumulated results
    print(f'Total True Positives (TP): {total_tp}')
    print(f'Total True Negatives (TN): {total_tn}')
    print(f'Total False Positives (FP): {total_fp}')
    print(f'Total False Negatives (FN): {total_fn}')

    PR = PR_AuC(total_tp, total_tn, total_fp, total_fn)

    return PR