from sklearn.metrics import auc
import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.datasets.kitti_loader.dataset_2D import DataGenerator
from sklearn.metrics import precision_recall_curve


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

def plot_pr_curve(pr_data, plot_file):
    plt.figure(figsize=(8, 6))
    plt.plot(pr_data['Recall'], pr_data['Precision'], marker='.', label=f'PR AUC = {pr_data["pr_auc"]:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_file)
    plt.show()

def evaluation(device, data_root, model_cls, perturb_file):
    model_cls.to(device)
    model_cls.eval()

    eval_gen = DataGenerator(data_root, 'test', perturb_filenames=perturb_file)
    eval_dataloader = eval_gen.create_data(64)
    
    num_run = '6'
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f'output_{num_run}_pr_auc.txt')
    plot_file = os.path.join(output_dir, f'output_{num_run}_plot.png')

    label = torch.empty(0, 1, device=device)
    prediction = torch.empty(0, 1, device=device)

    with torch.no_grad():
        print("Evaluation is started")
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
            print(pred_cls)
            
            # Concatenate the batch results to the full tensors
            label = torch.cat((label, label_val), dim=0)
            prediction = torch.cat((prediction, pred_cls), dim=0)

    # np.set_printoptions(threshold=np.inf)
    results = pr_auc(label=label, prediction=prediction)
 
    with open(results_file, 'w') as f:
        f.write("Precision:\n")
        f.write("%s\n" % results["Precision"])
        f.write("Recall:\n")
        f.write("%s\n" % results["Recall"])
        f.write("Thresholds:\n")
        f.write("%s\n" % results["thresholds"])
        f.write("PR AUC:\n")
        f.write("%s\n" % results["pr_auc"])

    plot_pr_curve(results, plot_file=plot_file)

    return results
