from sklearn.metrics import auc
import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.dataset.kitti_loader.dataset_2D import DataGenerator
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

def confusion_matrix(label, prediction, threshold=0.5, name_list=None):
    # Convert tensors to numpy arrays
    label = np.concatenate([tensor.cpu().numpy() for tensor in label])
    prediction = np.concatenate([tensor.cpu().numpy() for tensor in prediction])
    
    # Binarize predictions based on the threshold
    binarized_prediction = (prediction >= threshold).astype(int)

    # Calculate confusion matrix components
    TP = np.sum((label == 1) & (binarized_prediction == 1))
    TN = np.sum((label == 0) & (binarized_prediction == 0))
    FP = np.sum((label == 0) & (binarized_prediction == 1))
    FN = np.sum((label == 1) & (binarized_prediction == 0))
    
    # Extract FP and FN names if name_list is provided
    if name_list is not None:
        # Get indices of FP and FN
        fp_indices = np.where((label == 0) & (binarized_prediction == 1))[0]
        fn_indices = np.where((label == 1) & (binarized_prediction == 0))[0]
        
        # Extract names of FP and FN
        fp_names = [name_list[i] for i in fp_indices]
        fn_names = [name_list[i] for i in fn_indices]
        
        return TP, TN, FP, FN, fp_names, fn_names
    
    return TP, TN, FP, FN

def evaluation(device, data_root, model_cls, perturb_file):
    model_cls.to(device)
    model_cls.eval()

    eval_gen = DataGenerator(data_root, 'test', perturb_filenames=perturb_file)
    eval_dataloader = eval_gen.create_data(64)
    
    num_run = '6'
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    fp_output_file = os.path.join(output_dir, f'output_{num_run}_fp.txt')
    fn_output_file = os.path.join(output_dir, f'output_{num_run}_fn.txt')
    results_file = os.path.join(output_dir, f'output_{num_run}_pr_auc.txt')
    plot_file = os.path.join(output_dir, f'output_{num_run}_plot.png')

    label = torch.empty(0, 1, device=device)
    prediction = torch.empty(0, 1, device=device)
    fp_list = []
    fn_list = []
    sum_TP = 0
    sum_TN = 0
    sum_FP = 0
    sum_FN = 0
    iter = 0

    with torch.no_grad():
        print("Evaluation is started")
        for batch in eval_dataloader:
            left_img_batch = batch['left_img'].to(device)  # batch of left image, id 02
            depth_batch = batch['depth'].to(device)  # the corresponding depth ground truth of given id
            depth_neg = batch['depth_neg'].to(device)
            depth_name = batch['name']

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

            TP, TN, FP, FN, fp_names, fn_names = confusion_matrix(label=label_val, prediction=pred_cls, name_list=depth_name)
            fp_list.extend(fp_names)  # Use extend instead of append to add elements directly to the list
            fn_list.extend(fn_names)  # Use extend instead of append to add elements directly to the list
            sum_TP += TP
            sum_TN += TN
            sum_FP += FP
            sum_FN += FN

            iter += 1
            print(f'Iteration {iter} finished')

    print(f'True Positives: {sum_TP}')
    print(f'True Negatives: {sum_TN}')
    print(f'False Positives: {sum_FP}')
    print(f'False Negatives: {sum_FN}')

    # Save FP list to a text file
    with open(fp_output_file, 'w') as f:
        f.write("False Positives (FP):\n")
        for item in fp_list:
            f.write("%s\n" % item)
    
    # Save FN list to a text file
    with open(fn_output_file, 'w') as f:
        f.write("False Negatives (FN):\n")
        for item in fn_list:
            f.write("%s\n" % item)
    
    # np.set_printoptions(threshold=np.inf) 
    results = pr_auc(label=label, prediction=prediction)
 
    with open(results_file, 'w') as f:
        f.write("TP(0.5): %i\n" % sum_TP)
        f.write("TN(0.5): %i\n" % sum_TN)
        f.write("FP(0.5): %i\n" % sum_FP)
        f.write("FN(0.5): %i\n" % sum_FN)
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
