from sklearn.metrics import auc

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