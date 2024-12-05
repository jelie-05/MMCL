# Multimodal Contrastive Learning for LiDAR-Camera Calibration

This project explores **multimodal self-supervised learning (SSL)** techniques for monitoring the calibration of **LiDAR** and **camera** data using **contrastive learning**. The work aims to detect and correct sensor miscalibration in LiDAR projection into image plane.

## Project Overview

The project investigates how different calibration methods affect the alignment of feature representations between **LiDAR** and **image encoders**. Using **contrastive learning**, we develop a system that learns robust feature representations from both sensor modalities. We also employ **CKA analysis** to evaluate the similarity of the learned features across various layers of the encoders and **Evaluation Metrics** to quantify the detection results.

### Key Contributions
- **Contrastive Learning**: Developed a contrastive loss function that enhances the calibration process by aligning multimodal features.
- **Fault Detection**: Using the representations from encoders trained with contrastive learning, we trained the classifier head to detect errors in the input.
- **Robust Calibration Evaluation**: A thorough evaluation of the models using different error modes providing insights into the learned representations and the generalization across different calibration matrices.

## Table of Contents
- [Usage](#usage)
- [Dataset](#dataset)
- [Results](#results)
- [Sources](#sources)

## Usage
### Training
```bash
python ./src/main.py --save_name (save_name) --classifier
```
`--save_name` refers to the name the outputs will be saved (e.g., resnet18_small_aug_240924) and also the configs file `configs_(save_name).yaml`. `--classifier` activates the classifier's training.

#### Configuration File Options

The following options are available for configuring the model via the `config.yaml` file:

| Parameter    | Options                        | Description                                     |
|--------------|--------------------------------|-------------------------------------------------|
| `backbone`   | `resnet`, `vit`                | Specifies the model backbone architecture.      |
| `model_name` | `resnet_small`, `resnet_all`, `vit_small` | Model variations for the selected backbone.      |
| `optimizer`  | `adam`, `adamw`                | Optimizer used for training the model.           |

Furthermore, pre-trained encoders can be used by enabling `pretrained_encoder` and inserting the `save_name` in `pretrained_name`. 
The encoders can be further retrained by enabling `retrain` and giving the starting epoch.

### Evaluation 
```bash
python ./inference/eval/main.py --save_name (save_name) --perturbation (CSV file) --eval_metrics --cka
```
`--perturbation` refers to CSV file, where the miscalibration errors are saved, e.g., `neg_master`. Two evaluation methods are implemented, namely evaluation metrics and CKA analysis. They can be enabled by giving `--eval_metrics` and `--cka` respectively.

## Dataset
This paper used the KITTI Raw dataset and the derived miscalibration dataset. The dataset should be stored in folder 'data', with the miscalibration dataset saved as `perturbation_(type).csv`.

## Results
Our model shows better performance compared to the state of the art model. More detail result will follow after peer-review of submitted paper.

| **Metrics**     | **Methods** | **All Errors** | **Rot Hard** | **Trans Easy** |
|------------------|-------------|----------------|--------------|----------------|
| **Accuracy**     | Wei         | 95.13%         | 86.28%       | 92.05%         |
|                  | Ours        | **99.08%**     | **99.00%**   | **99.97%**     |
| **Precision**    | Wei         | 92.02%         | 78.24%       | 86.59%         |
|                  | Ours        | **100.00%**    | **100.00%**  | **100.00%**    |
| **Recall**       | Wei         | **99.05%**     | **99.96%**   | 99.65%         |
|                  | Ours        | 98.17%         | 97.99%       | **99.93%**     |

## Sources
External sources:

Kitti downloader is from https://github.com/Deepak3994/Kitti-Dataset.git

Kitti loader adapted from https://github.com/joseph-zhang/KITTI-TorchLoader
