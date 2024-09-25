# Multimodal Contrastive Learning for LiDAR-Camera Calibration

This project explores **multimodal self-supervised learning (SSL)** techniques for monitoring the calibration of **LiDAR** and **camera** data using **contrastive learning**. The work aims to detect and correct sensor miscalibration in LiDAR projection into image plane.

## Project Overview

The project investigates how different calibration methods affect the alignment of feature representations between **LiDAR** and **image encoders**. Using **contrastive learning**, we develop a system that learns robust feature representations from both sensor modalities. We also employ **CKA analysis** to evaluate the similarity of the learned features across various layers of the encoders and **Evaluation Metrics** to quantify the detection results.

### Key Contributions
- **Contrastive Learning**: Developed a contrastive loss function that enhances the calibration process by aligning multimodal features.
- **Fault Detection**: Using the representations from encoders trained with contrastive learning, we trained the classifier head to detect errors in the input.
- **Robust Calibration Evaluation**: A thorough evaluation of the models using different error modes providing insights into which calibration leads to better sensor fusion.

## Table of Contents
- [Usage](#usage)
- [Dataset](#dataset)
- [Results](#results)
- [Future Work](#future-work)
- [Contributors](#contributors)
- [License](#license)

## Usage
### Training
```bash
python ./src/main.py --save_name resnet18_small_aug_240924 --classifier
```
'save_name' refers to the name the outputs will be saved and also the configs file 'configs_(save_name).yaml'. '--classifier' activates the training of the classifier.

External sources:
Kitti downloader is from https://github.com/Deepak3994/Kitti-Dataset.git
Kitti loader adapted from https://github.com/joseph-zhang/KITTI-TorchLoader
