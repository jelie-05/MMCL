# Multimodal Self-Supervised Learning for LiDAR-Camera Calibration

This project explores **multimodal self-supervised learning (SSL)** techniques for sensor fusion, specifically focusing on the calibration of **LiDAR** and **camera** data using **contrastive learning** and **Centered Kernel Alignment (CKA)** analysis. The work aims to detect and correct sensor miscalibration through a combination of image and depth information, leveraging **ResNet18-small** as the backbone model.

## Project Overview

The project investigates how different calibration methods affect the alignment of feature representations between **LiDAR** and **image encoders**. Using **contrastive learning**, we develop a system that learns robust feature representations from both sensor modalities. We also employ **CKA analysis** to evaluate the similarity of the learned features across various layers of the encoders, providing insights into the effectiveness of different calibration methods.

External sources:
Kitti downloader is from https://github.com/Deepak3994/Kitti-Dataset.git
Kitti loader adapted from https://github.com/joseph-zhang/KITTI-TorchLoader
