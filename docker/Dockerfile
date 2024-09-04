# Use an official CUDA base image from the NVIDIA repository
FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

# Install Python and other necessary dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    python3-setuptools \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the default Python version to 3.10
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install Python packages
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8000
EXPOSE 8000

# Run main.py when the container launches

# 1: done in Edna; 03/09 18:30
# CMD ["sh", "-c", "python ./src/main.py --save_name resnet_small_240902 --config configs_resnet18_small_240902_augm.yaml --classifier"]

# 2: compare to 1.
CMD ["sh", "-c", "python ./src/main.py --save_name resnet_small_noaug_240903 --config configs_resnet18_small_240903_noaug.yaml --classifier"]

# 3: check init
# Try other initialization for small

# 4: train larger model -> compare 1 & 2: which better
# CMD ["sh", "-c", "python ./src/main.py --save_name resnet_all_240904 --config configs_resnet18_all_.yaml --classifier"]

# 5: Train ViT

# CMD ["sh", "-c", "python ./inference/train/main.py --name_im 240802_masking_1_image --name_lid 240802_masking_1_lid --name_cls 240802_masking_1_cls --masking --pixel_wise"]
# CUDA_VISIBLE_DEVICES=0 for 2 cudas

#&& python ./inference/eval/precision_recall/main.py --model_im image_240708_full_1 --model_lid lidar_240708_full_1 --name_cls cls_240708_full_1 
