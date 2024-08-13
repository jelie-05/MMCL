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
CMD ["sh", "-c", "python ./src/main.py --save_name 240813 --augmentation"]
# CMD ["sh", "-c", "python ./src/main.py --save_name 240813"]
# CMD ["sh", "-c", "python ./inference/train/main.py --name_im 240802_masking_1_image --name_lid 240802_masking_1_lid --name_cls 240802_masking_1_cls --masking --pixel_wise"]

# CUDA_VISIBLE_DEVICES=0 for 2 cudas

#&& python ./inference/eval/precision_recall/main.py --model_im image_240708_full_1 --model_lid lidar_240708_full_1 --name_cls cls_240708_full_1 
