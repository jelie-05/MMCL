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
CMD ["sh", "-c", "python ./inference/train/main.py --model_im 240722_full_1_image --model_lid 240722_full_1_lid --name_cls 240722_full_1_cls --pixel_wise"]

#&& python ./inference/eval/precision_recall/main.py --model_im image_240708_full_1 --model_lid lidar_240708_full_1 --name_cls cls_240708_full_1 
