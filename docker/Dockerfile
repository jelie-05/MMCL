# Use an official CUDA base image from the NVIDIA repository
FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

# Install Python and other necessary dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    python3-setuptools \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the default Python version to 3.9
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

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
CMD ["python", "./inference/train/main.py", "--model_im", "image_backbone_2D_test2", "--model_lid", "lidar_backbone_2D_test2"]
