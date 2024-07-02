FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000

# Run main.py when the container launches
CMD ["python", "./inference/train/main.py", "--model_im", "image_backbone_2D_test2", "--model_lid", "lidar_backbone_2D_test2"]
