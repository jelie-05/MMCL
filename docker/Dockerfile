FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

RUN pip install -r requirements.txt

# Run main.py when the container launches
CMD ["python", "inference/train/main.py"]
