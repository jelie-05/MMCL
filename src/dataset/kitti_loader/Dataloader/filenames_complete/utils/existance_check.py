import os

# Path to your list.txt file
list_file_path = 'src/dataset/kitti_loader/Dataloader/filenames/eigen_val_files.txt'

# Path to the output file
output_file_path = 'src/dataset/kitti_loader/Dataloader/filenames_complete/utils/checked.txt'

# Base folder to prepend
base_folder = 'data/kitti'

# Read the list.txt file
with open(list_file_path, 'r') as file:
    lines = file.readlines()

# Open the output file in write mode
with open(output_file_path, 'w') as output_file:
    # Check the existence of each file
    for line in lines:
        files = line.strip().split()
        for file_path in files:
            full_path = os.path.join(base_folder, file_path)
            if not os.path.exists(full_path):
                output_file.write(f"Does not exist: {full_path}\n")
    output_file.write(f"Checking done")

print(f"Results have been saved to {output_file_path}")
