import os.path
import random

# Read the lines from the input file
current_path = os.path.abspath(os.getcwd())
# filename = 'eigen_train_files.txt'
filename = 'eigen_val_files.txt'
full_filename = os.path.join(current_path, 'KITTIRaw', 'Dataloader', 'filenames', filename)

with open(full_filename, 'r') as f:
    lines = f.readlines()

# Shuffle the lines
random.shuffle(lines)

# Write the shuffled lines to the output file
with open(full_filename, 'w') as f:
    f.writelines(lines)