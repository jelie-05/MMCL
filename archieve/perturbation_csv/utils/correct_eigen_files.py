import os

current_file_path = os.path.abspath(__file__)
root = os.path.abspath(os.path.join(current_file_path, '../../../../../../..'))

input_path = os.path.join(root,'src/dataset/kitti_loader/Dataloader/perturbation_csv/eigen_val_files_old.txt')
output_file_path = os.path.join(root,'src/dataset/kitti_loader/Dataloader/perturbation_csv/eigen_val_files.txt')

# Read the input file
with open(input_path, 'r') as file:
    lines = file.readlines()

# Process each line
updated_lines = []
for line in lines:
    parts = line.split()
    # Extract the necessary parts
    drive_id = parts[0].split('/')[1]
    file_id = parts[0].split('/')[-1].split('.')[0]
    # Construct the string to append
    new_part = f"{drive_id}_{file_id}"
    # Append the new part to the line
    updated_line = f"{line.strip()} {new_part}\n"
    updated_lines.append(updated_line)

# Write the updated lines to a new file
with open(output_file_path, 'w') as file:
    file.writelines(updated_lines)

print(f"Processing complete. Check {output_file_path} for results.")
