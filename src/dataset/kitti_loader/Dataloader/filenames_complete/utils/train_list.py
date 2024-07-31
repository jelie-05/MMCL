import os

current_file_path = os.path.abspath(__file__)
root = os.path.abspath(os.path.join(current_file_path, '../../../../../..'))

# Specify the path to the folder list file
sync_test_path = os.path.join(root,'src/dataset/kitti_loader/Dataloader/filenames_complete/list_sync_test.txt')

sync_list_path = os.path.join(root,'src/dataset/kitti_loader/Dataloader/filenames_complete/list_sync_all.txt')

output_path = os.path.join(root,'src/dataset/kitti_loader/Dataloader/filenames_complete/list_sync_train_eval.txt')

# Read patterns from remove.txt
with open(sync_test_path, 'r') as remove_file:
    patterns = [line.strip() for line in remove_file]

# Read lines from list.txt
with open(sync_list_path, 'r') as list_file:
    lines = list_file.readlines()

# Write to adjusted_file.txt excluding lines that match any pattern
with open(output_path, 'w') as adjusted_file:
    for line in lines:
        if line.strip() not in patterns:
            adjusted_file.write(line)

print(f"File names have been written to {output_path}")