import os

current_file_path = os.path.abspath(__file__)
root = os.path.abspath(os.path.join(current_file_path, '../../../../../../..'))

# Specify the path to the folder list file
sync_list_path = os.path.join(root,'src/dataset/kitti_loader/Dataloader/filenames_complete/utils/list_sync_all.txt')
output_file_path = os.path.join(root,'src/dataset/kitti_loader/Dataloader/filenames_complete/filenames_all.txt')

# sync_list_path = os.path.join(root,'src/dataset/kitti_loader/Dataloader/filenames_complete/utils/list_sync_test.txt')
# output_file_path = os.path.join(root,'src/dataset/kitti_loader/Dataloader/filenames_complete/eigen_test_files.txt')

# sync_list_path = os.path.join(root,'src/dataset/kitti_loader/Dataloader/filenames_complete/utils/list_sync_train_mini.txt')
# output_file_path = os.path.join(root,'src/dataset/kitti_loader/Dataloader/filenames_complete/eigen_train_files.txt')

# Read the folder paths from the folder list file
with open(sync_list_path, 'r') as file:
    syncs = [line.strip() for line in file.readlines()]

# Open the output file in write mode
with open(output_file_path, 'w') as output_file:
    for sync in syncs:

        date_folder = sync.split('_drive_')[0]
        sync_name = sync + '_sync'
        image02_path = os.path.join(root, "data/kitti", date_folder, sync_name, 'image_02/data')
        velo_path = os.path.join(root, "data/kitti", date_folder, sync_name, 'velodyne_points/data')

        # Check if the folder path is valid
        if os.path.isdir(image02_path):
            # Get the list of files in the current folder
            images_02 = os.listdir(image02_path)

            for image in images_02:
                # Get file number
                number_parts = image.split('.')[0]

                # Formatting file name
                image02_name = os.path.join(date_folder, sync_name, r'image_02/data', f'{number_parts}.png')
                image03_name = os.path.join(date_folder, sync_name, r'image_03/data', f'{number_parts}.png')
                lidar_name = os.path.join(date_folder, sync_name, r'velodyne_points/data', f'{number_parts}.bin')

                # Write the folder name (optional, for clarity)
                output_file.write(f"{image02_name} {image03_name} {date_folder} {lidar_name}\n")

        else:
            output_file.write(f"Folder: {image02_path} (Invalid path)\n\n")

print(f"File names have been written to {output_file_path}")
