import os

# Specify the path to the folder list file
sync_list_path = './sync_list.txt'

# Specify the output file path
output_file_path = './complete_filenames.txt'

current_file_path = os.path.abspath(__file__)
root = os.path.abspath(os.path.join(current_file_path, '../../../../../..'))
print(root)

# Read the folder paths from the folder list file
with open(sync_list_path, 'r') as file:
    syncs = [line.strip() for line in file.readlines()]

# Open the output file in write mode
with open(output_file_path, 'w') as output_file:
    for sync in syncs:

        date_folder = sync.split('_drive_')[0]
        sync_name = sync + '_sync'
        image02_path = os.path.join(root, r"data\kitti", date_folder, sync_name, r'image_02\data')

        # Check if the folder path is valid
        if os.path.isdir(image02_path):
            # Get the list of files in the current folder
            images_02 = os.listdir(image02_path)

            for image in images_02:
                # Get file number
                number_parts = image.split('.')[0]

                # Formatting file name
                image02_name = os.path.join(date_folder, sync_name, r'image_02\data', f'{number_parts}.png')
                image03_name = os.path.join(date_folder, sync_name, r'image_03\data', f'{number_parts}.png')
                lidar_name = os.path.join(date_folder, sync_name, r'velodyne_points\data', f'{number_parts}.bin')

                # Write the folder name (optional, for clarity)
                output_file.write(f"{image02_name} {image03_name} {date_folder} {lidar_name}\n")

        else:
            output_file.write(f"Folder: {image02_path} (Invalid path)\n\n")

print(f"File names have been written to {output_file_path}")
