import os
import re
from collections import defaultdict

# Construct the file paths
current_file_path = os.path.abspath(__file__)
root = os.path.abspath(os.path.join(current_file_path, '../../../../../../..'))

# Path to your .txt file
input_file_path = os.path.join(root, 'src/dataset/kitti_loader/Dataloader/filenames_complete/eigen_val_files.txt')
output_file_path = os.path.join(root, 'src/dataset/kitti_loader/Dataloader/filenames_complete/val_file_numbers.txt')


# Function to extract and count data
def extract_and_count_data(file_path):
    drive_counts = defaultdict(int)
    date_counts = defaultdict(int)

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            if len(parts) >= 4:
                # Extracting the drive number using regex
                drive_match = re.search(r'drive_(\d+)_sync', parts[0])
                # Extracting the date
                date_match = re.search(r'(\d{4}_\d{2}_\d{2})', parts[0])

                if drive_match and date_match:
                    drive_number = drive_match.group(1)
                    date = date_match.group(1)
                    mm_dd = '_'.join(date.split('_')[1:3])

                    # Incrementing counts
                    drive_counts[drive_number] += 1
                    date_counts[mm_dd] += 1

    return drive_counts, date_counts


# Extract and count the data
drive_counts, date_counts = extract_and_count_data(input_file_path)

# Save the counts to a new file
with open(output_file_path, 'w') as output_file:
    # output_file.write("Drive Number Counts:\n")
    for drive_number, count in drive_counts.items():
        output_file.write(f"{drive_number} {count}\n")

    output_file.write("\nDate Counts (mm_dd):\n")
    for mm_dd, count in date_counts.items():
        output_file.write(f"Date {mm_dd}: {count}\n")

print(f"Data counts successfully extracted and saved to {output_file_path}")
