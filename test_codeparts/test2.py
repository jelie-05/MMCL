import csv
import os

# Define file paths
current_file_path = os.path.abspath(__file__)
root = os.path.abspath(os.path.join(current_file_path, '../..'))

csv_file_path = os.path.join(root,'src/dataset/kitti_loader/Dataloader/perturbation_csv/perturbation_neg.csv')

# Read the CSV file
def find_row_by_name(filename, target_name):
    with open(filename, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            if row['name'] == target_name:
                return row
    return None

target_name = "2011_09_26_drive_0101_sync_0000000667"

row = find_row_by_name(csv_file_path, target_name)
print(row)
x = row['x']
print(type(x))