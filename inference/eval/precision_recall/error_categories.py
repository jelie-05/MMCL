# import csv
#
# def data_reader(csv_path, target_name):
#     with open(csv_path, mode='r') as file:
#         csv_reader = csv.DictReader(file)
#         for data in csv_reader:
#             if data['name'] == target_name:
#                 return data
#     return None
#
# def eval_categories(csv_path, filelist_path):
#     return

import pandas as pd
import os

# Load the CSV file
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
csv_file_path = os.path.join(root, 'data', 'kitti', 'perturbation_pos_augmentation.csv')
df = pd.read_csv(csv_file_path)

# Load the list of names
names_file_path = os.path.join(os.path.dirname(__file__), 'outputs', 'output_3_fn.txt')  # Replace with the path to your names file
with open(names_file_path, 'r') as file:
    names_list = file.read().splitlines()

# Filter the DataFrame to include only rows whose 'name' is in the names_list
filtered_df = df[df['name'].isin(names_list)]

# angle_0_2 = filtered_df[['theta_rad1', 'theta_rad2', 'theta_rad3']].apply(lambda col: ((col.abs() <= 2) & (col != 0)).sum())

# Apply the threshold condition to the theta_rad columns and the xyz columns
theta_columns = ['theta_rad1', 'theta_rad2', 'theta_rad3']
xyz_columns = ['x', 'y', 'z']

theta_threshold = 0.49
xyz_threshold = 0.09
# Create a boolean DataFrame for theta_rad columns greater than the threshold
filtered_theta_df = filtered_df[theta_columns] < theta_threshold

# Create a boolean DataFrame for xyz columns greater than the threshold
filtered_xyz_df = filtered_df[xyz_columns] < xyz_threshold

# Sum the boolean values across each row to count the number of theta_rad columns greater than the threshold
theta_non_zero_count = filtered_theta_df.sum(axis=1)

# Check if all xyz columns are greater than the threshold for each row
xyz_all_above_threshold = filtered_xyz_df.all(axis=1)

# Count the number of rows where exactly one of the theta_rad columns is greater than the threshold and all xyz columns are greater than the threshold
single_theta_non_zero_and_xyz = ((theta_non_zero_count == 1) & xyz_all_above_threshold).sum()

# Count the number of rows where any of the theta_rad columns is greater than the threshold and any of the xyz columns are greater than the threshold
any_theta_non_zero_and_xyz = ((theta_non_zero_count > 0) & filtered_xyz_df.any(axis=1)).sum()

# Display the results
print(f"The number of rows where exactly one theta_rad column is smaller than {theta_threshold} and all xyz columns are smaller than {xyz_threshold}: {single_theta_non_zero_and_xyz}")
print(f"The number of rows where any theta_rad column is smaller than {theta_threshold} and any xyz column is smaller than {xyz_threshold}: {any_theta_non_zero_and_xyz}")

# Optionally, if you want to see the filenames that meet the criteria, you can also output those
single_theta_non_zero_and_xyz_filenames = filtered_df[(theta_non_zero_count == 1) & xyz_all_above_threshold]['name'].tolist()
any_theta_non_zero_and_xyz_filenames = filtered_df[(theta_non_zero_count > 0) & filtered_xyz_df.any(axis=1)]['name'].tolist()

print(f"Filenames with exactly one theta_rad column greater than {theta_threshold} and all xyz columns greater than {xyz_threshold}: {single_theta_non_zero_and_xyz_filenames}")
print(f"Filenames with any theta_rad column greater than {theta_threshold} and any xyz column greater than {xyz_threshold}: {any_theta_non_zero_and_xyz_filenames}")