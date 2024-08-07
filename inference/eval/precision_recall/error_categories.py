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
csv_file_path = os.path.join(root, 'data', 'kitti', 'perturbation_neg.csv')
df = pd.read_csv(csv_file_path)

# Load the list of names
names_file_path = os.path.join(os.path.dirname(__file__), 'outputs', 'output_3_fn.txt')  # Replace with the path to your names file
with open(names_file_path, 'r') as file:
    names_list = file.read().splitlines()

# Filter the DataFrame to include only rows whose 'name' is in the names_list
filtered_df = df[df['name'].isin(names_list)]

# # Calculate the number of non-zero values for each column
# # non_zero_counts = filtered_df[['x', 'y', 'z', 'theta_rad1', 'theta_rad2', 'theta_rad3']].apply(lambda col: (col != 0).sum())
# non_zero_counts = filtered_df[['theta_rad1', 'theta_rad2', 'theta_rad3']].apply(lambda col: (col != 0).sum())
# # Calculate the number of values where absolute(value) > 0.5 for each column
#
# angle_0_2 = filtered_df[['theta_rad1', 'theta_rad2', 'theta_rad3']].apply(lambda col: ((col.abs() <= 2) & (col != 0)).sum())
# angle_2_3 = filtered_df[['theta_rad1', 'theta_rad2', 'theta_rad3']].apply(lambda col: ((col.abs() > 2) & (col.abs() <= 3)).sum())
# angle_3_4 = filtered_df[['theta_rad1', 'theta_rad2', 'theta_rad3']].apply(lambda col: ((col.abs() > 3) & (col.abs() <= 4)).sum())
# angle_larger_4 = filtered_df[['theta_rad1', 'theta_rad2', 'theta_rad3']].apply(lambda col: (col.abs() > 4).sum())
#
# translation_02 = filtered_df[['x', 'y', 'z']].apply(lambda col: ((col.abs() <= 0.25) & (col != 0)).sum())
#
# non_zero_df = filtered_df[['x', 'y', 'z', 'theta_rad1', 'theta_rad2', 'theta_rad3']] != 0
# non_zero_count = non_zero_df.sum(axis=1)
# single_non_zero_count = (non_zero_count == 1).sum()
#
# # # Display the result
# # import ace_tools as tools; tools.display_dataframe_to_user(name="Non-zero Values Count by Column", dataframe=non_zero_counts.to_frame(name='Non-zero Count'))
#
# print(f'non_zero_counts:\n{non_zero_counts}')
# print(f'angle_0_2:\n{angle_0_2}')
# print(f'angle_2_3:\n{angle_2_3}')
# print(f'angle_3_4:\n{angle_3_4}')
# print(f'angle_larger_4:\n{angle_larger_4}')
# print(f'translation_02:\n{translation_02}')
# print(f'single_non_zero_count:\n{single_non_zero_count}')

# Cases:
# 1. Check hard & easy cases:
#   - Hard: all of the variables < threshold (trans: 0.2, angle: 2)
# 2. Based on number of error:
#   - 0-6
# 3.

# non_zero_df = filtered_df[['x', 'y', 'z', 'theta_rad1', 'theta_rad2', 'theta_rad3']] != 0
# non_zero_count = non_zero_df.sum(axis=1)
# non_zero_1 = (non_zero_count == 1).sum()
# non_zero_2 = (non_zero_count == 2).sum()
# non_zero_3 = (non_zero_count == 3).sum()
# non_zero_4 = (non_zero_count == 4).sum()
# non_zero_5 = (non_zero_count == 5).sum()
# non_zero_6 = (non_zero_count == 6).sum()
# print(f'non_zero_1: {non_zero_1}')
# print(f'non_zero_2: {non_zero_2}')
# print(f'non_zero_3: {non_zero_3}')
# print(f'non_zero_4: {non_zero_4}')
# print(f'non_zero_5: {non_zero_5}')
# print(f'non_zero_6: {non_zero_6}')
#
# angle_below_2 = filtered_df[['theta_rad1', 'theta_rad2', 'theta_rad3']].apply(lambda col: ((col.abs() <= 2) & (col != 0)).sum())
# angle_above_2 = filtered_df[['theta_rad1', 'theta_rad2', 'theta_rad3']].apply(lambda col: (col.abs() > 2).sum())
# print(f'angle_below_2: \n{angle_below_2}')
# print(f'angle_above_2: \n{angle_above_2}')

# Apply the threshold condition to the theta_rad columns
# theta_columns = ['theta_rad1', 'theta_rad2', 'theta_rad3']
# filtered_theta_df = filtered_df[theta_columns] > 2
#
# # Sum the boolean values across each row to count the number of theta_rad columns greater than the threshold
# non_zero_count = filtered_theta_df.sum(axis=1)
#
# # Count the number of rows where exactly one of the theta_rad columns is greater than the threshold
# single_non_zero_count = (non_zero_count == 1).sum()
# print(f'single_non_zero_count: {single_non_zero_count}')
# any_non_zero_count = (non_zero_count > 0).sum()
# print(f'any_non_zero_count: {any_non_zero_count}')
# print(filtered_theta_df)

# Filter the DataFrame to include only rows whose 'name' is in the names_list
filtered_df = df[df['name'].isin(names_list)]

# Apply the threshold condition to the theta_rad columns and the xyz columns
theta_columns = ['theta_rad1', 'theta_rad2', 'theta_rad3']
xyz_columns = ['x', 'y', 'z']

theta_threshold = 2
xyz_threshold = 0.2
# Create a boolean DataFrame for theta_rad columns greater than the threshold
filtered_theta_df = filtered_df[theta_columns] > theta_threshold

# Create a boolean DataFrame for xyz columns greater than the threshold
filtered_xyz_df = filtered_df[xyz_columns] > xyz_threshold

# Sum the boolean values across each row to count the number of theta_rad columns greater than the threshold
theta_non_zero_count = filtered_theta_df.sum(axis=1)

# Check if all xyz columns are greater than the threshold for each row
xyz_all_above_threshold = filtered_xyz_df.all(axis=1)

# Count the number of rows where exactly one of the theta_rad columns is greater than the threshold and all xyz columns are greater than the threshold
single_theta_non_zero_and_xyz = ((theta_non_zero_count == 1) & xyz_all_above_threshold).sum()

# Count the number of rows where any of the theta_rad columns is greater than the threshold and any of the xyz columns are greater than the threshold
any_theta_non_zero_and_xyz = ((theta_non_zero_count > 0) & filtered_xyz_df.any(axis=1)).sum()

# Display the results
print(f"The number of rows where exactly one theta_rad column is greater than {theta_threshold} and all xyz columns are greater than {xyz_threshold}: {single_theta_non_zero_and_xyz}")
print(f"The number of rows where any theta_rad column is greater than {theta_threshold} and any xyz column is greater than {xyz_threshold}: {any_theta_non_zero_and_xyz}")

# Optionally, if you want to see the filenames that meet the criteria, you can also output those
single_theta_non_zero_and_xyz_filenames = filtered_df[(theta_non_zero_count == 1) & xyz_all_above_threshold]['name'].tolist()
any_theta_non_zero_and_xyz_filenames = filtered_df[(theta_non_zero_count > 0) & filtered_xyz_df.any(axis=1)]['name'].tolist()

print(f"Filenames with exactly one theta_rad column greater than {theta_threshold} and all xyz columns greater than {xyz_threshold}: {single_theta_non_zero_and_xyz_filenames}")
print(f"Filenames with any theta_rad column greater than {theta_threshold} and any xyz column greater than {xyz_threshold}: {any_theta_non_zero_and_xyz_filenames}")