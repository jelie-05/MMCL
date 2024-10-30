import os
import numpy as np
import csv
from collections import Counter


def find_row_by_name(filename, target_name):
    with open(filename, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for perturbation in csv_reader:
            if perturbation['name'] == target_name:
                return perturbation
    return None

def disturb_matrices(perturbation_csv, target_name):

    perturbation = find_row_by_name(perturbation_csv, target_name)
    theta_rad1 = float(perturbation["theta_rad1"])/180 * 2 * np.pi
    theta_rad2 = float(perturbation["theta_rad2"])/180 * 2 * np.pi
    theta_rad3 = float(perturbation["theta_rad3"])/180 * 2 * np.pi
    x = float(perturbation["x"])
    y = float(perturbation["y"])
    z = float(perturbation["z"])

    R1 = np.array([[np.cos(theta_rad1), -np.sin(theta_rad1), 0, 0],
                   [np.sin(theta_rad1), np.cos(theta_rad1), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    R2 = np.array([[1, 0, 0, 0],
                   [0, np.cos(theta_rad2), -np.sin(theta_rad2), 0],
                   [0, np.sin(theta_rad2), np.cos(theta_rad2), 0],
                   [0, 0, 0, 1]])
    R3 = np.array([[np.cos(theta_rad3), 0, np.sin(theta_rad3), 0],
                   [0, 1, 0, 0],
                   [-np.sin(theta_rad3), 0, np.cos(theta_rad3), 0],
                   [0, 0, 0, 1]])
    rot_error = np.dot(R1, np.dot(R2, R3))

    translation_error = np.array([[0, 0, 0, x],
                                  [0, 0, 0, y],
                                  [0, 0, 0, z],
                                  [0, 0, 0, 0]])

    return rot_error, translation_error

def project_velodyne_to_camera(velodyne_points, im_shape, T_cam_velo, R_rect, P_rect, perturb_path, name, augmentation=None):
    """
    Projects Velodyne points onto the camera image plane and includes depth information.

    Args:
        velodyne_points (np.array): Array of Velodyne points (N, 4) where each point has (x, y, z, reflectance).
        T_cam_velo (np.array): Transformation matrix from Velodyne to camera frame.
        R_rect (np.array): Rectification matrix.
        P_rect (np.array): Projection matrix for the camera.

    Returns:
        np.array: Projected 2D points on the image plane with depth (M, 3), where each point has (x, y, z).
    """
    perturb_dir = os.path.dirname(perturb_path)

    R_error, T_error = disturb_matrices(perturbation_csv=perturb_path, target_name=name)
    T_cam_velo_err = T_cam_velo @ R_error + T_error

    if augmentation is not None:
        augmentation_csv = os.path.join(perturb_dir, augmentation)  # During eval
        rot_error, translation_error = disturb_matrices(perturbation_csv=augmentation_csv, target_name=name)
        T_cam_velo = np.dot(T_cam_velo, rot_error) + translation_error
    else:
        T_cam_velo = T_cam_velo

    # Combine the transformation: P_rect * R_rect * T_cam_velo
    full_transform = P_rect @ R_rect @ T_cam_velo  # Shape (3, 4)
    error_transform = P_rect @ R_rect @ T_cam_velo_err

    # Convert Velodyne points to homogeneous coordinates (N, 4)
    velo_hom = np.hstack((velodyne_points[:, :3], np.ones((velodyne_points.shape[0], 1))))

    # Apply the full transformation in one step
    points_2d_hom = (full_transform @ velo_hom.T).T  # Resulting shape (N, 3)

    # Normalize to get 2D coordinates and retain depth (z)
    x_y = points_2d_hom[:, :2] / points_2d_hom[:, 2][:, np.newaxis]  # Normalize x, y by z
    z = points_2d_hom[:, 2]  # Depth information

    # Concatenate x, y, and z to form (N, 3)
    points_2d = np.hstack((x_y, z[:, np.newaxis]))

    # Filter points within image boundaries
    img_width, img_height = im_shape
    mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < img_width) & (points_2d[:, 1] >= 0) & (points_2d[:, 1] < img_height)
    depth = points_2d[mask]

    ## Again for miscalibration
    # Apply the full transformation in one step
    points_2d_hom_err = (error_transform @ velo_hom.T).T  # Resulting shape (N, 3)

    # Normalize to get 2D coordinates and retain depth (z)
    x_y = points_2d_hom_err[:, :2] / points_2d_hom_err[:, 2][:, np.newaxis]  # Normalize x, y by z
    z = points_2d_hom_err[:, 2]  # Depth information

    # Concatenate x, y, and z to form (N, 3)
    points_2d_err = np.hstack((x_y, z[:, np.newaxis]))

    # Filter points within image boundaries
    img_width, img_height = im_shape
    mask = (points_2d_err[:, 0] >= 0) & (points_2d_err[:, 0] < img_width) & (points_2d_err[:, 1] >= 0) & (points_2d_err[:, 1] < img_height)
    depth_neg = points_2d_err[mask]

    return depth, depth_neg