import os
import numpy as np
from PIL import Image
from collections import Counter
from scipy.interpolate import LinearNDInterpolator
from concurrent.futures import ProcessPoolExecutor
import torch

def read_calib_file(path):
    # taken from https://github.com/hunse/kitti
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass
    return data


def sub2ind(matrixSize, rowSub, colSub):
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1


def load_velodyne_points(file_name):
    # adapted from https://github.com/hunse/kitti
    points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points


def lin_interp(shape, xyd):
    # taken from https://github.com/hunse/kitti
    m, n = shape
    ij, d = xyd[:, 1::-1], xyd[:, 2]
    f = LinearNDInterpolator(ij, d, fill_value=0)
    J, I = np.meshgrid(np.arange(n), np.arange(m))
    IJ = np.vstack([I.flatten(), J.flatten()]).T
    disparity = f(IJ).reshape(shape)
    return disparity


# Function to randomly zero out at least one error
def choose_errors(theta_rad1, theta_rad2, theta_rad3, x, y, z):
    # Collect all errors in a list
    errors = [theta_rad1, theta_rad2, theta_rad3, x, y, z]
    # Randomly choose how many errors to zero out (1 or 2)
    num_zero_out = np.random.choice([1, 2, 3, 4, 5])
    # Randomly choose which errors to zero out
    zero_out_indices = np.random.choice(range(len(errors)), num_zero_out, replace=False)

    for index in zero_out_indices:
        errors[index] = 0

    return errors

def lidar_to_plane(cam2cam, velo2cam, velo, im_shape, cam=2, vel_depth=True):

    velo = velo[0].numpy()
    velo = velo[velo[:, 0] > 0, :]
    # compute projection matrix velodyne->image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3,:3] = cam2cam['R_rect_00'].reshape(3,3)
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3,4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:,:2] / velo_pts_im[:,2][..., np.newaxis]

    # depth information: 1st column of velo and 3rd column of velo_points_im
    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:,0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:,1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:,0] < im_shape[1]) & (velo_pts_im[:,1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((im_shape))
    depth[velo_pts_im[:, 1].astype(np.int_), velo_pts_im[:, 0].astype(np.int_)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds==dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth<0] = 0

    # For negative samples
    # Translation and Rotation Calibration Error
    # TODO: check why not all points translated?
    # Rotation Angles
    theta1 = np.random.uniform(1,5)  # Yaw
    theta_rad1 = theta1/180 * 2 * np.pi * np.random.choice([-1, 1])
    theta2 = np.random.uniform(1,5)  # Roll
    theta_rad2 = theta2 / 180 * 2 * np.pi * np.random.choice([-1, 1])
    theta3 = np.random.uniform(1,5)  # Pitch
    theta_rad3 = theta3 / 180 * 2 * np.pi * np.random.choice([-1, 1])

    # Translation
    x = np.random.uniform(0.1,0.5) * np.random.choice([-1, 1])  # in m
    y = np.random.uniform(0.1,0.5) * np.random.choice([-1, 1])
    z = np.random.uniform(0.1,0.5) * np.random.choice([-1, 1])

    theta_rad1, theta_rad2, theta_rad3, x, y, z = choose_errors(theta_rad1, theta_rad2, theta_rad3, x, y, z)

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

    Translation_error = np.array([[0, 0, 0, x],
                                  [0, 0, 0, y],
                                  [0, 0, 0, z],
                                 [0, 0, 0, 0]])

    velo2cam_error = np.dot(velo2cam, np.dot(R1, np.dot(R2, R3))) + Translation_error

    # compute projection matrix velodyne->image plane
    P_velo2im2 = np.dot(np.dot(P_rect, R_cam2rect), velo2cam_error)

    # project the points to the camera
    velo_pts_im2 = np.dot(P_velo2im2, velo.T).T
    velo_pts_im2[:, :2] = velo_pts_im2[:, :2] / velo_pts_im2[:, 2][..., np.newaxis]

    # depth information (z): 1st column of velo and 3rd column of velo_points_im
    if vel_depth:
        velo_pts_im2[:, 2] = velo[:, 0]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im2[:, 0] = np.round(velo_pts_im2[:, 0]) - 1
    velo_pts_im2[:, 1] = np.round(velo_pts_im2[:, 1]) - 1
    val_inds2 = (velo_pts_im2[:, 0] >= 0) & (velo_pts_im2[:, 1] >= 0)
    val_inds2 = val_inds2 & (velo_pts_im2[:, 0] < im_shape[1]) & (velo_pts_im2[:, 1] < im_shape[0])
    velo_pts_im2 = velo_pts_im2[val_inds2, :]

    # project to image
    depth_neg = np.zeros((im_shape))
    depth_neg[velo_pts_im2[:, 1].astype(np.int_), velo_pts_im2[:, 0].astype(np.int_)] = velo_pts_im2[:, 2]

    # find the duplicate points and choose the closest depth
    inds2 = sub2ind(depth_neg.shape, velo_pts_im2[:, 1], velo_pts_im2[:, 0])
    dupe_inds2 = [item for item, count in Counter(inds2).items() if count > 1]
    for dd in dupe_inds2:
        pts = np.where(inds2 == dd)[0]
        x_loc = int(velo_pts_im2[pts[0], 0])
        y_loc = int(velo_pts_im2[pts[0], 1])
        depth_neg[y_loc, x_loc] = velo_pts_im2[pts, 2].min()
    depth_neg[depth_neg < 0] = 0

    return depth, depth_neg

def process_lidar(cam2cam, velo2cam, lidar_item, im_shape):
    return lidar_to_plane(cam2cam=cam2cam, velo2cam=velo2cam, velo=lidar_item, im_shape=im_shape)

def lidar_batch_transform(cam2cam, velo2cam, lidar_batch, im_shape):
    # Combine the parameters into a single iterable using zip
    params_list = zip(lidar_batch)

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_lidar, [cam2cam] * len(lidar_batch),
                                    [velo2cam] * len(lidar_batch),
                                    params_list,
                                    [im_shape] * len(lidar_batch)))

    # Separate the results into depth_batch and distorted_batch
    depth_batch, distorted_batch = zip(*results)

    # Convert the lists to PyTorch tensors
    depth_batch_tensor = torch.tensor(np.array(depth_batch)).unsqueeze(1)
    distorted_batch_tensor = torch.tensor(np.array(distorted_batch)).unsqueeze(1)

    return depth_batch_tensor, distorted_batch_tensor
