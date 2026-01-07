import numpy as np
import pandas as pd


def rotate_2d(xy, angle):
    # data shape should be (2, n) with n data points
    # first dimension should be x, y
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return rotation_matrix @ xy

def rotate_3d(xyz, angle, axis):
    # data shape should be (3, n) with n data points
    # first dimension should be x, y, z
    assert axis in ('x', 'y', 'z'), f"axis must be 'x', 'y', or 'z', got {axis}"

    rotation_matrix = np.eye(3)
    rotation_matrix[1, 1] = np.cos(angle)
    rotation_matrix[1, 2] = -np.sin(angle)
    rotation_matrix[2, 1] = np.sin(angle)
    rotation_matrix[2, 2] = np.cos(angle)

    if axis == 'x':
        permute_order = (0, 1, 2)
    elif axis == 'y':
        permute_order = (2, 0, 1)
    else:  # axis must be z
        permute_order = (1, 2, 0)

    rotation_matrix = rotation_matrix[permute_order, :][:, permute_order]

    return rotation_matrix @ xyz

def xyz_to_rtp(xyz):
    r = np.sqrt(xyz[0, :]**2 + xyz[1, :]**2 + xyz[2, :]**2)
    theta = np.arccos(xyz[2, :] / r)
    phi = np.arctan2(xyz[1, :], xyz[0, :])

    return np.vstack((r, theta, phi))

def rtp_to_xyz(rtp):
    x = rtp[0, :] * np.sin(rtp[1, :]) * np.cos(rtp[2, :])
    y = rtp[0, :] * np.sin(rtp[1, :]) * np.sin(rtp[2, :])
    z = rtp[0, :] * np.cos(rtp[1, :])

    return np.vstack((x, y, z))
