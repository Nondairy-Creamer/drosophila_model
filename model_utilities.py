import torch
import numpy as np
from matplotlib import pyplot as plt


def create_sparse_matrix(synapse_counts_df, device=torch.device('cpu'), dtype=torch.float32, normalize_rows=False):
    index_dtype = torch.int64
    num_neurons = np.max((synapse_counts_df.pre_matrix_ind.values, synapse_counts_df.post_matrix_ind.values)) + 1

    # --- choose which axis is row/col ---
    # values → torch tensor on the right device and dtype
    values = torch.tensor(synapse_counts_df['synapse_counts'].values, dtype=dtype, device=device)

    # indices must be int64
    rows = torch.tensor(synapse_counts_df['post_matrix_ind'].values, dtype=index_dtype, device=device)
    cols = torch.tensor(synapse_counts_df['pre_matrix_ind'].values, dtype=index_dtype, device=device)

    # shape the indices as (2, nnz)
    indices = torch.stack([rows, cols], dim=0)

    # --- build sparse matrix ---
    # first construct with coo, then coalesce, then convert to CSR for speed
    coo = torch.sparse_coo_tensor(indices, values, size=(num_neurons, num_neurons), device=device)

    coo = coo.coalesce()  # coalesce: sums duplicates and sorts indices (required for reliable CSR conversion)
    # csr = coo.to_sparse_csr()  # dtype is preserved

    return coo

def create_sine_wave(sample_phi, sample_theta, contrast=1, wavelength=np.pi / 6, temporal_frequency=[4],
                     dt=0.01, max_time=1, plot_figs=False):
    # generates what a point of light at sample_phi[i] and sample[theta_i] would look like at each time point
    # from the perspective of a spherical obsrever looking at a cylinder with a sine wave stimulus on it
    # positive tfs correspond to rightward. negative tfs correspond to leftward
    if type(temporal_frequency) is not list:
        temporal_frequency = [temporal_frequency]

    temporal_frequency = [np.abs(tf) for tf in temporal_frequency]

    filter_std_base = 2.5  # degrees
    t = np.arange(0, max_time, step=dt)
    phi_mesh, t_mesh = np.meshgrid(sample_phi, t, indexing='ij')
    theta_mesh, t_mesh = np.meshgrid(sample_theta, t, indexing='ij')

    # smooth the sign wave according to its location in theta. For simplicity, just assume that the ommatidia applies
    # gaussian filtering with std 2.5 degrees. As theta increases, this linearly scales the std of the filter
    # as a result, the amplitude of the sine wave at the top / bottom of the cylinder gets its amplitude damped
    filter_std = filter_std_base * 1 / np.cos(sample_theta - np.pi / 2)
    contrast_scaling_factor = np.exp(-2 * filter_std ** 2 * wavelength ** 2)
    contrast_scaling_factor /= np.max(contrast_scaling_factor)

    # length scales for vertical sine waves (since the cylinder is flat vertically)
    vert_wavelength = 2 * np.tan(wavelength / 2) # length of sine wave in cartesian space
    theta_to_length = np.tan(theta_mesh - np.pi / 2)

    sine_wave_dict = {'right': [], 'left': [], 'up': [], 'down': []}
    for tf in temporal_frequency:
        # rightward sine wave
        sine_wave = contrast * np.sin(2 * np.pi * (phi_mesh / wavelength - tf * t_mesh))
        sine_wave *= contrast_scaling_factor[:, None]
        sine_wave_dict['right'].append(sine_wave)

        # leftward sine wave
        sine_wave = contrast * np.sin(2 * np.pi * (phi_mesh / wavelength + tf * t_mesh))
        sine_wave *= contrast_scaling_factor[:, None]
        sine_wave_dict['left'].append(sine_wave)

        # upward sine wave
        sine_wave = contrast * np.sin(2 * np.pi * (theta_to_length / vert_wavelength - tf * t_mesh))
        sine_wave *= contrast_scaling_factor[:, None]
        sine_wave_dict['up'].append(sine_wave)

        # downward sine wave
        sine_wave = contrast * np.sin(2 * np.pi * (theta_to_length / vert_wavelength + tf * t_mesh))
        sine_wave *= contrast_scaling_factor[:, None]
        sine_wave_dict['down'].append(sine_wave)

        if plot_figs:
            for direction in ['right', 'left', 'up', 'down']:
                plt.figure()
                plt.imshow(sine_wave_dict[direction][0], cmap='gray')
                plt.show()

            plt.figure()
            plt.scatter(sample_theta, contrast_scaling_factor)
            plt.show()

    return sine_wave_dict

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
    theta = np.acos(xyz[2, :] / r)
    phi = np.arctan2(xyz[1, :], xyz[0, :])

    return np.vstack((r, theta, phi))

def rtp_to_xyz(rtp):
    x = rtp[0, :] * np.sin(rtp[1, :]) * np.cos(rtp[2, :])
    y = rtp[0, :] * np.sin(rtp[1, :]) * np.sin(rtp[2, :])
    z = rtp[0, :] * np.cos(rtp[1, :])

    return np.vstack((x, y, z))

def scatter_3d_with_origin_axes(xyz):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = xyz[0, :], xyz[1, :], xyz[2, :]

    # Scatter plot
    sc = ax.scatter(x, y, z, c=z, cmap='viridis', s=20, alpha=0.8)

    # Equal aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Axis arrows starting at origin (0,0,0)
    axis_length = max_range * 1.2
    ax.quiver(0, 0, 0, axis_length, 0, 0, color='k', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, axis_length, 0, color='k', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, axis_length, color='k', arrow_length_ratio=0.1)

    # Labels at tips
    ax.text(axis_length, 0, 0, 'X', color='k', fontsize=12)
    ax.text(0, axis_length, 0, 'Y', color='k', fontsize=12)
    ax.text(0, 0, axis_length, 'Z', color='k', fontsize=12)

    # Labels + colorbar
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Scatter with Origin Axes")

    fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10, label="Z value (color)")
    plt.show()

def run_rnn(cell, inputs):
    """
    cell: SimpleRnn instance
    U: (batch, T, m)  # input sequences
    h0: (batch, n) initial state, defaults to zeros
    """
    num_batches, num_time, num_neurons = inputs.shape
    outputs = torch.zeros((num_batches, num_time, cell.num_neurons), device=inputs.device, dtype=inputs.dtype)

    for t in range(1, num_time):
        outputs[:, t, :] = cell(outputs[:, t-1, :], inputs[:, t-1, :])   # (B, n)

    return outputs