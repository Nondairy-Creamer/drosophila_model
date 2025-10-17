import torch
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


def create_sine_wave(sample_phi, sample_theta, contrast=1, wavelength=np.pi/6, temporal_frequency=[4],
                     dt=0.01, max_time=1):
    # generates what a point of light at sample_phi[i] and sample[theta_i] would look like at each time point
    # from the perspective of a spherical obsrever looking at a cylinder with a sine wave stimulus on it
    # positive tfs correspond to rightward. negative tfs correspond to leftward
    if type(temporal_frequency) is not list:
        temporal_frequency = [temporal_frequency]

    temporal_frequency = [np.abs(tf) for tf in temporal_frequency]

    filter_std_base = 2.5 * np.pi / 180  # radians
    t = np.arange(0, max_time, step=dt)
    t_mesh, phi_mesh = np.meshgrid(t, sample_phi, indexing='ij')
    t_mesh, theta_mesh = np.meshgrid(t, sample_theta, indexing='ij')

    # smooth the sign wave according to its location in theta. For simplicity, just assume that the ommatidia applies
    # gaussian filtering with std 2.5 degrees. As theta increases, this linearly scales the std of the filter
    # as a result, the amplitude of the sine wave at the top / bottom of the cylinder gets its amplitude damped
    filter_std = filter_std_base * 1 / np.cos(sample_theta - np.pi / 2)
    contrast_scaling_factor = np.exp(-2 * np.pi**2 * filter_std**2 / wavelength**2)

    # length scales for vertical sine waves (since the cylinder is flat vertically)
    vert_wavelength = 2 * np.tan(wavelength / 2) # length of sine wave in cartesian space
    theta_to_length = np.tan(theta_mesh - np.pi / 2)

    sine_wave_dict = {'right': [], 'left': [], 'up': [], 'down': []}
    for tf in temporal_frequency:
        # rightward sine wave
        sine_wave = contrast * np.sin(2 * np.pi * (phi_mesh / wavelength + tf * t_mesh))
        sine_wave *= contrast_scaling_factor[None, :]
        sine_wave_dict['right'].append(sine_wave)
        sine_wave_dict['right'].append(-sine_wave)

        # leftward sine wave
        sine_wave = contrast * np.sin(2 * np.pi * (phi_mesh / wavelength - tf * t_mesh))
        sine_wave *= contrast_scaling_factor[None, :]
        sine_wave_dict['left'].append(sine_wave)
        sine_wave_dict['left'].append(-sine_wave)

        # upward sine wave
        sine_wave = contrast * np.sin(2 * np.pi * (theta_to_length / vert_wavelength + tf * t_mesh))
        sine_wave *= contrast_scaling_factor[None, :]
        sine_wave_dict['up'].append(sine_wave)
        sine_wave_dict['up'].append(-sine_wave)

        # downward sine wave
        sine_wave = contrast * np.sin(2 * np.pi * (theta_to_length / vert_wavelength - tf * t_mesh))
        sine_wave *= contrast_scaling_factor[None, :]
        sine_wave_dict['down'].append(sine_wave)
        sine_wave_dict['down'].append(-sine_wave)

    # save a figure showing the stimuli
    for direction in ['right', 'left', 'up', 'down']:
        plt.figure()
        plt.imshow(sine_wave_dict[direction][0][:, :100], cmap='gray')
        plt.colorbar()
        plt.savefig('figures/sine_stim_' + direction + '.png')
        plt.close()

    plt.figure()
    plt.scatter(sample_theta, contrast_scaling_factor)
    plt.savefig('figures/contrast_scaling.png')
    plt.close()

    num_time, num_cell = phi_mesh.shape
    stimulus = np.zeros((0, num_time, num_cell))
    for direction in ['right', 'left', 'up', 'down']:
        this_stim = np.stack(sine_wave_dict[direction])
        stimulus = np.concatenate((stimulus, this_stim), axis=0)

    return stimulus

def map_stimulus_to_cells(stimulus, column_info_df, root_ids, stim_column_ids, input_neurons=('L1', 'L2', 'L3')):
    # --- define the external inputs to cells ---
    # in this case we will directly stimulate L1, L2, and L3 with negative contrast
    # we aren't stimulating receptors because they have a complicated relationship between location and viewing angle
    # we are using negative contrast because L1-3 are inhibited by light

    # we have a larger list of neuron ids for each neuron
    # we have a subset of that list that says for each neuron Id what column it belongs to
    # need to find the neuron_index in the larger list for each entry in the subset list
    # then we need to find the index in the stimulus that belongs to each column.
    # with those, we can assign the correct stimulus to each neuron
    column_info_df['root_ind'] = pd.Index(root_ids).get_indexer(column_info_df.root_id)
    column_info_df['column_ind'] = pd.Index(stim_column_ids).get_indexer(column_info_df.column_id)

    # get rid of any entries where we have column info but not cell info, or we have L2/L3 but not L1
    column_info_df = column_info_df.query('root_ind != -1 and column_ind != -1')

    # get the lam neurons
    input_neuron_bool = column_info_df['type'].isin(input_neurons)
    root_id_to_column_id = column_info_df[input_neuron_bool].copy()

    num_stim = stimulus.shape[0]
    num_time = stimulus.shape[1]
    num_neuron = len(root_ids)
    cell_inputs = torch.zeros((num_stim, num_time, num_neuron), device=stimulus.device, dtype=stimulus.dtype)
    neuron_idx = torch.as_tensor(root_id_to_column_id.root_ind.to_numpy(), device=stimulus.device, dtype=torch.long)
    column_idx = torch.as_tensor(root_id_to_column_id.column_ind.to_numpy(), device=stimulus.device, dtype=torch.long)
    cell_inputs[:, :, neuron_idx] = stimulus[:, :, column_idx]

    return cell_inputs

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