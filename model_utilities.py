import torch
import numpy as np
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
    def contrast_scaling_func(theta):
        filter_std = filter_std_base * 1 / np.cos(theta - np.pi / 2)
        return np.exp(-2 * np.pi**2 * filter_std**2 / wavelength**2)

    contrast_scaling_factor = contrast_scaling_func(sample_theta)

    # length scales for vertical sine waves (since the cylinder is flat vertically)
    vert_wavelength = 2 * np.tan(wavelength / 2) # length of sine wave in cartesian space
    theta_to_length = np.tan(theta_mesh - np.pi / 2)

    sine_wave_dict = {'right': [], 'left': [], 'up': [], 'down': []}
    tf_dict = {'right': [], 'left': [], 'up': [], 'down': []}
    # generate sine waves at a specific temporal frequency for up down left right
    # at phase 0, 180, 
    for tf in temporal_frequency:
        # -- rightward sine wave --
        sine_wave_right = contrast * np.sin(2 * np.pi * (phi_mesh / wavelength + tf * t_mesh))
        sine_wave_right_shifted = contrast * np.sin(2 * np.pi * (phi_mesh / wavelength + tf * t_mesh) + np.pi / 2)
        sine_wave_right *= contrast_scaling_factor[None, :]
        sine_wave_right_shifted *= contrast_scaling_factor[None, :]
        sine_wave_dict['right'].append(sine_wave_right)
        sine_wave_dict['right'].append(sine_wave_right_shifted)
        sine_wave_dict['right'].append(-sine_wave_right)
        sine_wave_dict['right'].append(-sine_wave_right_shifted)
        tf_dict['right'] += [tf] * 4

        # -- leftward sine wave --
        sine_wave_left = contrast * np.sin(2 * np.pi * (phi_mesh / wavelength - tf * t_mesh))
        sine_wave_left_shifted = contrast * np.sin(2 * np.pi * (phi_mesh / wavelength - tf * t_mesh) + np.pi / 2)
        sine_wave_left *= contrast_scaling_factor[None, :]
        sine_wave_left_shifted *= contrast_scaling_factor[None, :]
        sine_wave_dict['left'].append(sine_wave_left)
        sine_wave_dict['left'].append(sine_wave_left_shifted)
        sine_wave_dict['left'].append(-sine_wave_left)
        sine_wave_dict['left'].append(-sine_wave_left_shifted)
        tf_dict['left'] += [-tf] * 4

        # -- upward sine wave -- 
        sine_wave_up = contrast * np.sin(2 * np.pi * (theta_to_length / vert_wavelength + tf * t_mesh))
        sine_wave_up_shifted = contrast * np.sin(2 * np.pi * (theta_to_length / vert_wavelength + tf * t_mesh) + np.pi / 2)
        sine_wave_up *= contrast_scaling_factor[None, :]
        sine_wave_up_shifted *= contrast_scaling_factor[None, :]
        sine_wave_dict['up'].append(sine_wave_up)
        sine_wave_dict['up'].append(sine_wave_up_shifted)
        sine_wave_dict['up'].append(-sine_wave_up)
        sine_wave_dict['up'].append(-sine_wave_up_shifted)
        tf_dict['up'] += [tf] * 4

        # -- downward sine wave --
        sine_wave_down = contrast * np.sin(2 * np.pi * (theta_to_length / vert_wavelength - tf * t_mesh))
        sine_wave_down_shifted = contrast * np.sin(2 * np.pi * (theta_to_length / vert_wavelength - tf * t_mesh) + np.pi / 2)
        sine_wave_down *= contrast_scaling_factor[None, :]
        sine_wave_down_shifted *= contrast_scaling_factor[None, :]
        sine_wave_dict['down'].append(sine_wave_down)
        sine_wave_dict['down'].append(sine_wave_down_shifted)
        sine_wave_dict['down'].append(-sine_wave_down)
        sine_wave_dict['down'].append(-sine_wave_down_shifted)
        tf_dict['down'] += [-tf] * 4

    num_time, num_cell = phi_mesh.shape
    stimulus = np.zeros((0, num_time, num_cell))
    tf_out = []
    for direction in ['right', 'left', 'up', 'down']:
        this_stim = np.stack(sine_wave_dict[direction])
        stimulus = np.concatenate((stimulus, this_stim), axis=0)
        tf_out += tf_dict[direction]

    return stimulus, tf_out, contrast_scaling_func

def get_stim_to_cell_mapping(cell_info_df, column_info_df, input_neurons=('L1', 'L2', 'L3')):
    # --- define the external inputs to cells ---
    # in this case we will directly stimulate L1, L2, and L3 with negative contrast
    # we aren't stimulating receptors because they have a complicated relationship between location and viewing angle
    # we are using negative contrast because L1-3 are inhibited by light
    # each L1-3 neuron exists in a single visual column that corresponds to a point of light

    # we have a stimulus tensor that is [stimuli, time, columns]
    # we need to map it to a cell input that is [stimuli, time, neurons] where only the input_neurons receive the stimulus
    # throughout, ID is the actual unique ID of the neuron / column
    # index referes to its position in these arrays

    num_neurons = len(cell_info_df)
    neuron_indexes = np.arange(len(cell_info_df))  # index of every neuron in the model
    column_index_for_each_neuron = pd.Index(column_info_df.column_id).get_indexer(cell_info_df.column_id)  # for each neuron, the index of the column its in
    neuron_is_input = cell_info_df.type.isin(input_neurons)  # whether a neuron is being stimuluated
    
    # Only select input neurons whose columns are in column_info_df (i.e., column_index != -1)
    valid_column = column_index_for_each_neuron >= 0
    neuron_is_input_and_valid = neuron_is_input & valid_column
    
    cell_indices = neuron_indexes[neuron_is_input_and_valid]  # the indexes of the sitmulated neurons
    stimulus_indices = column_index_for_each_neuron[neuron_is_input_and_valid]  # the column indexes of the sitmulated neurons

    stim_to_cell_mapping = np.stack((stimulus_indices, cell_indices))

    return stim_to_cell_mapping

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