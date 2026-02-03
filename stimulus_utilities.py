import numpy as np
import pickle
from pathlib import Path
import torch

# All stimulus generating function should return
# Stimulus: [examples, time, points in space]
# Optic Flow: [examples, time, points in space, LR/UD optic flow]

# import h5py
import numpy as np

# def loadmat_v73(path):
#     out = {}
#     with h5py.File(path, "r") as f:
#         for k in f.keys():
#             out[k] = np.array(f[k])
#     return out

def make_lr_sine_wave(sample_phi, sample_theta, contrast=1, wavelength=np.pi/6, temporal_frequency=4,
                      phase_init=0, dt=0.01, stim_time=1.0, pre_stim_time=0.0, post_stim_time=0.0):
    """
    Generate a basic left-right sine wave stimulus.
    
    Args:
        sample_phi: azimuthal angles (radians)
        sample_theta: polar angles (radians)
        contrast: contrast of the sine wave
        wavelength: spatial wavelength (radians)
        temporal_frequency: temporal frequency (positive = rightward, negative = leftward)
        phase_init: initial phase offset
        dt: time step
        stim_time: stimulus duration
        pre_stim_time: pre-stimulus time
        post_stim_time: post-stimulus time
    
    Returns:
        sine_wave: [time, points in space] array
        optic_flow: [time, points in space, 2] array (LR/UD flow)
    """
    filter_std_base = 2.5 * np.pi / 180  # radians
    t = np.arange(0, pre_stim_time + stim_time + post_stim_time, step=dt)
    pre_stim_index = int(pre_stim_time / dt)
    post_stim_index = int((pre_stim_time + stim_time) / dt)
    t[:pre_stim_index] = t[pre_stim_index]
    t[post_stim_index:] = t[post_stim_index - 1]
    t_mesh, phi_mesh = np.meshgrid(t, sample_phi, indexing='ij')
    
    # smooth the sine wave according to its location in theta
    def contrast_scaling_func(theta):
        filter_std = filter_std_base * 1 / np.cos(theta - np.pi / 2)
        return np.exp(-2 * np.pi**2 * filter_std**2 / wavelength**2)
    
    contrast_scaling_factor = contrast_scaling_func(sample_theta)
    
    # Generate sine wave
    sine_wave = contrast * np.sin(2 * np.pi * (phi_mesh / wavelength + temporal_frequency * t_mesh) + phase_init)
    sine_wave *= contrast_scaling_factor[None, :]
    
    # Generate optic flow (only left/right component)
    optic_flow = np.zeros((sine_wave.shape[0], sine_wave.shape[1], 2))
    optic_flow[:, :, 0] = temporal_frequency / wavelength  # LR flow
    optic_flow[:, :, 1] = 0  # UD flow
    
    return sine_wave, optic_flow

def make_ud_sine_wave(sample_phi, sample_theta, contrast=1, wavelength=np.pi/6, temporal_frequency=4,
                      phase_init=0, dt=0.01, stim_time=1.0, pre_stim_time=0.0, post_stim_time=0.0):
    """
    Generate a basic up-down sine wave stimulus.
    
    Args:
        sample_phi: azimuthal angles (radians)
        sample_theta: polar angles (radians)
        contrast: contrast of the sine wave
        wavelength: spatial wavelength (radians)
        temporal_frequency: temporal frequency (positive = upward, negative = downward)
        phase_init: initial phase offset
        dt: time step
        stim_time: stimulus duration
        pre_stim_time: pre-stimulus time
        post_stim_time: post-stimulus time
    
    Returns:
        sine_wave: [time, points in space] array
        optic_flow: [time, points in space, 2] array (LR/UD flow)
    """
    filter_std_base = 2.5 * np.pi / 180  # radians
    t = np.arange(0, pre_stim_time + stim_time + post_stim_time, step=dt)
    pre_stim_index = int(pre_stim_time / dt)
    post_stim_index = int((pre_stim_time + stim_time) / dt)
    t[:pre_stim_index] = t[pre_stim_index]
    t[post_stim_index:] = t[post_stim_index - 1]
    t_mesh, theta_mesh = np.meshgrid(t, sample_theta, indexing='ij')
    
    # smooth the sine wave according to its location in theta
    def contrast_scaling_func(theta):
        filter_std = filter_std_base * 1 / np.cos(theta - np.pi / 2)
        return np.exp(-2 * np.pi**2 * filter_std**2 / wavelength**2)
    
    contrast_scaling_factor = contrast_scaling_func(sample_theta)
    
    # length scales for vertical sine waves (since the cylinder is flat vertically)
    vert_wavelength = 2 * np.tan(wavelength / 2)  # length of sine wave in cartesian space
    theta_to_length = np.tan(theta_mesh - np.pi / 2)
    
    # Generate sine wave
    sine_wave = contrast * np.sin(2 * np.pi * (theta_to_length / vert_wavelength + temporal_frequency * t_mesh) + phase_init)
    sine_wave *= contrast_scaling_factor[None, :]
    
    # Generate optic flow (only up/down component)
    optic_flow = np.zeros((sine_wave.shape[0], sine_wave.shape[1], 2))
    optic_flow[:, :, 0] = 0  # LR flow
    optic_flow[:, :, 1] = temporal_frequency / wavelength  # UD flow
    
    return sine_wave, optic_flow

def create_balanced_sine_wave(sample_phi, sample_theta, contrast=1, wavelength=np.pi/6, temporal_frequency=[4],
                              dt=0.01, stim_time=1.0, pre_stim_time=0.0, post_stim_time=0.0):
    """
    Generate balanced sine wave stimuli with multiple phases for left/right/up/down directions.
    This function calls make_lr_sine_wave and make_ud_sine_wave to create all variations.
    
    Args:
        sample_phi: azimuthal angles (radians)
        sample_theta: polar angles (radians)
        contrast: contrast of the sine wave
        wavelength: spatial wavelength (radians)
        temporal_frequency: list of temporal frequencies to test
        dt: time step
        stim_time: stimulus duration
        pre_stim_time: pre-stimulus time
        post_stim_time: post-stimulus time
    
    Returns:
        stimulus: [examples, time, points in space]
        optic_flow: [examples, time, points in space, LR/UD optic flow]
    """
    if type(temporal_frequency) is not list:
        temporal_frequency = [temporal_frequency]

    temporal_frequency = [np.abs(tf) for tf in temporal_frequency]

    blocks = ['right', 'left', 'up', 'down']
    sine_wave_dict = {i: [] for i in blocks}
    optic_flow_dict = {i: [] for i in blocks}
    
    # Generate sine waves at a specific temporal frequency for up down left right
    # at phase 0, π/2, π, 3π/2 (or equivalently 0, π/2, -0, -π/2)
    for tf in temporal_frequency:
        # -- rightward sine wave --
        sine_wave_right, optic_flow_right = make_lr_sine_wave(
            sample_phi, sample_theta, contrast, wavelength, tf, phase_init=0,
            dt=dt, stim_time=stim_time, pre_stim_time=pre_stim_time, post_stim_time=post_stim_time
        )
        sine_wave_right_shifted, optic_flow_right_shifted = make_lr_sine_wave(
            sample_phi, sample_theta, contrast, wavelength, tf, phase_init=np.pi/2,
            dt=dt, stim_time=stim_time, pre_stim_time=pre_stim_time, post_stim_time=post_stim_time
        )
        sine_wave_dict['right'].append(sine_wave_right[None, :, :])
        sine_wave_dict['right'].append(sine_wave_right_shifted[None, :, :])
        sine_wave_dict['right'].append((-sine_wave_right)[None, :, :])
        sine_wave_dict['right'].append((-sine_wave_right_shifted)[None, :, :])
        optic_flow_dict['right'].append(optic_flow_right[None, :, :, :])
        optic_flow_dict['right'].append(optic_flow_right_shifted[None, :, :, :])
        optic_flow_dict['right'].append(optic_flow_right[None, :, :, :])
        optic_flow_dict['right'].append(optic_flow_right_shifted[None, :, :, :])

        # -- leftward sine wave --
        sine_wave_left, optic_flow_left = make_lr_sine_wave(
            sample_phi, sample_theta, contrast, wavelength, -tf, phase_init=0,
            dt=dt, stim_time=stim_time, pre_stim_time=pre_stim_time, post_stim_time=post_stim_time
        )
        sine_wave_left_shifted, optic_flow_left_shifted = make_lr_sine_wave(
            sample_phi, sample_theta, contrast, wavelength, -tf, phase_init=np.pi/2,
            dt=dt, stim_time=stim_time, pre_stim_time=pre_stim_time, post_stim_time=post_stim_time
        )
        sine_wave_dict['left'].append(sine_wave_left[None, :, :])
        sine_wave_dict['left'].append(sine_wave_left_shifted[None, :, :])
        sine_wave_dict['left'].append((-sine_wave_left)[None, :, :])
        sine_wave_dict['left'].append((-sine_wave_left_shifted)[None, :, :])
        optic_flow_dict['left'].append(optic_flow_left[None, :, :, :])
        optic_flow_dict['left'].append(optic_flow_left_shifted[None, :, :, :])
        optic_flow_dict['left'].append(optic_flow_left[None, :, :, :])
        optic_flow_dict['left'].append(optic_flow_left_shifted[None, :, :, :])

        # -- upward sine wave --
        sine_wave_up, optic_flow_up = make_ud_sine_wave(
            sample_phi, sample_theta, contrast, wavelength, tf, phase_init=0,
            dt=dt, stim_time=stim_time, pre_stim_time=pre_stim_time, post_stim_time=post_stim_time
        )
        sine_wave_up_shifted, optic_flow_up_shifted = make_ud_sine_wave(
            sample_phi, sample_theta, contrast, wavelength, tf, phase_init=np.pi/2,
            dt=dt, stim_time=stim_time, pre_stim_time=pre_stim_time, post_stim_time=post_stim_time
        )
        sine_wave_dict['up'].append(sine_wave_up[None, :, :])
        sine_wave_dict['up'].append(sine_wave_up_shifted[None, :, :])
        sine_wave_dict['up'].append((-sine_wave_up)[None, :, :])
        sine_wave_dict['up'].append((-sine_wave_up_shifted)[None, :, :])
        optic_flow_dict['up'].append(optic_flow_up[None, :, :, :])
        optic_flow_dict['up'].append(optic_flow_up_shifted[None, :, :, :])
        optic_flow_dict['up'].append(optic_flow_up[None, :, :, :])
        optic_flow_dict['up'].append(optic_flow_up_shifted[None, :, :, :])

        # -- downward sine wave --
        sine_wave_down, optic_flow_down = make_ud_sine_wave(
            sample_phi, sample_theta, contrast, wavelength, -tf, phase_init=0,
            dt=dt, stim_time=stim_time, pre_stim_time=pre_stim_time, post_stim_time=post_stim_time
        )
        sine_wave_down_shifted, optic_flow_down_shifted = make_ud_sine_wave(
            sample_phi, sample_theta, contrast, wavelength, -tf, phase_init=np.pi/2,
            dt=dt, stim_time=stim_time, pre_stim_time=pre_stim_time, post_stim_time=post_stim_time
        )
        sine_wave_dict['down'].append(sine_wave_down[None, :, :])
        sine_wave_dict['down'].append(sine_wave_down_shifted[None, :, :])
        sine_wave_dict['down'].append((-sine_wave_down)[None, :, :])
        sine_wave_dict['down'].append((-sine_wave_down_shifted)[None, :, :])
        optic_flow_dict['down'].append(optic_flow_down[None, :, :, :])
        optic_flow_dict['down'].append(optic_flow_down_shifted[None, :, :, :])
        optic_flow_dict['down'].append(optic_flow_down[None, :, :, :])
        optic_flow_dict['down'].append(optic_flow_down_shifted[None, :, :, :])

    stimulus, optic_flow = consolidate_stim(sine_wave_dict, optic_flow_dict)

    return stimulus, optic_flow

def create_sine_wave(sample_phi, sample_theta, contrast=1, wavelength=np.pi/6, temporal_frequency=[4],
                     dt=0.01, stim_time=1.0, pre_stim_time=0.0, post_stim_time=0.0):
    """
    Legacy function name for backward compatibility.
    Calls create_balanced_sine_wave to generate balanced sine wave stimuli.
    """
    return create_balanced_sine_wave(
        sample_phi, sample_theta, contrast, wavelength, temporal_frequency,
        dt, stim_time, pre_stim_time, post_stim_time
    )

def create_random_tf_sine_wave(sample_phi, sample_theta, contrast=1, wavelength=np.pi/6, tf_amp=5, tf_time_scale=0.15, seed=None,
                               dt=0.01, stim_time=1.0, num_repeats=5):
    # generates what a point of light at sample_phi[i] and sample[theta_i] would look like at each time point
    # from the perspective of a spherical obsrever looking at a cylinder with a sine wave stimulus on it
    # the velocity of this sine wave is a smoothed gaussian random walk 
    # positive / negative temporal frequency is right / left and up / down sine waves
    
    # Create RNG from seed
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    # --- generate the space time coordinates ---
    filter_std_base = 2.5 * np.pi / 180  # radians
    # double the number of stim because we want to symmetrize the temporal frequnecies
    num_repeats = num_repeats * 2
    time_vect = np.arange(0, stim_time * num_repeats, step=dt)
    t_mesh, phi_mesh = np.meshgrid(time_vect, sample_phi, indexing='ij')
    t_mesh, theta_mesh = np.meshgrid(time_vect, sample_theta, indexing='ij')
    num_time, num_cell = phi_mesh.shape
    time_per_stim = num_time // num_repeats

    # --- generate the shifting temporal frequency ---
    tf = rng.standard_normal(num_time // 2)
    tf = np.concatenate((tf, -tf), axis=0)
    tf_filter = 1 / (np.sqrt(2 * np.pi * tf_time_scale**2)) * np.exp(-(time_vect-time_vect.mean())**2 / (2 * tf_time_scale**2)) * dt
    tf_filter /= tf_filter.sum()
    tf_filter = np.fft.ifftshift(tf_filter)  # shift for our circular convolution
    tf = np.fft.ifft(np.fft.fft(tf) * np.fft.fft(tf_filter)).real
    # rescale the tf after smoothing to get the correct temporal frequency
    expected_std = 1 / np.sqrt(2 * np.sqrt(np.pi)) * np.sqrt(dt / tf_time_scale)
    tf *= tf_amp / expected_std

    # smooth the sign wave according to its location in theta. For simplicity, just assume that the ommatidia applies
    # gaussian filtering with std 2.5 degrees. As theta increases, this linearly scales the std of the filter
    # as a result, the amplitude of the sine wave at the top / bottom of the cylinder gets its amplitude damped
    def contrast_scaling_func(theta):
        filter_std = filter_std_base * 1 / np.cos(theta - np.pi / 2)
        return np.exp(-2 * np.pi**2 * filter_std**2 / wavelength**2)

    contrast_scaling_factor = contrast_scaling_func(sample_theta)

    # length scales for vertical sine waves (since the cylinder is flat vertically)
    vert_wavelength = 2 * np.tan(wavelength / 2) # length of sine wave in cartesian space
    theta_to_length = np.tan(sample_theta - np.pi / 2)

    blocks = ['right_left', 'up_down']
    sine_wave_dict = {i: [] for i in blocks}
    optic_flow_dict = {i: [] for i in blocks}

    def make_lr_sine_wave(phase_init=0):
        phase = phase_init
        sine_wave = np.zeros_like(t_mesh)

        for ti in range(time_vect.shape[0]):
            phase += tf[ti] * dt
            sine_wave[ti, :] = contrast * np.sin(2 * np.pi * (sample_phi / wavelength + phase))

        return sine_wave

    def make_ud_sine_wave(phase_init=0):
        phase = phase_init
        sine_wave = np.zeros_like(t_mesh)

        for ti in range(time_vect.shape[0]):
            phase += tf[ti] * dt
            sine_wave[ti, :] = contrast * np.sin(2 * np.pi * (theta_to_length / vert_wavelength + phase))

        return sine_wave

    # -- right/left sine wave --
    sine_wave_right_left = make_lr_sine_wave(phase_init=0)
    sine_wave_right_left_shifted = make_lr_sine_wave(phase_init=np.pi/2)
    sine_wave_right_left *= contrast_scaling_factor[None, :]
    sine_wave_right_left_shifted *= contrast_scaling_factor[None, :]
    sine_wave_dict['right_left'].append(sine_wave_right_left.reshape(num_repeats, time_per_stim, num_cell))
    sine_wave_dict['right_left'].append(sine_wave_right_left_shifted.reshape(num_repeats, time_per_stim, num_cell))
    sine_wave_dict['right_left'].append((-sine_wave_right_left).reshape(num_repeats, time_per_stim, num_cell))
    sine_wave_dict['right_left'].append((-sine_wave_right_left_shifted).reshape(num_repeats, time_per_stim, num_cell))

    # add what the optic flow for the sine wave is at each point
    optic_flow_mat = np.tile(tf / wavelength, (num_cell, 1)).T.reshape(num_repeats, time_per_stim, num_cell)
    # this optic flow is only left/right so set the up/down flow to 0
    optic_flow_mat = np.concatenate((optic_flow_mat[:, :, :, None], np.zeros_like(optic_flow_mat[:, :, :, None])), axis=3)

    optic_flow_dict['right_left'] += [optic_flow_mat] * 4

    # -- updown sine wave -- 
    sine_wave_up_down = make_ud_sine_wave(phase_init=0)
    sine_wave_up_down_shifted = make_ud_sine_wave(phase_init=np.pi/2)
    sine_wave_up_down *= contrast_scaling_factor[None, :]
    sine_wave_up_down_shifted *= contrast_scaling_factor[None, :]
    sine_wave_dict['up_down'].append(sine_wave_up_down.reshape(num_repeats, time_per_stim, num_cell))
    sine_wave_dict['up_down'].append(sine_wave_up_down_shifted.reshape(num_repeats, time_per_stim, num_cell))
    sine_wave_dict['up_down'].append((-sine_wave_up_down).reshape(num_repeats, time_per_stim, num_cell))
    sine_wave_dict['up_down'].append((-sine_wave_up_down_shifted).reshape(num_repeats, time_per_stim, num_cell))

    # add what the optic flow for the sine wave is at each point
    # phase increases for upward motion (tf > 0), so upward flow should be positive
    # phase decreases for downward motion (tf < 0), so downward flow should be negative
    optic_flow_mat = np.tile(tf / wavelength, (num_cell, 1)).T.reshape(num_repeats, time_per_stim, num_cell)
    # this optic flow is only up/down so set the left/right flow to 0
    optic_flow_mat = np.concatenate((np.zeros_like(optic_flow_mat[:, :, :, None]), optic_flow_mat[:, :, :, None]), axis=3)
    optic_flow_dict['up_down'] += [optic_flow_mat] * 4

    stimulus, optic_flow = consolidate_stim(sine_wave_dict, optic_flow_dict)

    return stimulus, optic_flow


# def get_random_velocity_pano(sample_phi, sample_theta, contrast=1, wavelength=np.pi/6, velocity_amp=180, velocity_time_scale=0.25, seed=None,
#                              dt=0.01, stim_time=1.0, num_repeats=5):
#     # generates what a point of light at sample_phi[i] and sample[theta_i] would look like at each time point
#     # from the perspective of a spherical obsrever looking at a cylinder with a sine wave stimulus on it
#     # the velocity of this sine wave is a smoothed gaussian random walk 
#     # positive / negative temporal frequency is right / left and up / down sine waves
    
#     # Create RNG from seed
#     if seed is not None:
#         rng = np.random.default_rng(seed)
#     else:
#         rng = np.random.default_rng()
    
#     # --- load in the stimuli ---
#     pano = loadmat_v73('/Users/mcreamer/Documents/python/connectome_data/panoramic/0001-120515-142521.mat').T
#     pano_y_lim = -np.array([47, -58]) * np.pi / 180 + np.pi / 2
#     pano_x_lim = np.array([0, 2 * np.pi])

#     # --- generate the space time coordinates ---
#     # double the number of stim because we want to symmetrize the temporal frequnecies
#     num_repeats = num_repeats * 2
#     time_vect = np.arange(0, stim_time * num_repeats, step=dt)
#     t_mesh, phi_mesh = np.meshgrid(time_vect, sample_phi, indexing='ij')
#     t_mesh, theta_mesh = np.meshgrid(time_vect, sample_theta, indexing='ij')
#     num_time, num_cell = phi_mesh.shape
#     time_per_stim = num_time // num_repeats

#     # --- generate the shifting velocity and angle ---
#     #random velocity
#     velocity = rng.standard_normal(num_time // 2)
#     velocity = np.concatenate((velocity, -velocity), axis=0)
#     angle = rng.uniform(low=0.0, high=2*np.pi, size=num_time)

#     d_phi = velocity * np.cos(angle)
#     d_theta = velocity * np.sin(angle)

#     # make the filter of the velocity
#     velocity_filter = 1 / (np.sqrt(2 * np.pi * velocity_time_scale**2)) * np.exp(-(time_vect-time_vect.mean())**2 / (2 * velocity_time_scale**2)) * dt
#     velocity_filter /= velocity_filter.sum()
#     velocity_filter = np.fft.ifftshift(velocity_filter)  # shift for our circular convolution
#     expected_std = 1 / np.sqrt(2 * np.sqrt(np.pi)) * np.sqrt(dt / velocity_time_scale)

#     # filter the dx and dy
#     d_phi = np.fft.ifft(np.fft.fft(d_phi) * np.fft.fft(velocity_filter)).real
#     d_theta = np.fft.ifft(np.fft.fft(d_theta) * np.fft.fft(velocity_filter)).real
#     # rescale the tf after smoothing to get the correct amplitude
#     d_phi *= velocity_amp / expected_std
#     d_theta *= velocity_amp / expected_std



#     return stimulus, np.stack((d_phi, d_theta)).T

def consolidate_stim(stim_dict, optic_flow_dict):
    time_per_stim, num_cell = next(iter(stim_dict.values()))[0].shape[1:]
    stimulus = np.zeros((0, time_per_stim, num_cell))
    optic_flow = np.zeros((0, time_per_stim, num_cell, 2))

    for b in stim_dict:
        this_stim = np.concatenate(stim_dict[b])
        stimulus = np.concatenate((stimulus, this_stim), axis=0)
        this_optic_flow = np.concatenate(optic_flow_dict[b])
        optic_flow = np.concatenate((optic_flow, this_optic_flow))

    return stimulus, optic_flow

def get_stimulus(stimulus_type: str, config, column_phi_theta=None):
    device = torch.device(config['model']['device'])
    dtype = getattr(torch, config['model']['dtype'])
    
    model_dt = config['model']['dt']
    stim_time = config['stimulus']['stim_time']  # seconds the sine is moving
    pre_stim_time = config['stimulus']['pre_stim_time']  # seconds sine is stationary before the sine wave starts moving
    post_stim_time = config['stimulus']['post_stim_time']  # seconds sine is stationary after the sine wave finishes moving
    temporal_frequencies = config['stimulus']['temporal_frequencies']
    stimulus_seed = config['stimulus']['seed']

    if stimulus_type == 'static_sine':
        if column_phi_theta is None:
            raise Exception('sine waves require column_phi_theta input')
        # --- get a sign wave stimulus ---
        # generate a sine wave on a cylinder around the fly
        stimulus, optic_flow = create_balanced_sine_wave(
            column_phi_theta[:, 0], column_phi_theta[:, 1],
            temporal_frequency=temporal_frequencies, dt=model_dt, stim_time=stim_time, pre_stim_time=pre_stim_time, post_stim_time=post_stim_time
        )

        stimulus = torch.tensor(stimulus, device=device, dtype=dtype)
        stimulus = (stimulus + 1.0) / 2.0
        optic_flow = torch.tensor(optic_flow, device=device, dtype=dtype)

    elif stimulus_type == 'random_pano':
        if column_phi_theta is None:
            raise Exception('panoramic require column_phi_theta input')

        # get pano stimulus with random velocity
        stimulus, optic_flow = get_random_velocity_pano(
            column_phi_theta[:, 0], column_phi_theta[:, 1],
            velocity_amp=config['stimulus']['tf_amp'], velocity_time_scale=config['stimulus']['tf_time_scale'],
            num_repeats=config['stimulus']['num_repeats'], dt=model_dt, stim_time=stim_time, seed=stimulus_seed
        )

        stimulus = torch.tensor(stimulus, device=device, dtype=dtype)
        stimulus = (stimulus + 1.0) / 2.0
        optic_flow = torch.tensor(optic_flow, device=device, dtype=dtype)

    elif stimulus_type == 'random_sine':
        if column_phi_theta is None:
            raise Exception('sine waves require column_phi_theta input')

        # --- generate a sine wave with random tf over time
        stimulus, optic_flow = create_random_tf_sine_wave(
            column_phi_theta[:, 0], column_phi_theta[:, 1],
            tf_amp=config['stimulus']['tf_amp'], tf_time_scale=config['stimulus']['tf_time_scale'],
            num_repeats=config['stimulus']['num_repeats'], dt=model_dt, stim_time=stim_time, seed=stimulus_seed
        )

        stimulus = torch.tensor(stimulus, device=device, dtype=dtype)
        optic_flow = torch.tensor(optic_flow, device=device, dtype=dtype)

    elif stimulus_type == 'flow_360':
        final_length = 98
        # --- grab the natural movie stimulus ---
        stimulus_path = Path('../connectome_data/flow360_data_right_eye_mapped.pkl')
        with open(stimulus_path, 'rb') as f:
            stimulus_in = pickle.load(f)

        # most stimuli are 99 points long. some are 199. shorten so they're all the same
        expected_length = 99
        stimulus_dt = 1 / stimulus_in['framerate']
        for i in reversed(range(len(stimulus_in['responses']))):
            if stimulus_in['responses'][i].shape[0] > expected_length:
                long_run = stimulus_in['responses'].pop(i)
                stimulus_in['responses'].append(long_run[:expected_length])
                stimulus_in['responses'].append(long_run[expected_length:2*expected_length])

                long_run_vect = stimulus_in['forward_flows'].pop(i)
                stimulus_in['forward_flows'].append(long_run_vect[:expected_length])
                stimulus_in['forward_flows'].append(long_run_vect[expected_length:2*expected_length])

        stimulus = np.stack(stimulus_in['responses'])
        optic_flow = np.stack(stimulus_in['forward_flows'])
        stimulus /= 255.0

        stimulus = torch.tensor(stimulus, device=device, dtype=dtype)
        optic_flow = torch.tensor(optic_flow, device=device, dtype=dtype)
        
        # Interpolate to model timestep
        stimulus = interpolate_to_model_dt(stimulus, stimulus_dt, model_dt)
        optic_flow = interpolate_to_model_dt(optic_flow, stimulus_dt, model_dt)
        
        # Split into non-overlapping segments of final_length
        num_stimuli, time_length, num_neurons = stimulus.shape
        num_segments = time_length // final_length
        if num_segments > 0:
            # Reshape to split time dimension: (stimuli, segments*final_length, neurons) -> (stimuli*segments, final_length, neurons)
            stimulus = stimulus[:, :num_segments*final_length, :].reshape(num_stimuli * num_segments, final_length, num_neurons)
            # Same for optic_flow: (stimuli, segments*final_length, neurons, 2) -> (stimuli*segments, final_length, neurons, 2)
            optic_flow = optic_flow[:, :num_segments*final_length, :, :].reshape(num_stimuli * num_segments, final_length, num_neurons, optic_flow.shape[3])

    elif stimulus_type == 'sintel':
        final_length = 100
        sintel_path = Path('../connectome_data/sintel_data.pkl')
        with open(sintel_path, 'rb') as f:
            stimulus_in = pickle.load(f)

        expected_length = 49
        stimulus_dt = 1 / stimulus_in['framerate']
        stimulus_list = [torch.tensor(i, device=device, dtype=dtype) for i in stimulus_in['responses'] if i.shape[0] == expected_length]
        optic_flow_list = [torch.tensor(i, device=device, dtype=dtype) for i in stimulus_in['forward_flows'] if i.shape[0] == expected_length]

        stimulus = torch.stack(stimulus_list, axis=0)
        optic_flow = torch.stack(optic_flow_list, axis=0)
        optic_flow[:, :, :, 1] *= -1
        stimulus /= 255.0

        # Interpolate to model timestep
        stimulus = interpolate_to_model_dt(stimulus, stimulus_dt, model_dt)
        optic_flow = interpolate_to_model_dt(optic_flow, stimulus_dt, model_dt)
        
        # Split into non-overlapping segments of final_length
        num_stimuli, time_length, num_neurons = stimulus.shape
        num_segments = time_length // final_length
        if num_segments > 0:
            # Reshape to split time dimension: (stimuli, segments*final_length, neurons) -> (stimuli*segments, final_length, neurons)
            stimulus = stimulus[:, :num_segments*final_length, :].reshape(num_stimuli * num_segments, final_length, num_neurons)
            # Same for optic_flow: (stimuli, segments*final_length, neurons, 2) -> (stimuli*segments, final_length, neurons, 2)
            optic_flow = optic_flow[:, :num_segments*final_length, :, :].reshape(num_stimuli * num_segments, final_length, num_neurons, optic_flow.shape[3])

    # --- plot the stimuli ---
    # if column_phi_theta is not None:
    #     pu.xt_plot(column_phi_theta[:, 0], column_phi_theta[:, 1], optic_flow=optic_flow.detach().cpu().numpy(), save_folder='figures/stimulus', run_tag=seed)

    # Add a blank stimulus (all zeros) with zero optic flow
    blank_stim = torch.zeros((1, stimulus.shape[1], stimulus.shape[2]), device=device, dtype=dtype)
    blank_flow = torch.zeros((1, optic_flow.shape[1], optic_flow.shape[2], optic_flow.shape[3]), device=device, dtype=dtype)
    stimulus = torch.cat([blank_stim, stimulus], dim=0)
    optic_flow = torch.cat([blank_flow, optic_flow], dim=0)

    # Add pre-stimulus interval (zero stimulus with zero optic flow) to the beginning of each stimulus
    pre_stim_points = int(pre_stim_time / model_dt)
    if pre_stim_points > 0:
        pre_stim = torch.zeros((stimulus.shape[0], pre_stim_points, stimulus.shape[2]), device=device, dtype=dtype)
        pre_flow = torch.zeros((optic_flow.shape[0], pre_stim_points, optic_flow.shape[2], optic_flow.shape[3]), device=device, dtype=dtype)
        stimulus = torch.cat([pre_stim, stimulus], dim=1)
        optic_flow = torch.cat([pre_flow, optic_flow], dim=1)

    return stimulus, optic_flow

def interpolate_to_model_dt(data, stimulus_dt, model_dt):
    """
    Linearly interpolate movie data from stimulus_dt to model_dt.
    
    Args:
        data: Tensor of shape (batch, time, ...) - can be numpy array or torch tensor
        stimulus_dt: Original time step of the data
        model_dt: Target time step for the model
        
    Returns:
        Interpolated data with new time dimension matching model_dt
        
    Raises:
        ValueError: If model_dt > stimulus_dt (cannot upsample in time)
    """
    if model_dt > stimulus_dt:
        raise ValueError(
            f"model_dt ({model_dt}) cannot be larger than stimulus_dt ({stimulus_dt}). "
            f"Cannot upsample temporal resolution."
        )
    
    # Convert to torch if numpy
    was_numpy = isinstance(data, np.ndarray)
    if was_numpy:
        data = torch.from_numpy(data)
    
    # Store original dtype to preserve it
    original_dtype = data.dtype
    
    original_shape = data.shape
    batch_size, original_time = original_shape[0], original_shape[1]
    remaining_dims = original_shape[2:]
    
    # Calculate new time dimension
    total_duration = (original_time - 1) * stimulus_dt
    new_time = int(total_duration / model_dt) + 1
    
    # Flatten all dimensions after time for interpolation
    # Shape: (batch, time, *remaining) -> (batch, -1, time) for F.interpolate
    data_flat = data.reshape(batch_size, original_time, -1)  # (batch, time, features)
    data_flat = data_flat.permute(0, 2, 1)  # (batch, features, time)
    
    # Interpolate along time dimension (F.interpolate requires float)
    data_interp = torch.nn.functional.interpolate(
        data_flat.float(), 
        size=new_time, 
        mode='linear', 
        align_corners=True
    )
    
    # Convert back to original dtype
    data_interp = data_interp.to(dtype=original_dtype)
    
    # Reshape back: (batch, features, new_time) -> (batch, new_time, *remaining)
    data_interp = data_interp.permute(0, 2, 1)  # (batch, new_time, features)
    new_shape = (batch_size, new_time) + remaining_dims
    data_interp = data_interp.reshape(new_shape)
    
    # Convert back to numpy if input was numpy
    if was_numpy:
        data_interp = data_interp.numpy()
    
    return data_interp


def get_synthetic_data(model, stimulus, input_neurons, output_neurons, init_noise_mult=5):
    """
    Generate synthetic target data by running a noise-perturbed model.
    
    Creates a copy of the model, adds noise to its parameters, runs the stimulus
    through it, and returns the firing rates as target data.
    
    Args:
        model: Neural model instance with add_noise_to_parameters and predict methods
        stimulus: Input stimulus tensor [batches, time, columns]
        input_neurons: List of input neuron types (e.g., ['L1', 'L2', 'L3'])
        output_neurons: List of output neuron types (e.g., ['T4a', 'T4b', ...])
        
    Returns:
        target: Dictionary mapping output neuron types to their firing rate tensors
    """
    import copy
    
    # Create a deep copy of the model to avoid modifying the original
    perturbed_model = copy.deepcopy(model)
    
    # Add noise to the copied model's parameters
    perturbed_model.add_noise_to_parameters(model.init_std * init_noise_mult)
    
    # Run the stimulus through the perturbed model
    with torch.no_grad():
        pred, _, _ = perturbed_model.predict(
            stimulus,
            input_neurons=input_neurons,
            output_neurons=output_neurons,
            init_state=None
        )
    
    # Extract the firing rates as target dictionary
    target = {neuron: pred['firing_rate'][neuron].clone() for neuron in output_neurons}
    
    return target, perturbed_model

