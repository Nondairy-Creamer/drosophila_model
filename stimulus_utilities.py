import numpy as np
import pickle
from pathlib import Path

# All stimulus generating function should return
# Stimulus: [examples, time, points in space]
# Optic Flow: [examples, time, points in space, LR/UD optic flow]

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


