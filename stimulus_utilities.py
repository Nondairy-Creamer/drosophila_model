import numpy as np
from scipy.signal import fftconvolve


def create_sine_wave(sample_phi, sample_theta, contrast=1, wavelength=np.pi/6, temporal_frequency=[4],
                     dt=0.01, stim_time=1.0, pre_stim_time=0.0, post_stim_time=0.0):
    # generates what a point of light at sample_phi[i] and sample[theta_i] would look like at each time point
    # from the perspective of a spherical obsrever looking at a cylinder with a sine wave stimulus on it

    if type(temporal_frequency) is not list:
        temporal_frequency = [temporal_frequency]

    temporal_frequency = [np.abs(tf) for tf in temporal_frequency]

    filter_std_base = 2.5 * np.pi / 180  # radians
    t = np.arange(0, pre_stim_time + stim_time + post_stim_time, step=dt)
    pre_stim_index = int(pre_stim_time / dt)
    post_stim_index = int((pre_stim_time + stim_time) / dt)
    t[:pre_stim_index] = t[pre_stim_index]
    t[post_stim_index:] = t[post_stim_index - 1]
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

    blocks = ['right', 'left', 'up', 'down']
    sine_wave_dict = {i: [] for i in blocks}
    tf_dict = {i: [] for i in blocks}
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
        tf_dict['left'] += [tf] * 4

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
        tf_dict['down'] += [tf] * 4

    num_time, num_cell = phi_mesh.shape
    stimulus = np.zeros((0, num_time, num_cell))
    tf_out = np.zeros(0)
    for b in blocks:
        this_stim = np.stack(sine_wave_dict[b])
        stimulus = np.concatenate((stimulus, this_stim), axis=0)
        this_tf = np.stack(tf_dict[b])
        tf_out = np.concatenate((tf_out, this_tf))

    return stimulus, tf_out, contrast_scaling_factor, blocks

def create_random_tf_sine_wave(sample_phi, sample_theta, contrast=1, wavelength=np.pi/6, tf_amp=5, tf_time_scale=0.15, rng=np.random.default_rng(),
                               dt=0.01, stim_time=1.0, num_repeats=5):
    # generates what a point of light at sample_phi[i] and sample[theta_i] would look like at each time point
    # from the perspective of a spherical obsrever looking at a cylinder with a sine wave stimulus on it
    # the velocity of this sine wave is a smoothed gaussian random walk 
    
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
    tf_dict = {i: [] for i in blocks}

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
    tf_mat = np.tile(tf, (num_cell, 1)).T
    tf_dict['right_left'] += [tf_mat.reshape(num_repeats, time_per_stim, num_cell)] * 4

    # -- updown sine wave -- 
    sine_wave_up_down = make_ud_sine_wave(phase_init=0)
    sine_wave_up_down_shifted = make_ud_sine_wave(phase_init=np.pi/2)
    sine_wave_up_down *= contrast_scaling_factor[None, :]
    sine_wave_up_down_shifted *= contrast_scaling_factor[None, :]
    sine_wave_dict['up_down'].append(sine_wave_up_down.reshape(num_repeats, time_per_stim, num_cell))
    sine_wave_dict['up_down'].append(sine_wave_up_down_shifted.reshape(num_repeats, time_per_stim, num_cell))
    sine_wave_dict['up_down'].append((-sine_wave_up_down).reshape(num_repeats, time_per_stim, num_cell))
    sine_wave_dict['up_down'].append((-sine_wave_up_down_shifted).reshape(num_repeats, time_per_stim, num_cell))
    tf_mat = np.tile(tf, (num_cell, 1)).T
    tf_dict['up_down'] += [tf_mat.reshape(num_repeats, time_per_stim, num_cell)] * 4

    stimulus, target = consolidate_stim(sine_wave_dict, tf_dict)

    return stimulus, target, contrast_scaling_factor, blocks

def consolidate_stim(stim_dict, target_dict):
    time_per_stim, num_cell = next(iter(stim_dict.values()))[0].shape[1:]
    stimulus = np.zeros((0, time_per_stim, num_cell))
    target = np.zeros((0, time_per_stim, num_cell))

    for b in stim_dict:
        this_stim = np.concatenate(stim_dict[b])
        stimulus = np.concatenate((stimulus, this_stim), axis=0)
        this_target = np.concatenate(target_dict[b])
        target = np.concatenate((target, this_target))

    return stimulus, target
