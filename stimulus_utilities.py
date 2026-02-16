import numpy as np
import pickle
from pathlib import Path
import torch

# All stimulus generating function should return
# Stimulus: [examples, time, points in space]
# Optic Flow: [examples, time, points in space, LR/UD optic flow]

import h5py
import numpy as np

def loadmat_v73(path):
    out = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            out[k] = np.array(f[k])
    return out

def make_lr_sine_wave(sample_phi, sample_theta, contrast=1, wavelength=30, temporal_frequency=4,
                      phase_init=0, dt=0.01, stim_time=1.0, pre_stim_time=0.0, post_stim_time=0.0):
    """
    Generate a basic left-right sine wave stimulus.
    
    Args:
        sample_phi: azimuthal angles (radians)
        sample_theta: polar angles (radians)
        contrast: contrast of the sine wave
        wavelength: spatial wavelength (degrees)
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
    wavelength = wavelength * np.pi / 180  # convert degrees to radians
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
    
    # Generate optic flow assuming fly rotates in a virtual cylinder.
    # For LR sine waves, the fly rotates counter-clockwise about z at omega_z = velocity.
    from analysis.estimate_self_motion import generate_optic_flow
    velocity_rad_s = temporal_frequency * wavelength  # rad/s (wavelength already in radians)
    phi_t = torch.tensor(sample_phi, dtype=torch.float32)
    theta_t = torch.tensor(sample_theta, dtype=torch.float32)
    omega = torch.tensor([0.0, 0.0, float(velocity_rad_s)], dtype=torch.float32)
    T_vec = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    u_right, u_up = generate_optic_flow(phi_t, theta_t, omega, T_vec, 1.0)
    u_right_deg = u_right.numpy() * (180.0 / np.pi)
    u_up_deg = u_up.numpy() * (180.0 / np.pi)

    optic_flow = np.zeros((sine_wave.shape[0], sine_wave.shape[1], 2))
    optic_flow[pre_stim_index:post_stim_index, :, 0] = u_right_deg[None, :]
    optic_flow[pre_stim_index:post_stim_index, :, 1] = u_up_deg[None, :]
    
    sine_wave = (sine_wave + 1.0) / 2.0

    return sine_wave, optic_flow

def make_ud_sine_wave(sample_phi, sample_theta, contrast=1, wavelength=30, temporal_frequency=4,
                      phase_init=0, dt=0.01, stim_time=1.0, pre_stim_time=0.0, post_stim_time=0.0):
    """
    Generate a basic up-down sine wave stimulus.
    
    Args:
        sample_phi: azimuthal angles (radians)
        sample_theta: polar angles (radians)
        contrast: contrast of the sine wave
        wavelength: spatial wavelength (degrees)
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
    wavelength = wavelength * np.pi / 180  # convert degrees to radians
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
    
    # Generate optic flow assuming fly translates vertically in a virtual cylinder.
    # The cylinder wall is at distance Z = 1/sin(theta) (unit radius at theta=pi/2).
    # T_z is chosen so the flow at theta=pi/2 equals the requested velocity.
    from analysis.estimate_self_motion import generate_optic_flow
    velocity_rad_s = temporal_frequency * wavelength  # rad/s at theta=pi/2
    phi_t = torch.tensor(sample_phi, dtype=torch.float32)
    theta_t = torch.tensor(sample_theta, dtype=torch.float32)
    omega = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    T_vec = torch.tensor([0.0, 0.0, float(-velocity_rad_s)], dtype=torch.float32)
    Z = 1.0 / torch.sin(theta_t)
    u_right, u_up = generate_optic_flow(phi_t, theta_t, omega, T_vec, Z)
    u_right_deg = u_right.numpy() * (180.0 / np.pi)
    u_up_deg = u_up.numpy() * (180.0 / np.pi)

    optic_flow = np.zeros((sine_wave.shape[0], sine_wave.shape[1], 2))
    optic_flow[pre_stim_index:post_stim_index, :, 0] = u_right_deg[None, :]
    optic_flow[pre_stim_index:post_stim_index, :, 1] = u_up_deg[None, :]
    
    sine_wave = (sine_wave + 1.0) / 2.0

    return sine_wave, optic_flow

def create_balanced_sine_wave(sample_phi, sample_theta, contrast=1, wavelength=30, temporal_frequency=[4],
                              dt=0.01, stim_time=1.0, pre_stim_time=0.0, post_stim_time=0.0):
    """
    Generate balanced sine wave stimuli with multiple phases for left/right/up/down directions.
    This function calls make_lr_sine_wave and make_ud_sine_wave to create all variations.
    
    Args:
        sample_phi: azimuthal angles (radians)
        sample_theta: polar angles (radians)
        contrast: contrast of the sine wave
        wavelength: spatial wavelength (degrees)
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
        sine_wave_dict['right'].append((1.0 - sine_wave_right)[None, :, :])
        sine_wave_dict['right'].append((1.0 - sine_wave_right_shifted)[None, :, :])
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
        sine_wave_dict['left'].append((1.0 - sine_wave_left)[None, :, :])
        sine_wave_dict['left'].append((1.0 - sine_wave_left_shifted)[None, :, :])
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
        sine_wave_dict['up'].append((1.0 - sine_wave_up)[None, :, :])
        sine_wave_dict['up'].append((1.0 - sine_wave_up_shifted)[None, :, :])
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
        sine_wave_dict['down'].append((1.0 - sine_wave_down)[None, :, :])
        sine_wave_dict['down'].append((1.0 - sine_wave_down_shifted)[None, :, :])
        optic_flow_dict['down'].append(optic_flow_down[None, :, :, :])
        optic_flow_dict['down'].append(optic_flow_down_shifted[None, :, :, :])
        optic_flow_dict['down'].append(optic_flow_down[None, :, :, :])
        optic_flow_dict['down'].append(optic_flow_down_shifted[None, :, :, :])

    stimulus, optic_flow = consolidate_stim(sine_wave_dict, optic_flow_dict)

    return stimulus, optic_flow

def create_sine_wave(sample_phi, sample_theta, contrast=1, wavelength=30, temporal_frequency=[4],
                     dt=0.01, stim_time=1.0, pre_stim_time=0.0, post_stim_time=0.0):
    """
    Legacy function name for backward compatibility.
    Calls create_balanced_sine_wave to generate balanced sine wave stimuli.
    """
    return create_balanced_sine_wave(
        sample_phi, sample_theta, contrast, wavelength, temporal_frequency,
        dt, stim_time, pre_stim_time, post_stim_time
    )

def create_random_tf_sine_wave(sample_phi, sample_theta, contrast=1, wavelength=30, velocity_amp=150.0, velocity_time_scale=0.15, seed=None,
                               dt=0.01, stim_time=1.0, num_repeats=5):
    # generates what a point of light at sample_phi[i] and sample[theta_i] would look like at each time point
    # from the perspective of a spherical obsrever looking at a cylinder with a sine wave stimulus on it
    # the velocity of this sine wave is a smoothed gaussian random walk 
    # positive / negative temporal frequency is right / left and up / down sine waves
    
    # Convert velocity (deg/s) to temporal frequency (Hz): degrees cancel
    tf_amp = velocity_amp / wavelength
    # Convert wavelength to radians for spatial calculations
    wavelength = wavelength * np.pi / 180

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
    tf_filter = 1 / (np.sqrt(2 * np.pi * velocity_time_scale**2)) * np.exp(-(time_vect-time_vect.mean())**2 / (2 * velocity_time_scale**2)) * dt
    tf_filter /= tf_filter.sum()
    tf_filter = np.fft.ifftshift(tf_filter)  # shift for our circular convolution
    tf = np.fft.ifft(np.fft.fft(tf) * np.fft.fft(tf_filter)).real
    # rescale the tf after smoothing to get the correct temporal frequency
    expected_std = 1 / np.sqrt(2 * np.sqrt(np.pi)) * np.sqrt(dt / velocity_time_scale)
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

    # Compute optic flow for LR using generate_optic_flow (rotation about z-axis in cylinder)
    from analysis.estimate_self_motion import generate_optic_flow
    velocity_rad_s = tf * wavelength  # (num_time,) in rad/s
    phi_t = torch.tensor(sample_phi, dtype=torch.float32)
    theta_t = torch.tensor(sample_theta, dtype=torch.float32)
    omega_batch = torch.zeros(num_time, 3, dtype=torch.float32)
    omega_batch[:, 2] = torch.tensor(velocity_rad_s.copy(), dtype=torch.float32)
    T_batch = torch.zeros(num_time, 3, dtype=torch.float32)
    u_right_lr, u_up_lr = generate_optic_flow(phi_t, theta_t, omega_batch, T_batch, 1.0)
    lr_flow = np.stack([u_right_lr.numpy() * (180.0 / np.pi),
                        u_up_lr.numpy() * (180.0 / np.pi)], axis=-1)
    optic_flow_mat = lr_flow.reshape(num_repeats, time_per_stim, num_cell, 2)

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

    # Compute optic flow for UD using generate_optic_flow (vertical translation in cylinder)
    # Cylinder distance: Z = 1/sin(theta). T_z chosen so flow at theta=pi/2 = requested velocity.
    T_batch_ud = torch.zeros(num_time, 3, dtype=torch.float32)
    T_batch_ud[:, 2] = -torch.tensor(velocity_rad_s.copy(), dtype=torch.float32)
    omega_batch_ud = torch.zeros(num_time, 3, dtype=torch.float32)
    Z = 1.0 / torch.sin(theta_t)
    u_right_ud, u_up_ud = generate_optic_flow(phi_t, theta_t, omega_batch_ud, T_batch_ud, Z)
    ud_flow = np.stack([u_right_ud.numpy() * (180.0 / np.pi),
                        u_up_ud.numpy() * (180.0 / np.pi)], axis=-1)
    optic_flow_mat = ud_flow.reshape(num_repeats, time_per_stim, num_cell, 2)
    optic_flow_dict['up_down'] += [optic_flow_mat] * 4

    stimulus, optic_flow = consolidate_stim(sine_wave_dict, optic_flow_dict)
    stimulus = (stimulus + 1.0) / 2.0

    return stimulus, optic_flow


def create_directional_sine_wave(sample_phi, sample_theta, direction, contrast=1, wavelength=30, temporal_frequency=4,
                                 phase_init=0, dt=0.01, stim_time=1.0, pre_stim_time=0.0, post_stim_time=0.0):
    """
    Generate a single-direction sine wave stimulus.
    """
    direction = direction.lower()
    if direction in ["right", "left"]:
        tf = temporal_frequency if direction == "right" else -temporal_frequency
        sine_wave, optic_flow = make_lr_sine_wave(
            sample_phi, sample_theta, contrast, wavelength, tf, phase_init=phase_init,
            dt=dt, stim_time=stim_time, pre_stim_time=pre_stim_time, post_stim_time=post_stim_time
        )
    elif direction in ["up", "down"]:
        tf = temporal_frequency if direction == "up" else -temporal_frequency
        sine_wave, optic_flow = make_ud_sine_wave(
            sample_phi, sample_theta, contrast, wavelength, tf, phase_init=phase_init,
            dt=dt, stim_time=stim_time, pre_stim_time=pre_stim_time, post_stim_time=post_stim_time
        )
    else:
        raise ValueError(f"Unsupported direction for sine wave: {direction}")

    stimulus = sine_wave[None, :, :]
    optic_flow = optic_flow[None, :, :, :]
    return stimulus, optic_flow


def _random_rotation_matrix(rng):
    """Return a uniformly random 3x3 rotation matrix (Arvo 1992, fast uniform
    sampling of SO(3) via QR decomposition of a random matrix)."""
    z = rng.standard_normal((3, 3))
    q, r = np.linalg.qr(z)
    # Ensure a proper rotation (det = +1) by flipping signs of columns
    # whose corresponding diagonal entry of r is negative.
    d = np.diag(r)
    q *= np.sign(d)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def _load_all_panos():
    """Load all panoramic images from a single .pt file.

    The .pt file contains a (N, H, W) float32 tensor of raw panoramic images
    (created by consolidating the individual .mat files).

    Returns:
        panos: (N, H, W) numpy float64 array of raw panoramic images.
    """
    pt_path = Path('../connectome_data/all_panoramic.pt')
    if not pt_path.exists():
        raise FileNotFoundError(
            f"Consolidated panoramic file not found at {pt_path.resolve()}. "
            "Please run the consolidation step first."
        )
    panos = torch.load(pt_path, weights_only=True)
    return panos.numpy().astype(np.float64)


def _filter_pano(pano, filter_std_rad=2.5 * np.pi / 180):
    """Apply Gaussian low-pass filter to a panoramic image for anti-aliasing.

    Blurs the raw panorama with a Gaussian kernel (ommatidia optical filtering)
    so the image can be safely sampled at the sparse column grid without
    aliasing.  Contrast normalization is handled by the learnable input
    convolution in the model.

    Args:
        pano: (H, W) raw panoramic image array.
        filter_std_rad: Gaussian sigma in radians (default 2.5 deg).

    Returns:
        pano_out: (H, W) float64 array.
        pano_y_lim: (2,) theta limits of the panorama.
        pano_x_lim: (2,) phi limits of the panorama.
    """
    from scipy.ndimage import gaussian_filter

    pano_y_lim = -np.array([47, -58]) * np.pi / 180 + np.pi / 2  # theta range
    pano_x_lim = np.array([np.pi, -np.pi])  # phi range

    pano_height, pano_width = pano.shape
    y_pixels_per_rad = pano_height / (pano_y_lim[1] - pano_y_lim[0])
    x_pixels_per_rad = pano_width / (pano_x_lim[0] - pano_x_lim[1])
    sigma_y = filter_std_rad * y_pixels_per_rad
    sigma_x = filter_std_rad * x_pixels_per_rad

    pano_out = gaussian_filter(
        pano.astype(np.float64), sigma=[sigma_y, sigma_x], mode=['nearest', 'wrap']
    )

    return pano_out, pano_y_lim, pano_x_lim


def get_random_velocity_pano(sample_phi, sample_theta, velocity_amp=180, velocity_time_scale=0.25, seed=None,
                             dt=0.01, stim_time=1.0, num_repeats=5):
    """
    Generate panoramic stimuli with smoothly varying rotational velocity AND
    smoothly varying rotation axis.

    Loads all panoramic images from ``../connectome_data/panoramic/``, applies
    2D Gaussian spatial filtering (std=2.5 deg), then for **each** panorama
    simulates pure rotation with an independent temporally-smoothed random
    angular velocity vector (omega_x, omega_y, omega_z).  Each of the three
    components is an independent smoothed Gaussian random walk, so both the
    magnitude and direction of rotation evolve smoothly over time.  At each
    time step the viewing directions are incrementally rotated and the filtered
    panorama is sampled.  Optic flow is computed analytically (linear in omega).

    Coordinate convention: +x forward, +y left, +z up.

    Args:
        sample_phi: (N,) azimuthal angles of viewing directions (radians)
        sample_theta: (N,) polar angles of viewing directions (radians)
        velocity_amp: RMS amplitude of rotational speed (deg/s)
        velocity_time_scale: temporal smoothing scale for velocity (seconds)
        seed: random seed
        dt: time step (seconds)
        stim_time: duration per repeat (seconds)
        num_repeats: number of repeats *per panorama* (will be doubled for
            symmetrization)

    Returns:
        stimulus: [examples, time, points in space] — pixel values in [0, 1]
        optic_flow: [examples, time, points in space, 2] — (d_right, d_up) in deg/s
        rotation_vectors: [examples, time, 3] — body-frame angular velocity (rad/s)
    """
    from scipy.ndimage import map_coordinates
    from analysis.estimate_self_motion import generate_optic_flow

    # Create RNG from seed
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # --- load all panoramic images ---
    raw_panos = _load_all_panos()

    pano_images = []
    for i in range(raw_panos.shape[0]):
        pano_filtered, pano_y_lim, pano_x_lim = _filter_pano(raw_panos[i])
        pano_images.append(pano_filtered)

    # --- time / velocity parameters (shared across panoramas) ---
    num_repeats_total = num_repeats * 2  # double for symmetrization
    time_vect = np.arange(0, stim_time * num_repeats_total, step=dt)
    num_time = len(time_vect)
    time_per_stim = num_time // num_repeats_total

    # Gaussian smoothing filter (circular convolution)
    vel_filter = (1 / np.sqrt(2 * np.pi * velocity_time_scale**2)
                  * np.exp(-(time_vect - time_vect.mean())**2 / (2 * velocity_time_scale**2)) * dt)
    vel_filter /= vel_filter.sum()
    vel_filter = np.fft.ifftshift(vel_filter)

    expected_std = 1 / np.sqrt(2 * np.sqrt(np.pi)) * np.sqrt(dt / velocity_time_scale)
    velocity_amp_rad = velocity_amp * np.pi / 180  # convert deg/s to rad/s

    # Scale per-component amplitude so the total angular speed has the desired RMS
    # E[|omega|^2] = 3 * sigma_comp^2,  so sigma_comp = velocity_amp / sqrt(3)
    component_amp = velocity_amp_rad / np.sqrt(3)

    # --- convert sample viewing directions to Cartesian ---
    sample_xyz = np.stack([
        np.sin(sample_theta) * np.cos(sample_phi),
        np.sin(sample_theta) * np.sin(sample_phi),
        np.cos(sample_theta)
    ], axis=1)  # (N, 3)
    num_cells = len(sample_phi)

    T_zero = torch.tensor([0.0, 0.0, 0.0])
    phi_t = torch.tensor(sample_phi, dtype=torch.float32)
    theta_t = torch.tensor(sample_theta, dtype=torch.float32)

    stimulus_list = []
    optic_flow_list = []
    omega_body_list = []

    for pano_filtered in pano_images:
        pano_height, pano_width = pano_filtered.shape

        # --- generate 3 independent smoothed angular velocity components ---
        omega = np.zeros((3, num_time))
        for dim in range(3):
            v = rng.standard_normal(num_time // 2)
            v = np.concatenate((v, -v))  # symmetrize: second half is negated
            v = np.fft.ifft(np.fft.fft(v) * np.fft.fft(vel_filter)).real
            v *= component_amp / expected_std
            omega[dim] = v

        # Reshape into per-repeat segments: (3, num_repeats_total, time_per_stim)
        omega_per_repeat = omega.reshape(3, num_repeats_total, time_per_stim)

        for ri in range(num_repeats_total):
            omega_seg = omega_per_repeat[:, ri, :]  # (3, time_per_stim)

            # --- compose incremental rotations (Rodrigues' formula) ---
            # The update R = R_inc @ R is left-multiplication, so omega_seg is
            # in the world frame.  For optic-flow we need the body-frame
            # angular velocity: omega_body = R^T @ omega_world.
            # Start each repeat at a uniformly random orientation so the
            # observer samples different parts of the panorama across repeats.
            R = _random_rotation_matrix(rng)
            rotated_xyz = np.zeros((time_per_stim, num_cells, 3))
            omega_body = np.zeros((time_per_stim, 3))

            for ti in range(time_per_stim):
                omega_world = omega_seg[:, ti]
                angle = np.linalg.norm(omega_world) * dt
                if angle > 1e-12:
                    ax = omega_world / np.linalg.norm(omega_world)
                    K = np.array([[0, -ax[2], ax[1]],
                                  [ax[2], 0, -ax[0]],
                                  [-ax[1], ax[0], 0]])
                    R_inc = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
                    R = R_inc @ R
                rotated_xyz[ti] = (R @ sample_xyz.T).T
                omega_body[ti] = R.T @ omega_world

            # Convert back to spherical coordinates
            rotated_theta = np.arccos(np.clip(rotated_xyz[:, :, 2], -1, 1))
            rotated_phi = np.arctan2(rotated_xyz[:, :, 1], rotated_xyz[:, :, 0])

            # --- sample the filtered panoramic image ---
            y_indices = (rotated_theta - pano_y_lim[0]) / (pano_y_lim[1] - pano_y_lim[0]) * (pano_height - 1)
            x_indices = (pano_x_lim[0] - rotated_phi) / (pano_x_lim[0] - pano_x_lim[1]) * pano_width
            x_indices = x_indices % pano_width  # wrap horizontally

            # Mark out-of-bounds theta (above or below panorama) — set to mean
            out_of_bounds = (y_indices < 0) | (y_indices > pano_height - 1)
            y_indices = np.clip(y_indices, 0, pano_height - 1)

            stim = map_coordinates(
                pano_filtered, [y_indices.ravel(), x_indices.ravel()],
                order=1, mode='nearest'
            ).reshape(time_per_stim, num_cells)
            stim[out_of_bounds] = pano_filtered.mean()
            stimulus_list.append(stim)
            omega_body_list.append(omega_body)

            # --- compute optic flow (batched over time) ---
            omega_body_t = torch.tensor(omega_body, dtype=torch.float32)
            T_zeros = T_zero.expand(omega_body_t.shape[0], -1)
            u_right, u_up = generate_optic_flow(phi_t, theta_t, omega_body_t, T_zeros, 1.0)
            # generate_optic_flow returns rad/s; convert to deg/s
            of = np.stack([u_right.numpy(), u_up.numpy()], axis=-1) * (180.0 / np.pi)
            # Mark optic flow as NaN where stimulus is out-of-bounds (constant),
            # since the model cannot extract motion from a constant stimulus.
            # NaN-aware loss functions will ignore these positions.
            of[out_of_bounds, :] = np.nan
            optic_flow_list.append(of)

    # --- stack into final arrays ---
    stimulus = np.stack(stimulus_list)             # (num_panos * num_repeats_total, time_per_stim, num_cells)
    optic_flow = np.stack(optic_flow_list)         # (num_panos * num_repeats_total, time_per_stim, num_cells, 2)
    rotation_vectors = np.stack(omega_body_list)   # (num_panos * num_repeats_total, time_per_stim, 3)

    return stimulus, optic_flow, rotation_vectors


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

def get_stimulus(
    stimulus_type: str,
    config,
    column_phi_theta=None,
    *,
    max_duration_s: float | None = None,
    use_full_length: bool = False,
    sine_direction: str | None = None,
    sine_temporal_frequency: float | None = None,
    random_sine_duration_s: float | None = None,
    pre_stim_time_override: float | None = None,
    const_rand_const_time: float | None = None,
    const_rand_const_tf: float | None = None,
    const_rand_const_first: bool = True,
):
    device = torch.device(config['model']['device'])
    dtype = getattr(torch, config['model']['dtype'])
    middle_gray = 0.5

    model_dt = config['model']['dt']
    stim_time = config['stimulus']['stim_time']  # seconds the sine is moving
    pre_stim_time = config['stimulus']['pre_stim_time']  # seconds sine is stationary before the sine wave starts moving
    post_stim_time = config['stimulus']['post_stim_time']  # seconds sine is stationary after the sine wave finishes moving
    temporal_frequencies = config['stimulus']['temporal_frequencies']
    stimulus_seed = config['stimulus']['seed']
    if pre_stim_time_override is not None:
        pre_stim_time = float(pre_stim_time_override)
    sine_pre_stim_time = pre_stim_time
    sine_post_stim_time = post_stim_time

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
        optic_flow = torch.tensor(optic_flow, device=device, dtype=dtype)

    elif stimulus_type == 'directional_sine':
        if column_phi_theta is None:
            raise Exception('sine waves require column_phi_theta input')
        if sine_direction is None:
            raise ValueError('directional_sine requires sine_direction input')

        tf = float(sine_temporal_frequency) if sine_temporal_frequency is not None else 4.0
        sine_pre_stim_time = 0.0
        sine_post_stim_time = 0.0
        stimulus, optic_flow = create_directional_sine_wave(
            column_phi_theta[:, 0], column_phi_theta[:, 1],
            direction=sine_direction,
            temporal_frequency=tf,
            dt=model_dt,
            stim_time=stim_time,
            pre_stim_time=sine_pre_stim_time,
            post_stim_time=sine_post_stim_time,
        )
        stimulus = torch.tensor(stimulus, device=device, dtype=dtype)
        optic_flow = torch.tensor(optic_flow, device=device, dtype=dtype)

    elif stimulus_type == 'random_pano':
        if column_phi_theta is None:
            raise Exception('panoramic require column_phi_theta input')

        # get pano stimulus with random velocity
        stimulus, optic_flow, _ = get_random_velocity_pano(
            column_phi_theta[:, 0], column_phi_theta[:, 1],
            velocity_amp=config['stimulus']['velocity_amp'], velocity_time_scale=config['stimulus']['velocity_time_scale'],
            num_repeats=config['stimulus']['num_repeats'], dt=model_dt, stim_time=stim_time, seed=stimulus_seed
        )

        stimulus = torch.tensor(stimulus, device=device, dtype=dtype)
        optic_flow = torch.tensor(optic_flow, device=device, dtype=dtype)

    elif stimulus_type == 'random_sine':
        if column_phi_theta is None:
            raise Exception('sine waves require column_phi_theta input')

        # --- generate a sine wave with random tf over time
        random_sine_stim_time = stim_time
        random_sine_num_repeats = config['stimulus']['num_repeats']
        if random_sine_duration_s is not None:
            if random_sine_duration_s <= 0:
                raise ValueError("random_sine_duration_s must be positive")
            random_sine_stim_time = float(random_sine_duration_s) / 2.0
            random_sine_num_repeats = 1
        stimulus, optic_flow = create_random_tf_sine_wave(
            column_phi_theta[:, 0], column_phi_theta[:, 1],
            velocity_amp=config['stimulus']['velocity_amp'], velocity_time_scale=config['stimulus']['velocity_time_scale'],
            num_repeats=random_sine_num_repeats, dt=model_dt, stim_time=random_sine_stim_time, seed=stimulus_seed
        )

        stimulus = torch.tensor(stimulus, device=device, dtype=dtype)
        optic_flow = torch.tensor(optic_flow, device=device, dtype=dtype)

    elif stimulus_type == 'flow_360':
        default_length = 98
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
        
        # Resolve desired segment length after interpolation
        _, time_length, _ = stimulus.shape
        if use_full_length:
            final_length = time_length
        elif max_duration_s is not None:
            if max_duration_s <= 0:
                final_length = time_length
            else:
                final_length = min(time_length, int(round(max_duration_s / model_dt)))
        else:
            final_length = min(time_length, default_length)
        final_length = max(1, int(final_length))

        # Split into non-overlapping segments of final_length
        num_stimuli, time_length, num_neurons = stimulus.shape
        num_segments = time_length // final_length
        if num_segments > 0:
            # Reshape to split time dimension: (stimuli, segments*final_length, neurons) -> (stimuli*segments, final_length, neurons)
            stimulus = stimulus[:, :num_segments*final_length, :].reshape(num_stimuli * num_segments, final_length, num_neurons)
            # Same for optic_flow: (stimuli, segments*final_length, neurons, 2) -> (stimuli*segments, final_length, neurons, 2)
            optic_flow = optic_flow[:, :num_segments*final_length, :, :].reshape(num_stimuli * num_segments, final_length, num_neurons, optic_flow.shape[3])

    elif stimulus_type == 'sintel':
        default_length = 100
        sintel_path = Path('../connectome_data/sintel_data.pkl')
        with open(sintel_path, 'rb') as f:
            stimulus_in = pickle.load(f)

        expected_length = 49
        stimulus_dt = 1 / stimulus_in['framerate']
        # Filter jointly to keep stimulus-flow pairs aligned
        stimulus_list = []
        optic_flow_list = []
        for resp, flow in zip(stimulus_in['responses'], stimulus_in['forward_flows']):
            if resp.shape[0] == expected_length and flow.shape[0] == expected_length:
                stimulus_list.append(torch.tensor(resp, device=device, dtype=dtype))
                optic_flow_list.append(torch.tensor(flow, device=device, dtype=dtype))

        stimulus = torch.stack(stimulus_list, axis=0)
        optic_flow = torch.stack(optic_flow_list, axis=0)
        optic_flow[:, :, :, 1] *= -1
        stimulus /= 255.0

        # Interpolate to model timestep
        stimulus = interpolate_to_model_dt(stimulus, stimulus_dt, model_dt)
        optic_flow = interpolate_to_model_dt(optic_flow, stimulus_dt, model_dt)
        
        # Resolve desired segment length after interpolation
        _, time_length, _ = stimulus.shape
        if use_full_length:
            final_length = time_length
        elif max_duration_s is not None:
            if max_duration_s <= 0:
                final_length = time_length
            else:
                final_length = min(time_length, int(round(max_duration_s / model_dt)))
        else:
            final_length = min(time_length, default_length)
        final_length = max(1, int(final_length))

        # Split into non-overlapping segments of final_length
        num_stimuli, time_length, num_neurons = stimulus.shape
        num_segments = time_length // final_length
        if num_segments > 0:
            # Reshape to split time dimension: (stimuli, segments*final_length, neurons) -> (stimuli*segments, final_length, neurons)
            stimulus = stimulus[:, :num_segments*final_length, :].reshape(num_stimuli * num_segments, final_length, num_neurons)
            # Same for optic_flow: (stimuli, segments*final_length, neurons, 2) -> (stimuli*segments, final_length, neurons, 2)
            optic_flow = optic_flow[:, :num_segments*final_length, :, :].reshape(num_stimuli * num_segments, final_length, num_neurons, optic_flow.shape[3])
    
    elif stimulus_type == 'flyview':
        required_length = 80
        sintel_path = Path('../connectome_data/rendered_fly_view_dataset.pkl')
        with open(sintel_path, 'rb') as f:
            stimulus_in = pickle.load(f)

        stimulus_dt = 1 / stimulus_in['fps']
        # Split each entry into non-overlapping segments of required_length, discarding remainders.
        # Remove first time point from responses so lengths match forward_flows.
        stimulus_list = []
        optic_flow_list = []
        for resp, flow in zip(stimulus_in['responses'], stimulus_in['forward_flows']):
            stim = torch.tensor(resp[1:], device=device, dtype=dtype)
            of = torch.tensor(flow, device=device, dtype=dtype)
            num_segments = stim.shape[0] // required_length
            if num_segments == 0:
                continue
            for seg_idx in range(num_segments):
                start = seg_idx * required_length
                end = start + required_length
                stimulus_list.append(stim[start:end])
                optic_flow_list.append(of[start:end])

        stimulus = torch.stack(stimulus_list, axis=0)
        optic_flow = torch.stack(optic_flow_list, axis=0)
        optic_flow[:, :, :, 0] *= -1
        # # TODO remove this once this division error is fixed
        # optic_flow[torch.isnan(stimulus)] = 0
        # stimulus[torch.isnan(stimulus)] = 0
        # stimulus /= torch.max(stimulus)

        # Zero out entries where optic flow magnitude exceeds 1
        flow_magnitude = torch.sqrt(optic_flow[:, :, :, 0] ** 2 + optic_flow[:, :, :, 1] ** 2)
        bad_mask = flow_magnitude > 1
        optic_flow[bad_mask] = 0
        stimulus[bad_mask] = 0

        # Interpolate to model timestep
        stimulus = interpolate_to_model_dt(stimulus, stimulus_dt, model_dt)
        optic_flow = interpolate_to_model_dt(optic_flow, stimulus_dt, model_dt) * 180 / np.pi / model_dt
        
    else:
        raise ValueError(
            f"Unknown stimulus_type '{stimulus_type}'. "
            "Expected one of: static_sine, directional_sine, random_pano, random_sine, sine_const_rand, flow_360, sintel."
        )

    # --- plot the stimuli ---
    # if column_phi_theta is not None:
    #     pu.xt_plot(column_phi_theta[:, 0], column_phi_theta[:, 1], optic_flow=optic_flow.detach().cpu().numpy(), save_folder='figures/stimulus', run_tag=seed)

    # Add a blank stimulus (all zeros) with zero optic flow
    blank_stim = torch.zeros((2, stimulus.shape[1], stimulus.shape[2]), device=device, dtype=dtype)
    blank_stim[1, :, :] = 1.0
    blank_flow = torch.zeros((2, optic_flow.shape[1], optic_flow.shape[2], optic_flow.shape[3]), device=device, dtype=dtype)
    stimulus = torch.cat([blank_stim, stimulus], dim=0)
    optic_flow = torch.cat([blank_flow, optic_flow], dim=0)

    # Add pre-stimulus interval (mean-gray stimulus with zero optic flow) to the beginning of each stimulus.
    # Using 0.5 (mean of the [0,1] sine wave range) keeps the per-sample mean consistent with
    # training, so the invert_input transformation in predict() maps to the same value range.
    pre_stim_points = int(pre_stim_time / model_dt)
    if pre_stim_points > 0:
        pre_stim = torch.full((stimulus.shape[0], pre_stim_points, stimulus.shape[2]), middle_gray, device=device, dtype=dtype)
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
    through it, and returns targets matching the model's training path.
    
    Args:
        model: Neural model instance with add_noise_to_parameters and predict methods
        stimulus: Input stimulus tensor [batches, time, columns]
        input_neurons: List of input neuron types (e.g., ['L1', 'L2', 'L3'])
        output_neurons: List of output neuron types (e.g., ['T4a', 'T4b', ...])
        
    Returns:
        target: If model.use_decoder, a tensor [batches, time, columns, 2];
                otherwise a dict mapping output neuron types to firing rate tensors.
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
    
    if perturbed_model.use_decoder:
        target = pred['decoder'].clone()
    else:
        # Extract the firing rates as target dictionary
        target = {neuron: pred['firing_rate'][neuron].clone() for neuron in output_neurons}
    
    return target, perturbed_model

