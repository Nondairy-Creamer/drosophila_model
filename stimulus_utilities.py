import numpy as np
import pickle
from pathlib import Path
import torch

# All stimulus generating function should return
# Stimulus: [examples, time, points in space]
# Optic Flow: [examples, time, points in space, LR/UD optic flow]

import h5py
import numpy as np
from scipy.sparse import csr_matrix


def build_ommatidium_sampling_matrix(image_shape, image_phi_lim, image_theta_lim,
                                     sample_phi, sample_theta, filter_std=2.5):
    """
    Build a sparse matrix that samples a flattened image through Gaussian
    ommatidium filters via a single matrix-vector multiply.

    Treats (phi, theta) as flat Euclidean coordinates (no spherical correction).

    Args:
        image_shape: (H, W) of the image.
        image_phi_lim: (2,) phi limits [start, end] in radians.
        image_theta_lim: (2,) theta limits [start, end] in radians.
        sample_phi: (N,) azimuthal angles of ommatidia (radians).
        sample_theta: (N,) polar angles of ommatidia (radians).
        filter_std: Gaussian filter std in degrees (default 2.5).

    Returns:
        W: scipy.sparse.csr_matrix of shape (N, H*W).
           Usage: sampled = W @ image.ravel()
    """
    filter_std_rad = filter_std * np.pi / 180
    cutoff_rad = 2 * filter_std_rad

    H, W_img = image_shape
    N = len(sample_phi)

    theta_pixels = np.linspace(image_theta_lim[0], image_theta_lim[1], H)
    phi_pixels = np.linspace(image_phi_lim[0], image_phi_lim[1], W_img)

    d_theta_per_pixel = (image_theta_lim[1] - image_theta_lim[0]) / (H - 1)
    d_phi_per_pixel = (image_phi_lim[1] - image_phi_lim[0]) / (W_img - 1)

    cutoff_pixels_theta = int(np.ceil(cutoff_rad / np.abs(d_theta_per_pixel))) + 1
    cutoff_pixels_phi = int(np.ceil(cutoff_rad / np.abs(d_phi_per_pixel))) + 1

    rows = []
    cols = []
    vals = []

    for i in range(N):
        center_j = (sample_theta[i] - image_theta_lim[0]) / d_theta_per_pixel
        center_k = (sample_phi[i] - image_phi_lim[0]) / d_phi_per_pixel

        j_lo = max(0, int(np.floor(center_j)) - cutoff_pixels_theta)
        j_hi = min(H - 1, int(np.ceil(center_j)) + cutoff_pixels_theta)
        k_lo = max(0, int(np.floor(center_k)) - cutoff_pixels_phi)
        k_hi = min(W_img - 1, int(np.ceil(center_k)) + cutoff_pixels_phi)

        theta_patch = theta_pixels[j_lo:j_hi + 1]
        phi_patch = phi_pixels[k_lo:k_hi + 1]
        phi_grid, theta_grid = np.meshgrid(phi_patch, theta_patch)

        d_sq = (phi_grid - sample_phi[i]) ** 2 + (theta_grid - sample_theta[i]) ** 2
        weights = np.exp(-d_sq / (2 * filter_std_rad ** 2))
        weights[d_sq > cutoff_rad ** 2] = 0

        weight_sum = weights.sum()
        if weight_sum > 0:
            weights /= weight_sum

        nz_mask = weights > 0
        local_rows, local_cols = np.where(nz_mask)
        flat_indices = (local_rows + j_lo) * W_img + (local_cols + k_lo)

        n_nz = flat_indices.size
        rows.extend([i] * n_nz)
        cols.extend(flat_indices.tolist())
        vals.extend(weights[nz_mask].tolist())

    return csr_matrix((vals, (rows, cols)), shape=(N, H * W_img))


def get_drone_stimulus(model_dt, data_path='../connectome_data/drone_data.pt',
                        frame_start=0, frame_end=None):
    """
    Load preprocessed drone clips (already sampled at ommatidium locations)
    and temporally interpolate to model_dt.

    Rates are computed at the native video frame rate and then interpolated
    to model_dt, avoiding step-function artifacts from differentiating
    interpolated positions.

    Args:
        model_dt: model time step for temporal interpolation (seconds).
        data_path: path to drone_data.pt from process_drone.py.
        frame_start: first model time step to include (after interpolation).
        frame_end: last model time step to include (None = end of clip).

    Returns:
        List of dicts, one per clip, each with keys:
            'right': (time_i, N) array, right-eye stimulus in [0, 1].
            'left': (time_i, N) array, left-eye stimulus in [0, 1].
            'rotation': (time_i, 3) array, rotation position in degrees.
            'translation': (time_i, 3) array, translation position.
            'rotation_rate': (time_i, 3) array, rotation rate in deg/s.
            'translation_rate': (time_i, 3) array, translation rate.
            'video_dt': original video time step (seconds).
            'clip_id': integer clip identifier.
    """
    import torch as _torch

    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Drone data file not found: {data_path}")

    clips = _torch.load(data_path, weights_only=False)
    results = []

    for clip in clips:
        video_dt = clip['video_dt']
        T = clip['right'].shape[0]

        sampled_right = clip['right'].numpy()
        sampled_left = clip['left'].numpy()

        sampled_right = interpolate_to_model_dt(sampled_right[None], video_dt, model_dt)[0]
        sampled_left = interpolate_to_model_dt(sampled_left[None], video_dt, model_dt)[0]

        # Stored rotation/translation are already rates from extract_odometry
        rot_rate_native = clip['rotation'].numpy()
        trans_rate_native = clip['translation'].numpy()

        # Integrate rates to get cumulative positions
        from neural_models import integrate_angular_velocity, integrate_translational_velocity
        omega_t = torch.tensor(rot_rate_native, dtype=torch.float64)
        rotation = integrate_angular_velocity(omega_t, video_dt).numpy()
        translation = integrate_translational_velocity(
            torch.tensor(trans_rate_native, dtype=torch.float64),
            omega_t, video_dt,
        ).numpy()

        # Interpolate positions and rates to model_dt.
        video_times = np.arange(T) * video_dt
        num_model_steps = int(round((T - 1) * video_dt / model_dt)) + 1
        model_times = np.linspace(video_times[0], video_times[-1], num_model_steps)

        rotation_interp = np.column_stack([
            np.interp(model_times, video_times, rotation[:, i]) for i in range(3)
        ])
        translation_interp = np.column_stack([
            np.interp(model_times, video_times, translation[:, i]) for i in range(3)
        ])
        rot_rate_interp = np.column_stack([
            np.interp(model_times, video_times, rot_rate_native[:, i]) for i in range(3)
        ])
        trans_rate_interp = np.column_stack([
            np.interp(model_times, video_times, trans_rate_native[:, i]) for i in range(3)
        ])

        if 'in_fov_right' not in clip:
            raise KeyError(
                f"Clip {clip['clip_id']} missing 'in_fov_right'/'in_fov_left'. "
                "Re-run process_drone.py to regenerate drone_data.pt."
            )
        result = {
            'right': sampled_right[frame_start:frame_end],
            'left': sampled_left[frame_start:frame_end],
            'rotation': rotation_interp[frame_start:frame_end],
            'translation': translation_interp[frame_start:frame_end],
            'rotation_rate': rot_rate_interp[frame_start:frame_end],
            'translation_rate': trans_rate_interp[frame_start:frame_end],
            'video_dt': video_dt,
            'clip_id': clip['clip_id'],
            'in_fov_right': clip['in_fov_right'],
            'in_fov_left': clip['in_fov_left'],
        }
        results.append(result)

    return results


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
    num_time = int(round((pre_stim_time + stim_time + post_stim_time) / dt))
    t = np.arange(num_time) * dt
    pre_stim_index = int(round(pre_stim_time / dt))
    post_stim_index = int(round((pre_stim_time + stim_time) / dt))
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
    num_time = int(round((pre_stim_time + stim_time + post_stim_time) / dt))
    t = np.arange(num_time) * dt
    pre_stim_index = int(round(pre_stim_time / dt))
    post_stim_index = int(round((pre_stim_time + stim_time) / dt))
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

def make_counterphase_sine(sample_phi, sample_theta, contrast=1,
                           wavelength_phi=30, wavelength_theta=30,
                           temporal_frequency=4,
                           phase_phi=0, phase_theta=0, phase_t=0,
                           dt=0.01, stim_time=1.0,
                           pre_stim_time=0.0, post_stim_time=0.0):
    """
    Generate a counterphase (standing wave) sine stimulus on a sphere.

    A spatial checkerboard pattern that flickers in place without translating.
    Mathematically equivalent to the sum of two gratings drifting in opposite
    directions at the same speed.

    stimulus = sin(2π t tf + phase_t) · sin(2π φ/λ_φ + phase_φ) · sin(2π θ/λ_θ + phase_θ)

    Args:
        sample_phi: azimuthal angles (radians)
        sample_theta: polar angles (radians)
        contrast: contrast of the sine wave
        wavelength_phi: spatial wavelength in phi direction (degrees)
        wavelength_theta: spatial wavelength in theta direction (degrees)
        temporal_frequency: flicker frequency (Hz)
        phase_phi: spatial phase offset in phi (radians)
        phase_theta: spatial phase offset in theta (radians)
        phase_t: temporal phase offset (radians)
        dt: time step (seconds)
        stim_time: stimulus duration (seconds)
        pre_stim_time: pre-stimulus time (seconds)
        post_stim_time: post-stimulus time (seconds)

    Returns:
        stimulus: [time, points in space] array with values in [0, 1]
        optic_flow: [time, points in space, 2] array (zeros — no net motion)
    """
    wavelength_phi_rad = wavelength_phi * np.pi / 180
    wavelength_theta_rad = wavelength_theta * np.pi / 180
    filter_std_base = 2.5 * np.pi / 180

    num_time = int(round((pre_stim_time + stim_time + post_stim_time) / dt))
    t = np.arange(num_time) * dt
    pre_stim_index = int(round(pre_stim_time / dt))
    post_stim_index = int(round((pre_stim_time + stim_time) / dt))
    t[:pre_stim_index] = t[pre_stim_index]
    t[post_stim_index:] = t[post_stim_index - 1]

    # Contrast attenuation from ommatidia Gaussian filtering (same as LR sine)
    filter_std = filter_std_base / np.cos(sample_theta - np.pi / 2)
    contrast_scaling = np.exp(-2 * np.pi**2 * filter_std**2 / wavelength_phi_rad**2)

    spatial_phi = np.sin(2 * np.pi * sample_phi / wavelength_phi_rad + phase_phi)
    spatial_theta = np.sin(2 * np.pi * sample_theta / wavelength_theta_rad + phase_theta)
    spatial_pattern = spatial_phi * spatial_theta * contrast_scaling

    temporal = np.sin(2 * np.pi * temporal_frequency * t + phase_t)

    stimulus = contrast * temporal[:, None] * spatial_pattern[None, :]
    stimulus = (stimulus + 1.0) / 2.0

    optic_flow = np.zeros((stimulus.shape[0], stimulus.shape[1], 2))
    return stimulus, optic_flow


def create_counterphase_sine(sample_phi, sample_theta, contrast=1,
                              temporal_frequencies=None,
                              wavelengths_phi=None, wavelengths_theta=None,
                              seed=None, dt=0.01, stim_time=1.0,
                              pre_stim_time=0.0, post_stim_time=0.0):
    """
    Generate counterphase sine stimuli for all combinations of temporal
    frequency, phi wavelength, and theta wavelength, with randomized
    spatiotemporal phase for each.

    Args:
        sample_phi: azimuthal angles (radians)
        sample_theta: polar angles (radians)
        contrast: contrast of the sine wave
        temporal_frequencies: list of temporal frequencies in Hz (default [2, 4, 8])
        wavelengths_phi: list of phi wavelengths in degrees (default [30, 90])
        wavelengths_theta: list of theta wavelengths in degrees (default [30, 90])
        seed: random seed for phase randomization
        dt: time step (seconds)
        stim_time: stimulus duration (seconds)
        pre_stim_time: pre-stimulus time (seconds)
        post_stim_time: post-stimulus time (seconds)

    Returns:
        stimulus: [examples, time, points in space] array with values in [0, 1]
        optic_flow: [examples, time, points in space, 2] array (zeros)
    """
    if temporal_frequencies is None:
        temporal_frequencies = [2, 4, 8]
    if wavelengths_phi is None:
        wavelengths_phi = [30, 90]
    if wavelengths_theta is None:
        wavelengths_theta = [30, 90]

    rng = np.random.default_rng(seed)

    stim_list = []
    flow_list = []

    for tf in temporal_frequencies:
        for wl_phi in wavelengths_phi:
            for wl_theta in wavelengths_theta:
                phase_phi = rng.uniform(0, 2 * np.pi)
                phase_theta = rng.uniform(0, 2 * np.pi)
                phase_t = np.pi / 2 if tf == 0 else 0.0

                s, f = make_counterphase_sine(
                    sample_phi, sample_theta, contrast=contrast,
                    wavelength_phi=wl_phi, wavelength_theta=wl_theta,
                    temporal_frequency=tf,
                    phase_phi=phase_phi, phase_theta=phase_theta,
                    phase_t=phase_t,
                    dt=dt, stim_time=stim_time,
                    pre_stim_time=pre_stim_time,
                    post_stim_time=post_stim_time,
                )
                stim_list.append(s[None, :, :])
                flow_list.append(f[None, :, :, :])

    stimulus = np.concatenate(stim_list, axis=0)
    optic_flow = np.concatenate(flow_list, axis=0)
    return stimulus, optic_flow


def make_lr_edge(sample_phi, sample_theta, velocity, edge_polarity='light',
                 dt=0.01, stim_time=1.0, pre_stim_time=0.0, post_stim_time=0.0):
    """
    Generate a left-right moving edge stimulus on a gray background.

    The edge is a Heaviside step smoothed by the ommatidia Gaussian filter
    (2.5 deg std, scaled by 1/cos(theta-pi/2) like the sine-wave stimuli).
    A light (ON) edge transitions from gray (0.5) to bright (1.0) as it
    passes; a dark (OFF) edge transitions from gray (0.5) to dark (0.0).

    Args:
        sample_phi: azimuthal angles of columns (radians)
        sample_theta: polar angles of columns (radians)
        velocity: edge speed in degrees/s (positive = rightward)
        edge_polarity: 'light' or 'dark'
        dt: time step (seconds)
        stim_time: stimulus duration (seconds)
        pre_stim_time: pre-stimulus gray period (seconds)
        post_stim_time: post-stimulus frozen period (seconds)

    Returns:
        stimulus: (time, N) array with values in [0, 1]
        optic_flow: (time, N, 2) array (zeros)
    """
    from scipy.special import erf as sp_erf

    velocity_rad = velocity * np.pi / 180
    filter_std_base = 2.5 * np.pi / 180

    num_time = int(round((pre_stim_time + stim_time + post_stim_time) / dt))
    t = np.arange(num_time) * dt
    pre_stim_index = int(round(pre_stim_time / dt))
    post_stim_index = int(round((pre_stim_time + stim_time) / dt))
    t[:pre_stim_index] = t[pre_stim_index]
    t[post_stim_index:] = t[post_stim_index - 1]

    filter_std = filter_std_base / np.cos(sample_theta - np.pi / 2)  # (N,)

    margin = 5 * filter_std_base
    if velocity_rad >= 0:
        start_pos = np.max(sample_phi) + margin
    else:
        start_pos = np.min(sample_phi) - margin

    # Rightward motion = decreasing phi in the fly coordinate system
    edge_pos = start_pos - velocity_rad * (t - t[pre_stim_index])  # (T,)

    sign = 1.0 if velocity >= 0 else -1.0
    diff = sign * (sample_phi[None, :] - edge_pos[:, None])  # (T, N)
    H_smooth = 0.5 * (1.0 + sp_erf(diff / (filter_std[None, :] * np.sqrt(2))))

    if edge_polarity == 'light':
        stimulus = 0.5 + 0.5 * H_smooth
    elif edge_polarity == 'dark':
        stimulus = 0.5 - 0.5 * H_smooth
    else:
        raise ValueError(f"Unknown edge_polarity: {edge_polarity}. Use 'light' or 'dark'.")

    optic_flow = np.zeros((stimulus.shape[0], stimulus.shape[1], 2))
    return stimulus, optic_flow


def make_ud_edge(sample_phi, sample_theta, velocity, edge_polarity='light',
                 dt=0.01, stim_time=1.0, pre_stim_time=0.0, post_stim_time=0.0):
    """
    Generate an up-down moving edge stimulus on a gray background.

    The edge sweeps the visual field via rotation around the x-axis (forward)
    at constant angular velocity.  Each column transitions once — from gray
    to bright (light) or gray to dark (dark) — and stays at the new value.
    The sweep speed varies with position: faster for lateral-looking columns
    (near ±y) and slower near the x-axis poles.

    Args:
        sample_phi: azimuthal angles of columns (radians)
        sample_theta: polar angles of columns (radians)
        velocity: edge speed in degrees/s (positive = upward)
        edge_polarity: 'light' or 'dark'
        dt: time step (seconds)
        stim_time: stimulus duration (seconds)
        pre_stim_time: pre-stimulus gray period (seconds)
        post_stim_time: post-stimulus frozen period (seconds)

    Returns:
        stimulus: (time, N) array with values in [0, 1]
        optic_flow: (time, N, 2) array (zeros)
    """
    from scipy.special import erf as sp_erf

    velocity_rad = velocity * np.pi / 180
    filter_std_base = 2.5 * np.pi / 180

    num_time = int(round((pre_stim_time + stim_time + post_stim_time) / dt))
    t = np.arange(num_time) * dt
    pre_stim_index = int(round(pre_stim_time / dt))
    post_stim_index = int(round((pre_stim_time + stim_time) / dt))
    t[:pre_stim_index] = t[pre_stim_index]
    t[post_stim_index:] = t[post_stim_index - 1]

    # Column positions in Cartesian (x = forward, y = left, z = up)
    col_y = np.sin(sample_theta) * np.sin(sample_phi)  # (N,)
    col_z = np.cos(sample_theta)                        # (N,)

    # Rotation angle around x-axis (roll).
    # The edge normal is n = Rx(alpha) · z_hat = (0, -sin(alpha), cos(alpha)).
    # For right-eye columns (phi < 0 → col_y < 0), alpha decreasing from
    # pi/2 sweeps the edge upward (bottom columns transition first).
    margin = 5 * filter_std_base
    if velocity_rad >= 0:  # upward: edge starts below FOV
        alpha_start = np.pi / 2 + margin
    else:  # downward: edge starts above FOV
        alpha_start = -np.pi / 2 - margin

    alpha = alpha_start - velocity_rad * (t - t[pre_stim_index])  # (T,)

    # Signed distance from the edge great circle.
    # d = dot(column, n) where n = Rx(alpha) · z_hat = (0, -sin(alpha), cos(alpha))
    # d > 0 means the column is on the +n side of the edge.
    d = (-col_y[None, :] * np.sin(alpha[:, None])
         + col_z[None, :] * np.cos(alpha[:, None]))  # (T, N)

    # Convention: "swept" (behind the edge) is the side that has already
    # transitioned.  For upward sweep the swept side is below (d < 0 at
    # alpha ~ pi/2), for downward it is above (d > 0 at alpha ~ -pi/2).
    sign = 1.0 if velocity >= 0 else -1.0
    swept = -sign * d  # positive = swept

    # Running max ensures each column stays at its peak transition value.
    # Without this, the great circle wraps around and a trailing edge
    # would reverse the transition.
    swept = np.maximum.accumulate(swept, axis=0)

    H_smooth = 0.5 * (1.0 + sp_erf(swept / (filter_std_base * np.sqrt(2))))

    if edge_polarity == 'light':
        stimulus = 0.5 + 0.5 * H_smooth
    elif edge_polarity == 'dark':
        stimulus = 0.5 - 0.5 * H_smooth
    else:
        raise ValueError(f"Unknown edge_polarity: {edge_polarity}. Use 'light' or 'dark'.")

    optic_flow = np.zeros((stimulus.shape[0], stimulus.shape[1], 2))
    return stimulus, optic_flow


def make_full_field_flicker(num_points, temporal_frequency, dt, stim_time,
                            pre_stim_time=0.0, post_stim_time=0.0):
    """
    Generate a full-field flicker stimulus where luminance at every spatial
    point oscillates sinusoidally at a given temporal frequency.

    Args:
        num_points: number of spatial points (columns).
        temporal_frequency: flicker frequency in Hz.
        dt: time step in seconds.
        stim_time: stimulus duration in seconds.
        pre_stim_time: pre-stimulus hold at mean gray (seconds).
        post_stim_time: post-stimulus hold at final value (seconds).

    Returns:
        stimulus: (time, num_points) array with values in [0, 1].
        optic_flow: (time, num_points, 2) array of zeros.
    """
    total_time = pre_stim_time + stim_time + post_stim_time
    num_time = int(round(total_time / dt))
    t = np.arange(num_time) * dt
    pre_idx = int(round(pre_stim_time / dt))
    post_idx = int(round((pre_stim_time + stim_time) / dt))

    t_eff = t.copy()
    t_eff[:pre_idx] = t_eff[pre_idx]
    t_eff[post_idx:] = t_eff[post_idx - 1]
    t_eff -= t_eff[pre_idx]

    luminance = 0.5 + 0.5 * np.sin(2 * np.pi * temporal_frequency * t_eff)

    stimulus = np.broadcast_to(luminance[:, None], (len(t), num_points)).copy()
    optic_flow = np.zeros((len(t), num_points, 2))

    return stimulus, optic_flow


def create_edge_stimulus(sample_phi, sample_theta, velocities=None,
                         dt=0.01, stim_time=1.0, pre_stim_time=0.0, post_stim_time=0.0):
    """
    Generate edge stimuli for all four directions and both polarities.

    For each velocity, creates light and dark edges moving left, right,
    up, and down (4 directions x 2 polarities per velocity).

    Args:
        sample_phi: azimuthal angles (radians)
        sample_theta: polar angles (radians)
        velocities: list of edge speeds in deg/s (default [30, 60, 120, 240])
        dt: time step (seconds)
        stim_time: stimulus duration (seconds)
        pre_stim_time: pre-stimulus time (seconds)
        post_stim_time: post-stimulus time (seconds)

    Returns:
        stimulus: (examples, time, N) array with values in [0, 1]
        optic_flow: (examples, time, N, 2) array (zeros)
    """
    if velocities is None:
        velocities = [30, 60, 120, 240]
    if not isinstance(velocities, (list, np.ndarray)):
        velocities = [velocities]

    blocks = ['right', 'left', 'up', 'down']
    stim_dict = {b: [] for b in blocks}
    flow_dict = {b: [] for b in blocks}

    for velocity in velocities:
        for polarity in ['light', 'dark']:
            edge_r, flow_r = make_lr_edge(
                sample_phi, sample_theta, velocity=velocity,
                edge_polarity=polarity, dt=dt, stim_time=stim_time,
                pre_stim_time=pre_stim_time, post_stim_time=post_stim_time)
            stim_dict['right'].append(edge_r[None, :, :])
            flow_dict['right'].append(flow_r[None, :, :, :])

            edge_l, flow_l = make_lr_edge(
                sample_phi, sample_theta, velocity=-velocity,
                edge_polarity=polarity, dt=dt, stim_time=stim_time,
                pre_stim_time=pre_stim_time, post_stim_time=post_stim_time)
            stim_dict['left'].append(edge_l[None, :, :])
            flow_dict['left'].append(flow_l[None, :, :, :])

            edge_u, flow_u = make_ud_edge(
                sample_phi, sample_theta, velocity=velocity,
                edge_polarity=polarity, dt=dt, stim_time=stim_time,
                pre_stim_time=pre_stim_time, post_stim_time=post_stim_time)
            stim_dict['up'].append(edge_u[None, :, :])
            flow_dict['up'].append(flow_u[None, :, :, :])

            edge_d, flow_d = make_ud_edge(
                sample_phi, sample_theta, velocity=-velocity,
                edge_polarity=polarity, dt=dt, stim_time=stim_time,
                pre_stim_time=pre_stim_time, post_stim_time=post_stim_time)
            stim_dict['down'].append(edge_d[None, :, :])
            flow_dict['down'].append(flow_d[None, :, :, :])

    stimulus, optic_flow = consolidate_stim(stim_dict, flow_dict)
    return stimulus, optic_flow


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
    num_time = int(round(stim_time * num_repeats / dt))
    time_vect = np.arange(num_time) * dt
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


def _euler_to_rotation_matrix(roll_deg, pitch_deg, yaw_deg):
    """Convert extrinsic XYZ Euler angles (degrees) to a 3x3 rotation matrix."""
    r = np.radians(roll_deg)
    p = np.radians(pitch_deg)
    y = np.radians(yaw_deg)
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)

    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [  -sp,           cp*sr,           cp*cr],
    ])
    return R


def _euler_angles_to_angular_velocity(euler_angles, dt):
    """Convert a time series of Euler angles (degrees) to angular velocity (deg/s).

    Computes body-frame angular velocity via the relative rotation matrix
    between consecutive frames, avoiding gimbal-lock singularities that
    corrupt naive ``np.diff(euler) / dt`` when pitch is near ±90°.

    Uses ``R_rel = R_prev^T @ R_curr`` so the extracted axis-angle is in
    the body frame, consistent with ``get_random_velocity_pano`` which
    returns ``omega_body = R^T @ omega_world``.

    Args:
        euler_angles: (T, 3) array of [roll, pitch, yaw] in degrees.
        dt: time step in seconds.

    Returns:
        omega: (T, 3) angular velocity in deg/s (body frame).
               First frame is set to zero.
    """
    T = euler_angles.shape[0]
    omega = np.zeros_like(euler_angles)

    R_prev = _euler_to_rotation_matrix(*euler_angles[0])
    for t in range(1, T):
        R_curr = _euler_to_rotation_matrix(*euler_angles[t])
        # Body-frame relative rotation: R_prev^T @ R_curr
        R_rel = R_prev.T @ R_curr
        # Extract axis-angle via matrix logarithm (Rodrigues)
        cos_angle = (np.trace(R_rel) - 1.0) / 2.0
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        if angle < 1e-10:
            omega[t] = 0.0
        else:
            # Axis from skew-symmetric part of R_rel (body frame)
            axis = np.array([
                R_rel[2, 1] - R_rel[1, 2],
                R_rel[0, 2] - R_rel[2, 0],
                R_rel[1, 0] - R_rel[0, 1],
            ]) / (2.0 * np.sin(angle))
            omega[t] = axis * angle * (180.0 / np.pi) / dt
        R_prev = R_curr

    return omega


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
    aliasing.  Output is normalized to [0, 1].

    Args:
        pano: (H, W) raw panoramic image array.
        filter_std_rad: Gaussian sigma in radians (default 2.5 deg).

    Returns:
        pano_out: (H, W) float64 array with values in [0, 1].
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

    pano_min = pano_out.min()
    pano_max = pano_out.max()
    if pano_max > pano_min:
        pano_out = (pano_out - pano_min) / (pano_max - pano_min)
    else:
        pano_out = np.zeros_like(pano_out)

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
        oob_mask: [examples, time, points in space] — True where viewing direction
            falls outside panorama bounds (above/below image).
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
    time_per_stim = int(round(stim_time / dt))
    num_time = time_per_stim * num_repeats_total
    time_vect = np.arange(num_time) * dt

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
    oob_list = []

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
            oob_list.append(out_of_bounds)

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
    oob_mask = np.stack(oob_list)                  # (num_panos * num_repeats_total, time_per_stim, num_cells)

    return stimulus, optic_flow, rotation_vectors, oob_mask


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

def get_stimulus_one_eye(
    stimulus_type: str | list[str],
    config,
    column_phi_theta=None,
    *,
    max_duration_s: float | None = None,
    use_full_length: bool = False,
    sine_direction: str | None = None,
    sine_temporal_frequency: float | None = None,
    random_sine_duration_s: float | None = None,
    pre_stim_time_override: float | None = None,
):
    device = torch.device(config['model']['device'])
    dtype = getattr(torch, config['model']['dtype'])
    middle_gray = 0.5

    model_dt = config['model']['dt']
    stim_time = config['stimulus']['stim_time']
    pre_stim_time = config['stimulus']['pre_stim_time']
    post_stim_time = config['stimulus']['post_stim_time']
    temporal_frequencies = config['stimulus']['temporal_frequencies']
    stimulus_seed = config['stimulus']['seed']
    if pre_stim_time_override is not None:
        pre_stim_time = float(pre_stim_time_override)
    sine_pre_stim_time = pre_stim_time
    sine_post_stim_time = post_stim_time

    stimulus_types = [stimulus_type] if isinstance(stimulus_type, str) else stimulus_type
    stim_list, flow_list, mask_list = [], [], []

    for stim_type in stimulus_types:
        if stim_type == 'static_sine':
            if column_phi_theta is None:
                raise Exception('sine waves require column_phi_theta input')
            stimulus, optic_flow = create_balanced_sine_wave(
                column_phi_theta[:, 0], column_phi_theta[:, 1],
                temporal_frequency=temporal_frequencies, dt=model_dt, stim_time=stim_time, pre_stim_time=0.0, post_stim_time=0.0
            )
            stimulus = torch.tensor(stimulus, device=device, dtype=dtype)
            optic_flow = torch.tensor(optic_flow, device=device, dtype=dtype)
            train_mask = torch.ones(stimulus.shape[:3], dtype=torch.bool, device=device)

        elif stim_type == 'directional_sine':
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
            train_mask = torch.ones(stimulus.shape[:3], dtype=torch.bool, device=device)

        elif stim_type == 'edge':
            if column_phi_theta is None:
                raise Exception('edge stimulus requires column_phi_theta input')
            edge_velocities = config['stimulus'].get('edge_velocities', [30, 60, 120, 240])
            stimulus, optic_flow = create_edge_stimulus(
                column_phi_theta[:, 0], column_phi_theta[:, 1],
                velocities=edge_velocities, dt=model_dt, stim_time=stim_time,
                pre_stim_time=pre_stim_time, post_stim_time=post_stim_time,
            )
            stimulus = torch.tensor(stimulus, device=device, dtype=dtype)
            optic_flow = torch.tensor(optic_flow, device=device, dtype=dtype)
            train_mask = torch.ones(stimulus.shape[:3], dtype=torch.bool, device=device)

        elif stim_type == 'random_pano':
            if column_phi_theta is None:
                raise Exception('panoramic require column_phi_theta input')
            stimulus, optic_flow, _, _ = get_random_velocity_pano(
                column_phi_theta[:, 0], column_phi_theta[:, 1],
                velocity_amp=config['stimulus']['velocity_amp'], velocity_time_scale=config['stimulus']['velocity_time_scale'],
                num_repeats=config['stimulus']['num_repeats'], dt=model_dt, stim_time=stim_time, seed=stimulus_seed
            )
            stimulus = torch.tensor(stimulus, device=device, dtype=dtype)
            optic_flow = torch.tensor(optic_flow, device=device, dtype=dtype)
            train_mask = torch.ones(stimulus.shape[:3], dtype=torch.bool, device=device)

        elif stim_type == 'random_sine':
            if column_phi_theta is None:
                raise Exception('sine waves require column_phi_theta input')
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
            train_mask = torch.ones(stimulus.shape[:3], dtype=torch.bool, device=device)

        elif stim_type == 'flow_360':
            default_length = 98
            stimulus_path = Path('../connectome_data/flow360_data_right_eye_mapped.pkl')
            with open(stimulus_path, 'rb') as f:
                stimulus_in = pickle.load(f)

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

            stimulus = interpolate_to_model_dt(stimulus, stimulus_dt, model_dt)
            optic_flow = interpolate_to_model_dt(optic_flow, stimulus_dt, model_dt)

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

            num_stimuli, time_length, num_neurons = stimulus.shape
            num_segments = time_length // final_length
            if num_segments > 0:
                stimulus = stimulus[:, :num_segments*final_length, :].reshape(num_stimuli * num_segments, final_length, num_neurons)
                optic_flow = optic_flow[:, :num_segments*final_length, :, :].reshape(num_stimuli * num_segments, final_length, num_neurons, optic_flow.shape[3])
            train_mask = torch.ones(stimulus.shape[:3], dtype=torch.bool, device=device)

        elif stim_type == 'sintel':
            default_length = 100
            sintel_path = Path('../connectome_data/sintel_data.pkl')
            with open(sintel_path, 'rb') as f:
                stimulus_in = pickle.load(f)

            expected_length = 49
            stimulus_dt = 1 / stimulus_in['framerate']
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

            stimulus = interpolate_to_model_dt(stimulus, stimulus_dt, model_dt)
            optic_flow = interpolate_to_model_dt(optic_flow, stimulus_dt, model_dt)

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

            num_stimuli, time_length, num_neurons = stimulus.shape
            num_segments = time_length // final_length
            if num_segments > 0:
                stimulus = stimulus[:, :num_segments*final_length, :].reshape(num_stimuli * num_segments, final_length, num_neurons)
                optic_flow = optic_flow[:, :num_segments*final_length, :, :].reshape(num_stimuli * num_segments, final_length, num_neurons, optic_flow.shape[3])
            train_mask = torch.ones(stimulus.shape[:3], dtype=torch.bool, device=device)

        elif stim_type == 'flyview':
            required_length = 80
            sintel_path = Path('../connectome_data/rendered_fly_view_dataset.pkl')
            with open(sintel_path, 'rb') as f:
                stimulus_in = pickle.load(f)

            stimulus_dt = 1 / stimulus_in['fps']
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

            flow_magnitude = torch.sqrt(optic_flow[:, :, :, 0] ** 2 + optic_flow[:, :, :, 1] ** 2)
            bad_mask = flow_magnitude > 1
            optic_flow[bad_mask] = 0
            stimulus[bad_mask] = 0

            stimulus = interpolate_to_model_dt(stimulus, stimulus_dt, model_dt)
            optic_flow = interpolate_to_model_dt(optic_flow, stimulus_dt, model_dt) * 180 / np.pi / model_dt
            train_mask = torch.ones(stimulus.shape[:3], dtype=torch.bool, device=device)

        elif stim_type == 'counterphase_sine':
            if column_phi_theta is None:
                raise Exception('counterphase_sine requires column_phi_theta input')
            stimulus, optic_flow = create_counterphase_sine(
                column_phi_theta[:, 0], column_phi_theta[:, 1],
                temporal_frequencies=temporal_frequencies,
                seed=stimulus_seed, dt=model_dt, stim_time=stim_time,
                pre_stim_time=0.0, post_stim_time=0.0,
            )
            stimulus = torch.tensor(stimulus, device=device, dtype=dtype)
            optic_flow = torch.tensor(optic_flow, device=device, dtype=dtype)
            train_mask = torch.ones(stimulus.shape[:3], dtype=torch.bool, device=device)

        elif stim_type == 'full_field_step':
            if column_phi_theta is None:
                raise Exception('full_field_step requires column_phi_theta input')
            num_time_steps = int(round(stim_time / model_dt))
            n_cols = column_phi_theta.shape[0]
            stimulus = torch.zeros((3, num_time_steps, n_cols), device=device, dtype=dtype)
            stimulus[0] = 0.0
            stimulus[1] = 0.5
            stimulus[2] = 1.0
            optic_flow = torch.zeros((3, num_time_steps, n_cols, 2), device=device, dtype=dtype)
            train_mask = torch.ones((3, num_time_steps, n_cols), dtype=torch.bool, device=device)

        elif stim_type == 'full_field_flicker':
            if column_phi_theta is None:
                raise Exception('full_field_flicker requires column_phi_theta input')
            n_cols = column_phi_theta.shape[0]
            flicker_tfs = config['stimulus'].get('flicker_temporal_frequencies', temporal_frequencies)
            if not isinstance(flicker_tfs, list):
                flicker_tfs = [flicker_tfs]
            stim_arrays, flow_arrays = [], []
            for tf in flicker_tfs:
                s, f = make_full_field_flicker(n_cols, tf, model_dt, stim_time)
                stim_arrays.append(s[None])
                flow_arrays.append(f[None])
            stimulus = torch.tensor(np.concatenate(stim_arrays), device=device, dtype=dtype)
            optic_flow = torch.tensor(np.concatenate(flow_arrays), device=device, dtype=dtype)
            train_mask = torch.ones(stimulus.shape[:3], dtype=torch.bool, device=device)

        else:
            raise ValueError(
                f"Unknown stimulus_type '{stim_type}'. "
                "Expected one of: static_sine, directional_sine, edge, random_pano, "
                "random_sine, flow_360, sintel, flyview, full_field_step, "
                "full_field_flicker, counterphase_sine."
            )

        stim_list.append(stimulus)
        flow_list.append(optic_flow)
        mask_list.append(train_mask)

    stimulus = torch.cat(stim_list, dim=0)
    optic_flow = torch.cat(flow_list, dim=0)
    train_mask = torch.cat(mask_list, dim=0)

    num_time = stimulus.shape[1]
    num_cols = stimulus.shape[2]

    # Add pre-stimulus interval (mean-gray stimulus with zero optic flow)
    pre_stim_points = int(round(pre_stim_time / model_dt))
    if pre_stim_points > 0:
        n_examples = stimulus.shape[0]
        pre_stim = torch.full((n_examples, pre_stim_points, num_cols), middle_gray, device=device, dtype=dtype)
        pre_flow = torch.zeros((n_examples, pre_stim_points, optic_flow.shape[2], optic_flow.shape[3]), device=device, dtype=dtype)
        pre_mask = torch.zeros((n_examples, pre_stim_points, num_cols), dtype=torch.bool, device=device)
        stimulus = torch.cat([pre_stim, stimulus], dim=1)
        optic_flow = torch.cat([pre_flow, optic_flow], dim=1)
        train_mask = torch.cat([pre_mask, train_mask], dim=1)

    return stimulus, optic_flow, train_mask


def get_stimulus_two_eyes(
    stimulus_type: str | list[str],
    config,
    column_phi_theta=None,
    *,
    pre_stim_time_override: float | None = None,
):
    """
    Load a two-eye stimulus and return sampled images, optic flow,
    and observer ego-motion.

    Args:
        stimulus_type: type of stimulus, or list of types to concatenate.
        config: configuration dict.
        column_phi_theta: (N, 2) right-eye (phi, theta) in radians.
            Left eye is sampled at (-phi, theta).
        pre_stim_time_override: override pre-stim time from config.
    Returns:
        stimulus: (examples, time, 2*N) right then left eye stimulus.
        optic_flow: (examples, time, 2*N, 2) right then left optic flow (NaN if unavailable).
        observer_rotation: (examples, time, 3) rotation rate (deg/s).
        observer_translation: (examples, time, 3) translation rate.
        train_mask: (examples, time, 2*N) bool — True where decoder output
            should be used for training/odometry (False for pre-stim,
            out-of-bounds columns, etc.).
    """
    device = torch.device(config['model']['device'])
    dtype = getattr(torch, config['model']['dtype'])
    middle_gray = 0.5

    model_dt = config['model']['dt']
    pre_stim_time = config['stimulus']['pre_stim_time']
    if pre_stim_time_override is not None:
        pre_stim_time = float(pre_stim_time_override)

    stimulus_types = [stimulus_type] if isinstance(stimulus_type, str) else stimulus_type
    stim_list, flow_list, rot_list, trans_list, mask_list = [], [], [], [], []

    for stim_type in stimulus_types:
        if stim_type == 'random_pano':
            if column_phi_theta is None:
                raise Exception('random_pano requires column_phi_theta input')

            stim_time = config['stimulus']['stim_time']
            stimulus_seed = config['stimulus'].get('seed', None)
            N = column_phi_theta.shape[0]

            both_phi = np.concatenate([column_phi_theta[:, 0], -column_phi_theta[:, 0]])
            both_theta = np.concatenate([column_phi_theta[:, 1], column_phi_theta[:, 1]])

            stimulus_both, optic_flow_both, rotation_vectors, _ = get_random_velocity_pano(
                both_phi, both_theta,
                velocity_amp=config['stimulus']['velocity_amp'],
                velocity_time_scale=config['stimulus']['velocity_time_scale'],
                num_repeats=config['stimulus']['num_repeats'],
                dt=model_dt, stim_time=stim_time, seed=stimulus_seed,
            )

            stimulus = torch.tensor(stimulus_both, device=device, dtype=dtype)
            optic_flow = torch.tensor(optic_flow_both, device=device, dtype=dtype)
            # NOTE: no u_right sign flip needed for the left eye.
            # generate_optic_flow returns u_right = -(u · e_phi), which already
            # gives the same sign for symmetric (phi, -phi) columns because both
            # the 3D flow and the e_phi basis vector mirror together.  The model's
            # lr_flip in estimate_ego_motion_from_image handles the decoder's
            # mirror convention separately.
            observer_rotation = torch.tensor(
                rotation_vectors * 180 / np.pi,  # rad/s → deg/s
                device=device, dtype=dtype,
            )
            observer_translation = torch.zeros_like(observer_rotation)
            train_mask = torch.ones(stimulus.shape[:3], dtype=torch.bool, device=device)

        elif stim_type == 'drone_view':
            if column_phi_theta is None:
                raise Exception('drone_view requires column_phi_theta input')

            stim_time = config['stimulus']['stim_time']
            drone_data_path = Path(config['data']['connectome_folder']) / 'drone_data.pt'
            drone_clips = get_drone_stimulus(
                model_dt, data_path=str(drone_data_path),
            )

            # FOV mask from the first clip (same geometry for all clips)
            in_fov = torch.cat([
                drone_clips[0]['in_fov_right'],
                drone_clips[0]['in_fov_left'],
            ], dim=0)  # (2*N,) True = in field of view

            seg_len = int(round(stim_time / model_dt))
            max_clip_len = max(clip['right'].shape[0] for clip in drone_clips)
            if seg_len > max_clip_len:
                import warnings
                warnings.warn(
                    f"stim_time={stim_time}s ({seg_len} steps) exceeds longest "
                    f"drone clip ({max_clip_len} steps = {max_clip_len * model_dt:.1f}s). "
                    f"Capping to {max_clip_len * model_dt:.1f}s."
                )
                seg_len = max_clip_len

            seg_rights, seg_lefts, seg_rots, seg_trans = [], [], [], []
            n_drone_segments = 0
            for clip in drone_clips:
                T_clip = clip['right'].shape[0]
                num_segments = T_clip // seg_len
                if num_segments == 0:
                    continue
                usable = num_segments * seg_len
                n_drone_segments += num_segments

                seg_rights.append(clip['right'][:usable].reshape(num_segments, seg_len, -1))
                seg_lefts.append(clip['left'][:usable].reshape(num_segments, seg_len, -1))
                seg_rots.append(clip['rotation_rate'][:usable].reshape(num_segments, seg_len, 3))
                seg_trans.append(clip['translation_rate'][:usable].reshape(num_segments, seg_len, 3))

            stim_right = np.concatenate(seg_rights, axis=0)
            stim_left = np.concatenate(seg_lefts, axis=0)
            true_rotation = np.concatenate(seg_rots, axis=0)
            true_translation = np.concatenate(seg_trans, axis=0)

            stimulus = torch.tensor(
                np.concatenate([stim_right, stim_left], axis=-1), device=device, dtype=dtype,
            )
            optic_flow = torch.full((*stimulus.shape, 2), float('nan'), device=device, dtype=dtype)
            observer_rotation = torch.tensor(true_rotation, device=device, dtype=dtype)
            observer_translation = torch.tensor(true_translation, device=device, dtype=dtype)

            train_mask = in_fov.unsqueeze(0).unsqueeze(0).expand(
                n_drone_segments, seg_len, -1).to(device=device)

        elif stim_type == 'edge':
            if column_phi_theta is None:
                raise Exception('edge stimulus requires column_phi_theta input')

            stim_time = config['stimulus']['stim_time']
            edge_velocities = config['stimulus'].get('edge_velocities', [30, 60, 120, 240])

            both_phi = np.concatenate([column_phi_theta[:, 0], -column_phi_theta[:, 0]])
            both_theta = np.concatenate([column_phi_theta[:, 1], column_phi_theta[:, 1]])

            stimulus_np, optic_flow_np = create_edge_stimulus(
                both_phi, both_theta,
                velocities=edge_velocities, dt=model_dt, stim_time=stim_time,
                pre_stim_time=0.0, post_stim_time=0.0,
            )
            stimulus = torch.tensor(stimulus_np, device=device, dtype=dtype)
            optic_flow = torch.tensor(optic_flow_np, device=device, dtype=dtype)
            observer_rotation = torch.zeros(
                (stimulus.shape[0], stimulus.shape[1], 3), device=device, dtype=dtype)
            observer_translation = torch.zeros_like(observer_rotation)
            train_mask = torch.ones(stimulus.shape[:3], dtype=torch.bool, device=device)

        elif stim_type == 'counterphase_sine':
            if column_phi_theta is None:
                raise Exception('counterphase_sine requires column_phi_theta input')
            stim_time = config['stimulus']['stim_time']
            stimulus_seed = config['stimulus'].get('seed', None)
            temporal_frequencies = config['stimulus'].get('temporal_frequencies', [0, 2, 4, 8])

            both_phi = np.concatenate([column_phi_theta[:, 0], -column_phi_theta[:, 0]])
            both_theta = np.concatenate([column_phi_theta[:, 1], column_phi_theta[:, 1]])

            stimulus_np, optic_flow_np = create_counterphase_sine(
                both_phi, both_theta,
                temporal_frequencies=temporal_frequencies,
                seed=stimulus_seed, dt=model_dt, stim_time=stim_time,
                pre_stim_time=0.0, post_stim_time=0.0,
            )
            stimulus = torch.tensor(stimulus_np, device=device, dtype=dtype)
            optic_flow = torch.tensor(optic_flow_np, device=device, dtype=dtype)
            n_cp = stimulus.shape[0]
            observer_rotation = torch.zeros((n_cp, stimulus.shape[1], 3), device=device, dtype=dtype)
            observer_translation = torch.zeros((n_cp, stimulus.shape[1], 3), device=device, dtype=dtype)
            train_mask = torch.ones(stimulus.shape[:3], dtype=torch.bool, device=device)

        elif stim_type == 'full_field_step':
            if column_phi_theta is None:
                raise Exception('full_field_step requires column_phi_theta input')
            stim_time = config['stimulus']['stim_time']
            num_time_steps = int(round(stim_time / model_dt))
            num_cols_both = 2 * column_phi_theta.shape[0]
            stimulus = torch.zeros((3, num_time_steps, num_cols_both), device=device, dtype=dtype)
            stimulus[0] = 0.0
            stimulus[1] = 0.5
            stimulus[2] = 1.0
            optic_flow = torch.zeros((3, num_time_steps, num_cols_both, 2), device=device, dtype=dtype)
            observer_rotation = torch.zeros((3, num_time_steps, 3), device=device, dtype=dtype)
            observer_translation = torch.zeros((3, num_time_steps, 3), device=device, dtype=dtype)
            train_mask = torch.ones((3, num_time_steps, num_cols_both), dtype=torch.bool, device=device)

        elif stim_type == 'full_field_flicker':
            if column_phi_theta is None:
                raise Exception('full_field_flicker requires column_phi_theta input')
            stim_time = config['stimulus']['stim_time']
            num_cols_both = 2 * column_phi_theta.shape[0]
            flicker_tfs = config['stimulus'].get('flicker_temporal_frequencies', [1.0])
            if not isinstance(flicker_tfs, list):
                flicker_tfs = [flicker_tfs]
            stim_arrays, flow_arrays = [], []
            for tf in flicker_tfs:
                s, f = make_full_field_flicker(num_cols_both, tf, model_dt, stim_time)
                stim_arrays.append(s[None])
                flow_arrays.append(f[None])
            n_flicker = len(flicker_tfs)
            stimulus = torch.tensor(np.concatenate(stim_arrays), device=device, dtype=dtype)
            optic_flow = torch.tensor(np.concatenate(flow_arrays), device=device, dtype=dtype)
            observer_rotation = torch.zeros((n_flicker, stimulus.shape[1], 3), device=device, dtype=dtype)
            observer_translation = torch.zeros((n_flicker, stimulus.shape[1], 3), device=device, dtype=dtype)
            train_mask = torch.ones(stimulus.shape[:3], dtype=torch.bool, device=device)

        else:
            raise ValueError(
                f"Unknown stimulus_type '{stim_type}' for two-eye stimulus. "
                "Expected one of: random_pano, drone_view, edge, full_field_step, "
                "full_field_flicker, counterphase_sine."
            )

        stim_list.append(stimulus)
        flow_list.append(optic_flow)
        rot_list.append(observer_rotation)
        trans_list.append(observer_translation)
        mask_list.append(train_mask)

    stimulus = torch.cat(stim_list, dim=0)
    optic_flow = torch.cat(flow_list, dim=0)
    observer_rotation = torch.cat(rot_list, dim=0)
    observer_translation = torch.cat(trans_list, dim=0)
    train_mask = torch.cat(mask_list, dim=0)

    num_time = stimulus.shape[1]
    num_cols = stimulus.shape[2]   # 2*N
    num_flow_cols = optic_flow.shape[2]

    # Add pre-stimulus interval (stationary observer, middle-gray stimulus)
    pre_stim_points = int(round(pre_stim_time / model_dt))
    if pre_stim_points > 0:
        n_examples = stimulus.shape[0]
        pre_stim = torch.full((n_examples, pre_stim_points, num_cols), middle_gray, device=device, dtype=dtype)
        pre_flow = torch.zeros((n_examples, pre_stim_points, num_flow_cols, 2), device=device, dtype=dtype)
        pre_rot = torch.zeros((n_examples, pre_stim_points, 3), device=device, dtype=dtype)
        pre_trans = torch.zeros((n_examples, pre_stim_points, 3), device=device, dtype=dtype)
        pre_mask = torch.zeros((n_examples, pre_stim_points, num_cols), dtype=torch.bool, device=device)

        stimulus = torch.cat([pre_stim, stimulus], dim=1)
        optic_flow = torch.cat([pre_flow, optic_flow], dim=1)
        observer_rotation = torch.cat([pre_rot, observer_rotation], dim=1)
        observer_translation = torch.cat([pre_trans, observer_translation], dim=1)
        train_mask = torch.cat([pre_mask, train_mask], dim=1)

    return stimulus, optic_flow, observer_rotation, observer_translation, train_mask


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
    new_time = int(round(total_duration / model_dt)) + 1
    
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
        target: A dict. If the model has flow readouts, includes
                'optic_flow' from the first flow readout; otherwise maps
                output neuron types to firing rate tensors.
    """
    import copy

    # Create a deep copy of the model to avoid modifying the original
    perturbed_model = copy.deepcopy(model)
    
    # Add noise to the copied model's parameters
    perturbed_model.add_noise_to_parameters(model.init_std * init_noise_mult)
    
    # Run the stimulus through the perturbed model
    with torch.no_grad():
        pred = perturbed_model.predict(
            stimulus,
            input_neurons=input_neurons,
            output_neurons=output_neurons,
            init_state=None
        )['outputs']
    
    flow_key = next(
        (k for k, m in perturbed_model._readout_modules.items() if m.predicts_flow and k in pred),
        None,
    )
    if flow_key is not None:
        target = {'optic_flow': pred[flow_key].clone()}
    else:
        target = {neuron: pred['firing_rate'][neuron].clone() for neuron in output_neurons}
    
    return target, perturbed_model

