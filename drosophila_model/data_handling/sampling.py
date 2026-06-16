"""Low-level spatial/temporal sampling primitives for stimuli.

These helpers are intentionally dependency-free with respect to the rest of
``data_handling``: both ``stimulus_generators.py`` (synthetic generators) and
``drone_utilities.py`` (raw drone-rig processing) import from here, so this
module must not import from either of them. Keeping the shared primitives in
one neutral place lets the dependency flow one way
(generators/utilities -> sampling) without import cycles.

Contents:
    * ``build_ommatidium_sampling_matrix`` / ``build_calibrated_sampling_matrix``
      -- sparse (image pixel -> ommatidium) samplers.
    * ``interpolate_to_model_dt`` / ``resample_time_to_length`` -- temporal
      resampling along the time axis.
"""

import numpy as np
import torch
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


def build_calibrated_sampling_matrix(image_shape, calib_u, calib_v,
                                     calib_phi, calib_theta,
                                     sample_phi, sample_theta, filter_std=2.5):
    """Ommatidium sampler for a camera with a scattered (pixel <-> angle) calibration.

    Unlike ``build_ommatidium_sampling_matrix`` (which assumes a linear
    equirectangular pixel->angle grid), this handles a rig — e.g. a dome — whose
    mapping is only known at a sparse set of calibration points. We interpolate:
        * an inverse map (phi, theta) -> (u, v) to place each ommatidium's pixel
          center (columns outside the calibration convex hull are out of FOV);
        * a forward per-pixel map (u, v) -> (phi, theta) used to weight pixels
          near each center by (flat phi/theta) angular distance with the same
          Gaussian ``filter_std`` (degrees) convention used elsewhere.

    Args:
        image_shape: (H, W) of the video frame.
        calib_u, calib_v: (M,) pixel coordinates of the calibration points.
        calib_phi, calib_theta: (M,) viewing angles (radians) at those pixels.
        sample_phi, sample_theta: (N,) ommatidium angles (radians) to sample.
        filter_std: Gaussian filter std in degrees.

    Returns:
        W: csr_matrix (N, H*W). Usage: sampled = W @ image.ravel().
        in_fov: (N,) bool, True where the column maps inside the calibration.
    """
    from scipy.interpolate import LinearNDInterpolator, griddata

    H, W_img = image_shape
    filter_std_rad = filter_std * np.pi / 180
    cutoff_rad = 2 * filter_std_rad

    calib_u = np.asarray(calib_u, dtype=float)
    calib_v = np.asarray(calib_v, dtype=float)
    calib_phi = np.asarray(calib_phi, dtype=float)
    calib_theta = np.asarray(calib_theta, dtype=float)
    sample_phi = np.asarray(sample_phi, dtype=float)
    sample_theta = np.asarray(sample_theta, dtype=float)
    N = len(sample_phi)

    calib_px = np.column_stack([calib_u, calib_v])
    calib_ang = np.column_stack([calib_phi, calib_theta])

    # Inverse map angle -> pixel for each ommatidium center (NaN outside hull).
    inv_interp = LinearNDInterpolator(calib_ang, calib_px)
    centers = inv_interp(np.column_stack([sample_phi, sample_theta]))  # (N, 2)
    in_fov = np.isfinite(centers).all(axis=1)

    # Forward per-pixel angle maps for angular-distance weighting.
    grid_u, grid_v = np.meshgrid(np.arange(W_img), np.arange(H))
    phi_map = griddata(calib_px, calib_phi, (grid_u, grid_v), method='linear')
    theta_map = griddata(calib_px, calib_theta, (grid_u, grid_v), method='linear')

    # Pixels-per-radian (global estimate) to size the search window in pixels.
    span_phi = max(calib_phi.max() - calib_phi.min(), 1e-6)
    span_theta = max(calib_theta.max() - calib_theta.min(), 1e-6)
    ppr = max((calib_u.max() - calib_u.min()) / span_phi,
              (calib_v.max() - calib_v.min()) / span_theta)
    window = int(np.ceil(cutoff_rad * ppr)) + 1

    rows, cols, vals = [], [], []
    for i in range(N):
        if not in_fov[i]:
            continue
        u0, v0 = centers[i]
        k_lo = max(0, int(np.floor(u0)) - window)
        k_hi = min(W_img - 1, int(np.ceil(u0)) + window)
        j_lo = max(0, int(np.floor(v0)) - window)
        j_hi = min(H - 1, int(np.ceil(v0)) + window)

        phi_patch = phi_map[j_lo:j_hi + 1, k_lo:k_hi + 1]
        theta_patch = theta_map[j_lo:j_hi + 1, k_lo:k_hi + 1]
        d_sq = (phi_patch - sample_phi[i]) ** 2 + (theta_patch - sample_theta[i]) ** 2

        weights = np.exp(-d_sq / (2 * filter_std_rad ** 2))
        weights[~np.isfinite(d_sq)] = 0
        weights[d_sq > cutoff_rad ** 2] = 0

        weight_sum = weights.sum()
        if weight_sum <= 0:
            in_fov[i] = False
            continue
        weights /= weight_sum

        nz_mask = weights > 0
        local_rows, local_cols = np.where(nz_mask)
        flat_indices = (local_rows + j_lo) * W_img + (local_cols + k_lo)
        n_nz = flat_indices.size
        rows.extend([i] * n_nz)
        cols.extend(flat_indices.tolist())
        vals.extend(weights[nz_mask].tolist())

    W = csr_matrix((vals, (rows, cols)), shape=(N, H * W_img))
    return W, in_fov


def resample_time_to_length(data, n_out: int):
    """
    Linearly resample along time (dimension 1) to exactly ``n_out`` steps.

    Used after ``interpolate_to_model_dt`` when the nominal stimulus duration
    (``stim_time``) does not match the duration implied by the available native
    frames (e.g. short clips upsampled to match config).
    """
    if n_out < 1:
        raise ValueError("n_out must be at least 1")

    was_numpy = isinstance(data, np.ndarray)
    if was_numpy:
        data = torch.from_numpy(data)

    original_dtype = data.dtype
    batch_size, t_in = data.shape[0], data.shape[1]
    remaining_dims = data.shape[2:]

    if t_in == n_out:
        out = data
    else:
        data_flat = data.reshape(batch_size, t_in, -1).permute(0, 2, 1)
        data_rs = torch.nn.functional.interpolate(
            data_flat.float(),
            size=n_out,
            mode='linear',
            align_corners=True,
        ).to(dtype=original_dtype)
        out = data_rs.permute(0, 2, 1).reshape((batch_size, n_out) + remaining_dims)

    if was_numpy:
        out = out.numpy()
    return out


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
        ValueError: If model_dt is meaningfully larger than stimulus_dt (would
            fabricate temporal resolution). A small relative tolerance allows
            datasets recorded at ~the model rate (e.g. 30.303 Hz data vs 30 Hz model).
    """
    # Treat rates within rel_tol as the same sampling rate; resample_time_to_length
    # handles the final exact length. This avoids hard-failing on negligible
    # rate mismatches while still blocking genuine upsampling.
    rel_tol = 0.05
    if model_dt > stimulus_dt * (1.0 + rel_tol):
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
