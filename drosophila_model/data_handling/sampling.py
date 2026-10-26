"""Low-level spatial sampling primitives for stimuli.

These helpers are intentionally dependency-free with respect to the rest of
``data_handling``: ``stimulus_generators.py`` and ``create_training_data.py``
import from here, so this module must not import from either of them.
Keeping the shared primitives in one neutral place lets the dependency flow
one way (generators -> sampling) without import cycles.

Contents:
    * ``load_column_phi_theta`` -- the right-eye ommatidium column directions
      every stimulus is sampled at.
    * ``build_ommatidium_sampling_matrix`` -- sparse (image pixel -> ommatidium)
      sampler.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def load_column_phi_theta(data_folder):
    """Load the right-eye column directions without building the model.

    Reads ``<data_folder>/connectome_data/baked/optic_lobe_columns.parquet``,
    which ships with the connectome data (no script in this repo writes it).
    Rows are ordered by ``column_index``, which must run ``0..N-1``: the model
    addresses stimulus channels by global column index, so row ``k`` of every
    stimulus must be column ``k``.

    Returns:
        ``(N, 2)`` array of ``(phi, theta)`` in radians.
    """
    col_path = Path(data_folder) / 'connectome_data' / 'baked' / 'optic_lobe_columns.parquet'
    if not col_path.exists():
        raise FileNotFoundError(
            f"Optic-lobe column file not found: {col_path}. It ships with the "
            "connectome data (no script in this repo produces it); place it in "
            "connectome_data/baked/."
        )
    column_info_df = pd.read_parquet(col_path).sort_values('column_index')
    if not np.array_equal(column_info_df['column_index'].to_numpy(),
                          np.arange(len(column_info_df))):
        raise ValueError(
            f"{col_path}: column_index must run 0..N-1 with no gaps or duplicates."
        )
    return column_info_df[['phi', 'theta']].to_numpy()


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
