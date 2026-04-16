"""
Bridge between the drosophila_model stimulus pipeline and FlyEye hexagonal
visualization.

The main repo generates stimuli as (time, columns) arrays where each column
has (phi, theta) and (p, q) coordinates.  The FlyEye package visualizes
ommatidial responses on a hexagonal grid using (p, q).

This module provides functions to:
  - Render any per-column stimulus or neural response as an animated hex mosaic.
  - Build a FlyVisMapper-compatible weight matrix from the model's column
    geometry so that arbitrary panoramic images can be sampled at model columns
    and visualized.

Coordinate conventions
----------------------
Main repo (physics):  theta from +z pole, phi azimuthal.  Radians.
FlyEye (geographic):  elevation from equator, azimuth horizontal.  Degrees.

Conversion:
    elevation_deg = 90 - degrees(theta)
    azimuth_deg   = degrees(phi)
"""

from pathlib import Path

import numpy as np
import matplotlib.cm as cm

from .hexagonal_animation import color_hexagons
from .fly_vis_mapper import FlyVisMapper
from .utils import unit_vectors_from_azel, angular_distance


def visualize_column_responses(
    column_p_q,
    values,
    *,
    colormap=None,
    fps=20,
    filename=None,
    plot=True,
    vmin=None,
    vmax=None,
    titles=None,
):
    """
    Animate per-column responses on the hexagonal ommatidial grid.

    The connectome's ``column_info_df`` stores raw (p, q).  The FlyEye
    visualization pipeline (``pixel_from_flat_hex`` + 60-degree rotation)
    expects q to be negated -- this is the same transform applied in
    ``extract_fly_eye_data.py`` when creating the ``.npz`` files from the
    connectome parquet.  This function handles that automatically.

    Parameters
    ----------
    column_p_q : ndarray (N, 2)
        Hex grid coordinates from ``column_info_df[['p', 'q']]``.
    values : ndarray (T, N)
        Per-column values over time.  Can be raw stimulus (0-1), model
        firing rates, or any scalar quantity.
    colormap : ndarray (256, 4), optional
        RGBA colormap.  Default: plasma.
    fps : int
        Frames per second for the animation.
    filename : str, optional
        If given, save the animation to this file (e.g. ``"stim.mp4"``).
    plot : bool
        Show the animation interactively.
    vmin, vmax : float, optional
        Clamp values to [vmin, vmax] before mapping to colormap.  Defaults
        to the data range.
    titles : list[str], optional
        Per-frame titles.
    """
    p = np.asarray(column_p_q[:, 0])
    q = -np.asarray(column_p_q[:, 1])  # negate q to match FlyEye convention (see extract_fly_eye_data.py)

    values = np.asarray(values, dtype=np.float64)
    if vmin is None:
        vmin = np.nanmin(values)
    if vmax is None:
        vmax = np.nanmax(values)

    span = vmax - vmin
    if span == 0:
        span = 1.0
    scaled = np.clip((values - vmin) / span, 0, 1) * 255
    scaled = np.nan_to_num(scaled, nan=0.0).astype(np.int32)

    if colormap is None:
        colormap = cm.plasma(np.linspace(0, 1, 256))

    color_hexagons(p, q, scaled, plot=plot, filename=filename,
                   colormap=colormap, fps=fps, titles=titles,
                   vmin=vmin, vmax=vmax)


def phi_theta_to_az_el_deg(phi, theta):
    """Convert main-repo (phi, theta) radians to FlyEye (azimuth, elevation) degrees."""
    azimuth_deg = np.degrees(phi)
    elevation_deg = 90.0 - np.degrees(theta)
    return azimuth_deg, elevation_deg


def make_column_mapper(
    column_phi_theta,
    image_shape,
    image_phi_lim,
    image_theta_lim,
    sigma_deg=2.5,
):
    """
    Build a FlyVisMapper-style weight matrix from the model's column geometry
    and an equirectangular image grid.

    This lets you sample arbitrary panoramic images at model column positions
    using the same angular-distance Gaussian weighting that FlyVisMapper uses,
    but without needing camera calibration files.

    Parameters
    ----------
    column_phi_theta : ndarray (N, 2)
        Column (phi, theta) in radians (main repo convention).
    image_shape : (H, W)
        Shape of the source panoramic image.
    image_phi_lim : (2,) float
        Phi limits of the image [start, end] in radians.
    image_theta_lim : (2,) float
        Theta limits of the image [start, end] in radians.
    sigma_deg : float
        Gaussian sigma in degrees for ommatidial filtering.

    Returns
    -------
    mapper : FlyVisMapper
        A mapper whose ``.W`` is set up to sample the given image grid.
        Call ``mapper.ommatidial_response(img)`` to get per-column values.
    """
    H, W = image_shape
    phi = column_phi_theta[:, 0]
    theta = column_phi_theta[:, 1]

    az_deg, el_deg = phi_theta_to_az_el_deg(phi, theta)

    theta_pixels = np.linspace(image_theta_lim[0], image_theta_lim[1], H)
    phi_pixels = np.linspace(image_phi_lim[0], image_phi_lim[1], W)
    phi_grid, theta_grid = np.meshgrid(phi_pixels, theta_pixels)

    img_az_deg = np.degrees(phi_grid)
    img_el_deg = 90.0 - np.degrees(theta_grid)

    mapper = FlyVisMapper.__new__(FlyVisMapper)
    mapper.camera_azimuth = img_az_deg
    mapper.camera_elevation = img_el_deg
    mapper.fly_azimuth = az_deg
    mapper.fly_elevation = el_deg
    mapper.p = np.zeros(len(phi), dtype=int)
    mapper.q = np.zeros(len(phi), dtype=int)

    mapper.make_weight_matrix(sigma_deg=sigma_deg, sparse=True)
    return mapper
