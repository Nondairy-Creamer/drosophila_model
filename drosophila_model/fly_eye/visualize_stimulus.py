"""
Render per-column stimuli or neural responses as an animated hexagonal mosaic.

Stimuli and model outputs are (time, columns) arrays; each optic-lobe column
has hex-lattice coordinates (p, q) (``model.regions['optic_lobe'].column_p_q``).
:func:`visualize_column_responses` draws one hexagon per column at
:func:`fly_eye.utils.column_hex_centers` and colours it frame by frame.
"""

import numpy as np
import matplotlib.cm as cm

from .hexagonal_animation import color_hexagons
from .utils import column_hex_centers


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

    Columns are placed by :func:`fly_eye.utils.column_hex_centers` (q negated,
    flat-hex layout rotated by ``-ROTATE_CORRECTION`` degrees), so raw
    connectome (p, q) can be passed directly.

    Parameters
    ----------
    column_p_q : ndarray (N, 2)
        Raw connectome hex-lattice coordinates per column
        (``model.regions['optic_lobe'].column_p_q``).
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
    x, y = column_hex_centers(column_p_q)

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

    color_hexagons(x, y, scaled, plot=plot, filename=filename,
                   colormap=colormap, fps=fps, titles=titles,
                   vmin=vmin, vmax=vmax)
