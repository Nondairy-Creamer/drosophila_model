"""
Shared utilities for movie scripts.

Provides:
  - Movie rendering helpers (masked_rgba, setup_eye_axes, write_movie,
    ylim_from_data). Per-eye flow rendering helpers are re-exported from
    analysis_utilities (prepare_eye_quiver, init_eye_quiver, update_eye_quiver,
    add_eye_colorwheel, ...).
  - Scatter/quiver factory functions for the factory-based movie pipeline
  - Re-exports from analysis_utilities (fig_to_rgb, angular_flow_to_cartesian_xz,
    get_flow_key, make_flow_comparison_movie, make_stimulus_flow_movie)
"""
import sys
from pathlib import Path

_repo_root = str(Path(__file__).parent.parent)
sys.path.insert(0, _repo_root)
sys.path.insert(0, str(Path(_repo_root) / "analysis"))

import numpy as np
import matplotlib
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
import imageio as iio_v2

from analysis_utilities import (
    fig_to_rgb,
    angular_flow_to_cartesian_xz,
    get_flow_key,
    make_flow_comparison_movie,
    make_stimulus_flow_movie,
    flow_direction_colors,
    update_direction_quiver,
    add_direction_colorwheel,
    # Eye-view single-source-of-truth helpers
    eye_cell_y,
    eye_color_flip,
    project_eye_flow,
    prepare_eye_quiver,
    setup_eye_axis,
    setup_eye_axes as _setup_eye_axes_canonical,
    eye_quiver_colors,
    init_eye_quiver,
    update_eye_quiver,
    add_eye_colorwheel,
)

_MASK_COLOR = np.array([1.0, 1.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# Shared helpers extracted from movie scripts
# ---------------------------------------------------------------------------


def masked_rgba(values, mask, cmap=None, norm=None):
    """Convert scalar values to RGBA, painting masked columns yellow.

    Parameters
    ----------
    values : (N,) array
    mask : (N,) bool array or None — True = valid, False = masked
    cmap : matplotlib Colormap (default: gray)
    norm : matplotlib Normalize
    """
    if cmap is None:
        cmap = plt.cm.gray
    if norm is None:
        norm = mcolors.Normalize(vmin=np.nanmin(values), vmax=np.nanmax(values))
    rgba = cmap(norm(values))
    if mask is not None:
        rgba[~mask] = _MASK_COLOR
    return rgba


def setup_eye_axes(ax_left, ax_right, cell_x, cell_z, x_margin=None, z_margin=None):
    """Configure a pair of left/right eye axes (canonical convention).

    Thin wrapper around the canonical
    :func:`analysis_utilities.setup_eye_axes` (re-exported here under
    ``_setup_eye_axes_canonical``) so existing call sites keep working.
    """
    _setup_eye_axes_canonical(ax_left, ax_right, cell_x, cell_z,
                              x_margin=x_margin, z_margin=z_margin)


def write_movie(fig, update_fn, num_frames, output_path, frame_rate, log_every=50):
    """Render frames from *update_fn* into an H.264 movie.

    Parameters
    ----------
    fig : matplotlib Figure
    update_fn : callable(t) that updates the figure for frame *t*
    num_frames : total frames
    output_path : Path-like
    frame_rate : fps
    log_every : print progress every N frames (0 to disable)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Writing {num_frames} frames at {frame_rate} fps -> {output_path}")
    with iio_v2.get_writer(str(output_path), codec="libx264", fps=frame_rate) as writer:
        for t in range(num_frames):
            update_fn(t)
            writer.append_data(fig_to_rgb(fig))
            if log_every and (t + 1) % log_every == 0:
                print(f"    Frame {t + 1}/{num_frames}")
    plt.close(fig)
    print(f"  Done! -> {output_path}")


def ylim_from_data(data, margin_frac=0.1, min_margin=0.01):
    """Compute y-axis limits with a margin around data range."""
    finite = data[np.isfinite(data)] if np.issubdtype(data.dtype, np.floating) else data.ravel()
    if finite.size == 0:
        return -1, 1
    lo, hi = float(finite.min()), float(finite.max())
    margin = max(margin_frac * (hi - lo), min_margin)
    return lo - margin, hi + margin



def get_fov(in_fov, t):
    """Index a field-of-view mask that may be static (N,) or time-varying (T, N)."""
    if in_fov.ndim == 2:
        return in_fov[t]
    return in_fov


def compute_filtered(model, input_batch):
    """Run input conv step-by-step, return (T, N) numpy array."""
    import torch
    filter_state = None
    frames = []
    for t in range(input_batch.shape[1]):
        y_t, filter_state = model.regions['optic_lobe'].apply_input_conv(input_batch[:, t, :], filter_state)
        frames.append(y_t[0])
    return torch.stack(frames, dim=0).cpu().numpy()


def subtract_rotation_flow(pred_flow_right, pred_flow_left, omega, column_phi_theta):
    """Subtract rotation-induced flow, returning translation-only angular flow."""
    phi_r = column_phi_theta[:, 0]
    theta = column_phi_theta[:, 1]
    phi_l = -phi_r

    def _rotation_angular_flow(phi, theta, omega):
        x = np.stack([np.sin(theta) * np.cos(phi),
                      np.sin(theta) * np.sin(phi),
                      np.cos(theta)], axis=-1)
        e_phi = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=-1)
        e_theta = np.stack([np.cos(theta) * np.cos(phi),
                            np.cos(theta) * np.sin(phi),
                            -np.sin(theta)], axis=-1)
        u_rot = np.cross(x[None, :, :], omega[:, None, :])
        d_right_rot = -np.einsum('tnj,nj->tn', u_rot, e_phi)
        d_up_rot = -np.einsum('tnj,nj->tn', u_rot, e_theta)
        return d_right_rot, d_up_rot

    dr_r, du_r = _rotation_angular_flow(phi_r, theta, omega)
    dr_l, du_l = _rotation_angular_flow(phi_l, theta, omega)

    out_right = pred_flow_right.copy()
    out_right[:, :, 0] -= dr_r
    out_right[:, :, 1] -= du_r

    out_left = pred_flow_left.copy()
    out_left[:, :, 0] -= dr_l
    out_left[:, :, 1] -= du_l

    return out_right, out_left


# ---------------------------------------------------------------------------
# Factory-based movie pipeline (formerly root movie_utilities.py)
# ---------------------------------------------------------------------------


def make_scatter_plot_factory(pred_plot, cell_x, cell_z):
    """
    Returns a factory callable that, given a direction index, initializes a scatter plot and
    provides an update function for per-frame rendering.
    """
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    def init_plot_for_direction(d, di):
        fig, ax = plt.subplots(figsize=(6,4), dpi=120)
        chosen_activity = pred_plot[di, :, :]

        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        sc = ax.scatter(cell_x, cell_z, c=chosen_activity[0, :], cmap="viridis", norm=norm)
        cbar = fig.colorbar(sc, ax=ax)
        ax.set_xlabel('location in x, posterior <-> anterior')
        ax.set_ylabel('location in z, ventral <-> dorsal')
        ax.set_title(d)
        num_frames = chosen_activity.shape[0]

        def update(frame_t):
            sc.set_array(chosen_activity[frame_t, :])
            cbar.update_normal(sc)

        return fig, update, num_frames

    init_plot_for_direction.num_stimuli = int(pred_plot.shape[0])
    init_plot_for_direction.uses_stimuli_index = True
    return init_plot_for_direction


def _load_flyeye_utils():
    """Load FlyEye/utils.py directly, bypassing FlyEye/__init__.py so we don't
    pull in optional deps (natsort, cv2) required by other FlyEye submodules."""
    import importlib.util
    utils_path = Path(_repo_root) / "FlyEye" / "utils.py"
    spec = importlib.util.spec_from_file_location("flyeye_utils_local", utils_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _hex_polys_centered_to_extent(column_p_q, target_x, target_z):
    """
    Build hexagon polygons for ``column_p_q`` (using FlyEye's flat-hex layout),
    centered at the origin and scaled to match the extent spanned by
    ``target_x``/``target_z``. Returns ``(verts, scale)`` where ``verts`` is
    an ``(N, 6, 2)`` array suitable for ``matplotlib.collections.PolyCollection``.
    """
    flyeye_utils = _load_flyeye_utils()
    pixel_from_flat_hex = flyeye_utils.pixel_from_flat_hex
    rotate_coords = flyeye_utils.rotate_coords
    hex_vertices = flyeye_utils.hex_vertices

    p = np.asarray(column_p_q[:, 0])
    q = -np.asarray(column_p_q[:, 1])  # FlyEye convention (see visualize_stimulus.py)
    x, y = pixel_from_flat_hex(p, q)
    x, y = rotate_coords(x, y, -60.0)  # ROTATE_CORRECTION
    x = x - x.mean()
    y = y - y.mean()

    target_span = max(np.ptp(target_x), np.ptp(target_z), 1e-9)
    raw_span = max(np.ptp(x), np.ptp(y), 1e-9)
    s = target_span / raw_span

    verts = hex_vertices(x * s, y * s, size=s)  # size is radius in pixel units
    return verts, s


def make_scatter_diamond_sequence_factory(responses_dict, cell_x, cell_z, offset=2.6, scale=1.0, size_scale=1.0, stim_cmap='viridis', stim_scale_frac=0.5, stim_size_frac=0.2, stim_p_q=None):
    """
    New simple interface:
      - responses_dict: dict with 5 keys (in insertion order):
          ['<right>', '<left>', '<up>', '<down>', 'stimulus']
        Each value is a numpy array shaped [num_stimuli, num_time, num_columns].
      - The four non-'stimulus' keys are plotted simultaneously in a diamond:
          order (in insertion): Right at (+offset, 0), Left at (-offset, 0),
          Up at (0, +offset), Down at (0, -offset).
        'stimulus' is plotted at the center with half spatial scale and smaller markers.
      - For each stimulus index, we render its full time series, followed by 25 blank frames.
      - If ``stim_p_q`` (an ``(N, 2)`` hex-grid coord array) is provided, the
        center stimulus is rendered as a hexagonal mosaic via FlyEye's hex
        layout instead of a scatter (cell panels remain scatters).
    The returned init callable conforms to make_movie: init(d, di) -> (fig, update(frame_t), num_frames).
    This factory concatenates ALL stimuli back-to-back into a single movie (with gaps),
    so the stimulus index parameter is ignored.
    """
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    keys_in_order = list(responses_dict.keys())
    cell_keys = [k for k in keys_in_order if k != 'stimulus']
    cell_keys = cell_keys[:4]
    position_by_index = {
        0: (+offset, 0.0),   # right
        1: (-offset, 0.0),   # left
        2: (0.0, +offset),   # up
        3: (0.0, -offset),   # down
    }

    def init_plot_for_direction(d, di):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
        try:
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax.set_position([0, 0, 1, 1])
        except Exception:
            pass

        stim_coord_scale = stim_scale_frac * scale
        coords = []
        for idx in range(len(cell_keys)):
            dx, dz = position_by_index[idx]
            coords.append((scale * cell_x + dx, scale * cell_z + dz))
        coords.append((stim_coord_scale * cell_x, stim_coord_scale * cell_z))
        x_all = np.concatenate([xy[0] for xy in coords]) if coords else (scale * cell_x)
        y_all = np.concatenate([xy[1] for xy in coords]) if coords else (scale * cell_z)
        x_min, x_max = np.min(x_all), np.max(x_all)
        y_min, y_max = np.min(y_all), np.max(y_all)
        x_span = max(1e-6, x_max - x_min)
        y_span = max(1e-6, y_max - y_min)
        span = max(x_span, y_span)
        margin_frac = 0.02
        half = 0.5 * span * (1.0 + margin_frac)
        x_center = 0.5 * (x_min + x_max)
        y_center = 0.5 * (y_min + y_max)
        ax.set_xlim(x_center - half, x_center + half)
        ax.set_ylim(y_center - half, y_center + half)
        ax.set_aspect('equal', 'box')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_axis_off()

        base_ms = plt.rcParams.get("lines.markersize", 6.0)
        base_s = float(base_ms) ** 2
        s_dir = max(1.0, base_s * size_scale)
        s_stim = max(1.0, s_dir * stim_size_frac)

        stim_all = responses_dict.get('stimulus', None)
        if stim_all is not None:
            num_stimuli, L, num_cols = stim_all.shape
        elif len(cell_keys) > 0:
            num_stimuli, L, num_cols = responses_dict[cell_keys[0]].shape
        else:
            num_stimuli, L, num_cols = 1, 0, cell_x.shape[0]

        zero_vec = np.zeros(num_cols, dtype=float)

        dir_scatters = []
        for idx, key in enumerate(cell_keys):
            arr_all = responses_dict.get(key, None)
            dx, dz = position_by_index[idx]
            initial = arr_all[0, 0, :] if (arr_all is not None and num_stimuli > 0 and L > 0) else zero_vec
            sc = ax.scatter(
                scale * cell_x + dx, scale * cell_z + dz,
                c=initial, cmap="viridis", norm=norm, s=s_dir,
                edgecolors='none', linewidths=0.0, antialiased=False,
            )
            dir_scatters.append((sc, arr_all))

        stim_initial = np.zeros(num_cols, dtype=float)
        if stim_all is not None and num_stimuli > 0 and L > 0:
            stim_initial = stim_all[0, 0, :]

        if stim_p_q is not None:
            from matplotlib.collections import PolyCollection
            if stim_all is not None and num_stimuli > 0 and L > 0:
                hex_valid_mask = np.any(~np.isnan(stim_all), axis=(0, 1))
            else:
                hex_valid_mask = np.ones(num_cols, dtype=bool)
            hex_p_q = np.asarray(stim_p_q)[hex_valid_mask]
            target_x = stim_coord_scale * cell_x
            target_z = stim_coord_scale * cell_z
            verts, _hex_scale = _hex_polys_centered_to_extent(hex_p_q, target_x, target_z)
            stim_artist = PolyCollection(
                verts, cmap=stim_cmap, norm=norm,
                edgecolors='none', linewidths=0.0, antialiased=False,
            )
            stim_artist.set_array(stim_initial[hex_valid_mask])
            ax.add_collection(stim_artist)
            stim_is_hex = True
        else:
            hex_valid_mask = None
            stim_artist = ax.scatter(
                stim_coord_scale * cell_x, stim_coord_scale * cell_z,
                c=stim_initial, cmap=stim_cmap, norm=norm, s=s_stim,
                edgecolors='none', linewidths=0.0, antialiased=False,
            )
            stim_is_hex = False

        blanks = 25
        total_frames = num_stimuli * (L + blanks)

        stim_zero = zero_vec[hex_valid_mask] if stim_is_hex else zero_vec

        def _stim_frame(arr_full):
            return arr_full[hex_valid_mask] if stim_is_hex else arr_full

        def update(frame_t):
            if total_frames == 0 or L == 0:
                for sc, _arr in dir_scatters:
                    sc.set_array(zero_vec)
                stim_artist.set_array(stim_zero)
                return
            block = L + blanks
            stim_idx, within = divmod(frame_t, block)
            if stim_idx >= num_stimuli:
                stim_idx = num_stimuli - 1
            if within < L:
                t = within
                for sc, arr_all in dir_scatters:
                    if arr_all is not None and stim_idx < arr_all.shape[0]:
                        sc.set_array(arr_all[stim_idx, t, :])
                    else:
                        sc.set_array(zero_vec)
                if stim_all is not None and stim_idx < stim_all.shape[0]:
                    stim_artist.set_array(_stim_frame(stim_all[stim_idx, t, :]))
                else:
                    stim_artist.set_array(stim_zero)
            else:
                for sc, _arr in dir_scatters:
                    sc.set_array(zero_vec)
                stim_artist.set_array(stim_zero)

        return fig, update, total_frames

    init_plot_for_direction.num_stimuli = 1
    init_plot_for_direction.uses_stimuli_index = False
    return init_plot_for_direction


def make_scatter_pair_sequence_factory(responses_dict, cell_x, cell_z, offset=2.6, scale=1.0, size_scale=1.0):
    """
    Side-by-side sequence view for a single cell type vs stimulus.
      - responses_dict: dict containing exactly one non-'stimulus' key (treated as the cell)
        and an optional 'stimulus' key.
        Values are numpy arrays shaped [num_stimuli, num_time, num_columns].
      - Plots the cell on the RIGHT at (+offset, 0) and the stimulus on the LEFT at (-offset, 0).
      - Concatenates all stimuli back-to-back with a blank gap between them (like diamond sequence).
    Returns an init callable suitable for make_movie that ignores the stimuli index parameter.
    """
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    keys_in_order = list(responses_dict.keys())
    cell_keys = [k for k in keys_in_order if k != 'stimulus']
    cell_key = cell_keys[0] if len(cell_keys) > 0 else None
    stim_all = responses_dict.get('stimulus', None)

    def init_plot_for_direction(d, di):
        fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
        try:
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax.set_position([0, 0, 1, 1])
        except Exception:
            pass

        left_x = -offset
        right_x = +offset
        coords = [
            (scale * cell_x + left_x, scale * cell_z + 0.0),
            (scale * cell_x + right_x, scale * cell_z + 0.0),
        ]
        x_all = np.concatenate([xy[0] for xy in coords])
        y_all = np.concatenate([xy[1] for xy in coords])
        x_min, x_max = np.min(x_all), np.max(x_all)
        y_min, y_max = np.min(y_all), np.max(y_all)
        x_span = max(1e-6, x_max - x_min)
        y_span = max(1e-6, y_max - y_min)
        span = max(x_span, y_span)
        margin_frac = 0.02
        half = 0.5 * span * (1.0 + margin_frac)
        x_center = 0.5 * (x_min + x_max)
        y_center = 0.5 * (y_min + y_max)
        ax.set_xlim(x_center - half, x_center + half)
        ax.set_ylim(y_center - half, y_center + half)
        ax.set_aspect('equal', 'box')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_axis_off()

        base_ms = plt.rcParams.get("lines.markersize", 6.0)
        base_s = float(base_ms) ** 2
        s_cell = max(1.0, base_s * size_scale)
        s_stim = max(1.0, s_cell * 0.8)

        if stim_all is not None:
            num_stimuli, L, num_cols = stim_all.shape
        else:
            cell_all = responses_dict.get(cell_key, None)
            if cell_all is not None:
                num_stimuli, L, num_cols = cell_all.shape
            else:
                num_stimuli, L, num_cols = 1, 0, cell_x.shape[0]

        zero_vec = np.zeros(num_cols, dtype=float)

        stim_initial = np.zeros(num_cols, dtype=float)
        if stim_all is not None and num_stimuli > 0 and L > 0:
            stim_initial = stim_all[0, 0, :]
        stim_scatter = ax.scatter(
            scale * cell_x + left_x, scale * cell_z + 0.0,
            c=stim_initial, cmap="viridis", norm=norm, s=s_stim,
            edgecolors='none', linewidths=0.0, antialiased=False,
        )

        cell_initial = np.zeros(num_cols, dtype=float)
        cell_all = responses_dict.get(cell_key, None)
        if cell_all is not None and num_stimuli > 0 and L > 0:
            cell_initial = cell_all[0, 0, :]
        cell_scatter = ax.scatter(
            scale * cell_x + right_x, scale * cell_z + 0.0,
            c=cell_initial, cmap="viridis", norm=norm, s=s_cell,
            edgecolors='none', linewidths=0.0, antialiased=False,
        )

        blanks = 25
        total_frames = num_stimuli * (L + blanks)

        def update(frame_t):
            if total_frames == 0 or L == 0:
                stim_scatter.set_array(zero_vec)
                cell_scatter.set_array(zero_vec)
                return
            block = L + blanks
            stim_idx, within = divmod(frame_t, block)
            if stim_idx >= num_stimuli:
                stim_idx = num_stimuli - 1
            if within < L:
                t = within
                if stim_all is not None and stim_idx < stim_all.shape[0]:
                    stim_scatter.set_array(stim_all[stim_idx, t, :])
                else:
                    stim_scatter.set_array(zero_vec)
                if cell_all is not None and stim_idx < cell_all.shape[0]:
                    cell_scatter.set_array(cell_all[stim_idx, t, :])
                else:
                    cell_scatter.set_array(zero_vec)
            else:
                stim_scatter.set_array(zero_vec)
                cell_scatter.set_array(zero_vec)

        return fig, update, total_frames

    init_plot_for_direction.num_stimuli = 1
    init_plot_for_direction.uses_stimuli_index = False
    return init_plot_for_direction


def make_quiver_plot_factory(x_flow, z_flow, cell_x, cell_z):
    """
    Returns a factory callable that, given a direction index, initializes a quiver plot (+X from pred_plot_0, +Y from pred_plot_1)
    and provides an update function for per-frame rendering. Arrow color encodes direction (hue) and amplitude (saturation).
    """
    _, max_mag = flow_direction_colors(x_flow, z_flow)

    def init_plot_for_direction(d, di):
        fig, ax = plt.subplots(figsize=(6,4), dpi=120)
        chosen_u = -x_flow[di, :, :]
        chosen_v = z_flow[di, :, :]

        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        colors_0, _ = flow_direction_colors(chosen_u[0, :], chosen_v[0, :], max_mag)
        q = ax.quiver(
            cell_x, cell_z,
            chosen_u[0, :], chosen_v[0, :],
            angles='xy', scale_units='xy', scale=1.0,
        )
        q.set_facecolor(colors_0)
        q.set_edgecolor(colors_0)
        add_direction_colorwheel(fig)
        ax.set_xlabel('location in x, posterior <-> anterior')
        ax.set_ylabel('location in z, ventral <-> dorsal')
        ax.set_title(d)
        num_frames = chosen_u.shape[0]

        def update(frame_t):
            update_direction_quiver(q, chosen_u[frame_t, :], chosen_v[frame_t, :], max_mag)

        return fig, update, num_frames

    init_plot_for_direction.num_stimuli = int(x_flow.shape[0])
    init_plot_for_direction.uses_stimuli_index = True
    return init_plot_for_direction


def make_movie(init_plot_for_direction, directions, fig_folder, frame_rate, save_name=''):
    """
    Generic movie maker that supports multiple stimuli per direction.
    - Determines the number of stimuli from the initializer's `num_stimuli` attribute.
    - Splits the stimuli into consecutive blocks across len(directions) directions.
    - If the initializer does not use a stimuli index (uses_stimuli_index=False), falls back to one movie per direction.
    - Output filenames: {save_name}_{direction}_{stimulus_index}.mp4 when per-stimulus; else {save_name}_{direction}.mp4
    """
    num_directions = len(directions)
    total_stimuli = getattr(init_plot_for_direction, 'num_stimuli', num_directions)
    uses_index = getattr(init_plot_for_direction, 'uses_stimuli_index', True)

    if not uses_index:
        for di, d in enumerate(directions):
            fig, update, num_frames = init_plot_for_direction(d, di)
            out_path = fig_folder / f"{save_name}_{d}.mp4"
            with iio_v2.get_writer(out_path, codec="libx264", fps=frame_rate) as writer:
                for t in range(num_frames):
                    update(t)
                    writer.append_data(fig_to_rgb(fig))
            plt.close(fig)
        return

    base = total_stimuli // num_directions
    remainder = total_stimuli % num_directions
    start_index = 0
    for dir_index, d in enumerate(directions):
        block_size = base + (1 if dir_index < remainder else 0)
        for stim_index in range(block_size):
            di = start_index + stim_index
            fig, update, num_frames = init_plot_for_direction(d, di)
            out_path = fig_folder / f"{save_name}_{d}_{stim_index}.mp4"
            with iio_v2.get_writer(out_path, codec="libx264", fps=frame_rate) as writer:
                for t in range(num_frames):
                    update(t)
                    writer.append_data(fig_to_rgb(fig))
            plt.close(fig)
        start_index += block_size


def make_scatter_grid_factory(pair_resp, cell_x, cell_z):
    """
    Returns a factory callable that, given a direction index, initializes a 2x2 grid
    of scatter plots for the four pair responses present in `pair_resp`.
    """
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    preferred_order = ['right_cells', 'left_cells', 'up_cells', 'down_cells']
    keys = [k for k in preferred_order if k in pair_resp]
    keys += [k for k in pair_resp.keys() if k not in keys]

    def init_plot_for_direction(d, di):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=120)
        axes = axes.ravel()

        scatters = []
        colorbars = []
        activities = []
        num_frames = None

        for ax, key in zip(axes, keys):
            chosen_activity = pair_resp[key][di, :, :]
            if num_frames is None:
                num_frames = chosen_activity.shape[0]

            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
            sc = ax.scatter(cell_x, cell_z, c=chosen_activity[0, :], cmap="viridis", norm=norm)
            cbar = fig.colorbar(sc, ax=ax)
            ax.set_xlabel('location in x, posterior <-> anterior')
            ax.set_ylabel('location in z, ventral <-> dorsal')
            ax.set_title(key)

            scatters.append(sc)
            colorbars.append(cbar)
            activities.append(chosen_activity)

        for ax in axes[len(keys):]:
            ax.axis('off')

        def update(frame_t):
            for sc, cbar, chosen_activity in zip(scatters, colorbars, activities):
                sc.set_array(chosen_activity[frame_t, :])
                cbar.update_normal(sc)

        return fig, update, num_frames

    if len(keys) > 0:
        init_plot_for_direction.num_stimuli = int(pair_resp[keys[0]].shape[0])
    else:
        init_plot_for_direction.num_stimuli = 1
    init_plot_for_direction.uses_stimuli_index = True
    return init_plot_for_direction


def make_scatter_grid_3x3_factory(pair_resp, cell_x, cell_z):
    """
    Returns a factory callable that, given a direction index, initializes a 3x3 grid
    of scatter plots where only four positions are populated:
      - 'up_cells' at (0, 1)
      - 'right_cells' at (1, 2)
      - 'left_cells' at (1, 0)
      - 'down_cells' at (2, 1)
    """
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    position_by_key = {
        'up_cells': (0, 1),
        'right_cells': (1, 2),
        'left_cells': (1, 0),
        'down_cells': (2, 1),
    }
    preferred_order = ['up_cells', 'right_cells', 'left_cells', 'down_cells']
    keys = [k for k in preferred_order if k in pair_resp]

    def init_plot_for_direction(d, di):
        fig = plt.figure(figsize=(6, 6), dpi=120)

        scatters = []
        activities = []
        num_frames = None

        cell_centers = {0: 1.0/6.0, 1: 3.0/6.0, 2: 5.0/6.0}
        base_size = 1.0 / 3.0
        side = min(0.45, base_size * 1.3)

        for key in keys:
            r, c = position_by_key[key]
            cx = cell_centers[c]
            cy = 1.0 - cell_centers[r]
            left = max(0.0, cx - side / 2.0)
            bottom = max(0.0, cy - side / 2.0)
            if left + side > 1.0:
                left = 1.0 - side
            if bottom + side > 1.0:
                bottom = 1.0 - side
            ax = fig.add_axes([left, bottom, side, side])

            chosen_activity = pair_resp[key][di, :, :]
            if num_frames is None:
                num_frames = chosen_activity.shape[0]

            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
            sc = ax.scatter(cell_x, cell_z, c=chosen_activity[0, :], cmap="viridis", norm=norm)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_aspect('equal', 'box')
            ax.set_title(key)

            scatters.append(sc)
            activities.append(chosen_activity)

        num_frames = num_frames if num_frames is not None else 0

        def update(frame_t):
            for sc, chosen_activity in zip(scatters, activities):
                sc.set_array(chosen_activity[frame_t, :])

        return fig, update, num_frames

    if len(keys) > 0:
        init_plot_for_direction.num_stimuli = int(pair_resp[keys[0]].shape[0])
    else:
        init_plot_for_direction.num_stimuli = 1
    init_plot_for_direction.uses_stimuli_index = True
    return init_plot_for_direction


def make_scatter_diamond_factory(pair_resp, cell_x, cell_z, offset=2.6, label=False, scale=1.0, size_scale=1.0):
    """
    Returns a factory callable that, given a direction index, initializes a single-axes
    scatter where the four datasets are arranged in a diamond configuration.
    """
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    positions = {
        'up': (0.0, +offset),
        'right': (+offset, 0.0),
        'left': (-offset, 0.0),
        'down': (0.0, -offset),
    }

    def init_plot_for_direction(d, di):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
        try:
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax.set_position([0, 0, 1, 1])
        except Exception:
            pass
        scatters = []
        activities = []
        num_frames = None
        coords = []
        for dx, dz in positions.values():
            coords.append((scale * cell_x + dx, scale * cell_z + dz))
        coords.append(((0.5 * scale) * cell_x, (0.5 * scale) * cell_z))

        if coords:
            x_all = np.concatenate([xy[0] for xy in coords])
            y_all = np.concatenate([xy[1] for xy in coords])
            x_min, x_max = np.min(x_all), np.max(x_all)
            y_min, y_max = np.min(y_all), np.max(y_all)
            x_span = max(1e-6, x_max - x_min)
            y_span = max(1e-6, y_max - y_min)
            span = max(x_span, y_span)
            margin_frac = 0.02
            half = 0.5 * span * (1.0 + margin_frac)
            x_center = 0.5 * (x_min + x_max)
            y_center = 0.5 * (y_min + y_max)
            ax.set_xlim(x_center - half, x_center + half)
            ax.set_ylim(y_center - half, y_center + half)
        else:
            xy_margin = 0.2
            max_extent = offset + (scale * 1.0) + xy_margin
            ax.set_xlim(-max_extent, +max_extent)
            ax.set_ylim(-max_extent, +max_extent)
        ax.set_aspect('equal', 'box')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_axis_off()

        base_ms = plt.rcParams.get("lines.markersize", 6.0)
        base_s = float(base_ms) ** 2
        s_val = max(1.0, base_s * size_scale)

        block = pair_resp.get(d, {})
        resp_arrays = []
        if 'responses' in block and isinstance(block['responses'], dict):
            values_in_order = list(block['responses'].values())
            mapping_dirs = ['right', 'left', 'up', 'down']
            for idx in range(4):
                arr = values_in_order[idx] if idx < len(values_in_order) else None
                resp_arrays.append((mapping_dirs[idx], arr))
        else:
            resp_arrays = [('right', None), ('left', None), ('up', None), ('down', None)]

        lengths = []
        for _, arr in resp_arrays:
            if arr is not None:
                lengths.append(arr.shape[0])
        if 'stimulus' in block:
            lengths.append(block['stimulus'].shape[0])
        num_frames = min(lengths) if lengths else 0

        zero_vec = np.zeros_like(cell_x)

        for dir_name, arr in resp_arrays:
            dx, dz = positions[dir_name]
            initial = arr[0, :] if (arr is not None and num_frames > 0) else zero_vec
            sc = ax.scatter(
                scale * cell_x + dx, scale * cell_z + dz,
                c=initial, cmap="viridis", norm=norm, s=s_val,
                edgecolors='none', linewidths=0.0, antialiased=False,
            )
            if label:
                ax.text(dx, dz + 0.9 * scale, dir_name, ha='center', va='bottom', fontsize=10, color='black')
            scatters.append(sc)
            activities.append(arr if arr is not None else np.zeros((max(1, num_frames), cell_x.shape[0])))

        stim_arr = block.get('stimulus', None)
        stim_initial = stim_arr[0, :] if (stim_arr is not None and num_frames > 0) else zero_vec
        sc = ax.scatter(
            (0.5 * scale) * cell_x, (0.5 * scale) * cell_z,
            c=stim_initial, cmap="viridis", norm=norm, s=max(1.0, s_val * 0.2),
            edgecolors='none', linewidths=0.0, antialiased=False,
        )
        if label:
            ax.text(0.0, 0.9 * 0.5 * scale, 'stimulus', ha='center', va='bottom', fontsize=10, color='black')
        scatters.append(sc)
        activities.append(stim_arr if stim_arr is not None else np.zeros((max(1, num_frames), cell_x.shape[0])))

        num_frames = num_frames if num_frames is not None else 0

        def update(frame_t):
            for sc, chosen_activity in zip(scatters, activities):
                if chosen_activity.shape[0] == 0:
                    sc.set_array(zero_vec)
                else:
                    t = min(frame_t, chosen_activity.shape[0] - 1)
                    sc.set_array(chosen_activity[t, :])

        return fig, update, num_frames

    init_plot_for_direction.num_stimuli = 1
    init_plot_for_direction.uses_stimuli_index = False
    return init_plot_for_direction


def make_quiver_pair_sequence_factory(responses_dict, cell_x, cell_z, offset=2.6, scale=1.0, arrow_scale=1.0):
    """
    Side-by-side sequence view: stimulus (left) and quiver vector field (right).
    """
    stim_norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    keys_in_order = list(responses_dict.keys())
    cell_keys = [k for k in keys_in_order if k != 'stimulus'][:4]
    while len(cell_keys) < 4:
        cell_keys.append(None)
    stim_all = responses_dict.get('stimulus', None)

    arr1 = responses_dict.get(cell_keys[0], None) if cell_keys[0] is not None else None
    arr2 = responses_dict.get(cell_keys[1], None) if cell_keys[1] is not None else None
    arr3 = responses_dict.get(cell_keys[2], None) if cell_keys[2] is not None else None
    arr4 = responses_dict.get(cell_keys[3], None) if cell_keys[3] is not None else None

    def first_shape(*candidates):
        for a in candidates:
            if a is not None:
                return a.shape
        return None

    shape = stim_all.shape if stim_all is not None else first_shape(arr1, arr2, arr3, arr4)
    if shape is None:
        num_stimuli, L, num_cols = 1, 0, cell_x.shape[0]
    else:
        num_stimuli, L, num_cols = shape

    def safe_diff(a, b):
        if a is None and b is None:
            return None
        if a is None:
            return -b
        if b is None:
            return a
        return a - b

    d1 = safe_diff(arr1, arr2)
    d2 = safe_diff(arr3, arr4)
    if d1 is not None and d2 is not None and d1.size > 0 and d2.size > 0:
        _, max_mag = flow_direction_colors(d1, d2)
    else:
        max_mag = 1.0

    def init_plot_for_direction(d, di):
        fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
        try:
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax.set_position([0, 0, 1, 1])
        except Exception:
            pass

        left_x = -offset
        right_x = +offset
        coords = [
            (scale * cell_x + left_x, scale * cell_z + 0.0),
            (scale * cell_x + right_x, scale * cell_z + 0.0),
        ]
        x_all = np.concatenate([xy[0] for xy in coords])
        y_all = np.concatenate([xy[1] for xy in coords])
        x_min, x_max = np.min(x_all), np.max(x_all)
        y_min, y_max = np.min(y_all), np.max(y_all)
        x_span = max(1e-6, x_max - x_min)
        y_span = max(1e-6, y_max - y_min)
        span = max(x_span, y_span)
        margin_frac = 0.02
        half = 0.5 * span * (1.0 + margin_frac)
        x_center = 0.5 * (x_min + x_max)
        y_center = 0.5 * (y_min + y_max)
        ax.set_xlim(x_center - half, x_center + half)
        ax.set_ylim(y_center - half, y_center + half)
        ax.set_aspect('equal', 'box')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_axis_off()

        base_ms = plt.rcParams.get("lines.markersize", 6.0)
        base_s = float(base_ms) ** 2
        s_stim = max(1.0, base_s * 0.8)

        zero_vec = np.zeros(num_cols, dtype=float)

        stim_initial = np.zeros(num_cols, dtype=float)
        if stim_all is not None and num_stimuli > 0 and L > 0:
            stim_initial = stim_all[0, 0, :]
        stim_scatter = ax.scatter(
            scale * cell_x + left_x, scale * cell_z + 0.0,
            c=stim_initial, cmap="viridis", norm=stim_norm, s=s_stim,
            edgecolors='none', linewidths=0.0, antialiased=False,
        )

        if L > 0 and num_stimuli > 0:
            u0 = (arr1[0, 0, :] if arr1 is not None else 0.0) - (arr2[0, 0, :] if arr2 is not None else 0.0)
            v0 = (arr3[0, 0, :] if arr3 is not None else 0.0) - (arr4[0, 0, :] if arr4 is not None else 0.0)
        else:
            u0 = np.zeros(num_cols, dtype=float)
            v0 = np.zeros(num_cols, dtype=float)
        colors_0, _ = flow_direction_colors(u0, v0, max_mag)
        q = ax.quiver(
            scale * cell_x + right_x, scale * cell_z + 0.0,
            u0, v0,
            angles='xy', scale_units='xy',
            scale=max(1e-9, 1.0 / float(arrow_scale)),
        )
        q.set_facecolor(colors_0)
        q.set_edgecolor(colors_0)
        add_direction_colorwheel(fig)

        blanks = 25
        total_frames = num_stimuli * (L + blanks)

        def update(frame_t):
            if total_frames == 0 or L == 0:
                stim_scatter.set_array(zero_vec)
                update_direction_quiver(q, zero_vec, zero_vec, max_mag)
                return
            block = L + blanks
            stim_idx, within = divmod(frame_t, block)
            if stim_idx >= num_stimuli:
                stim_idx = num_stimuli - 1
            if within < L:
                t = within
                if stim_all is not None and stim_idx < stim_all.shape[0]:
                    stim_scatter.set_array(stim_all[stim_idx, t, :])
                else:
                    stim_scatter.set_array(zero_vec)
                u = ((arr1[stim_idx, t, :] if arr1 is not None and stim_idx < arr1.shape[0] else 0.0) -
                     (arr2[stim_idx, t, :] if arr2 is not None and stim_idx < arr2.shape[0] else 0.0))
                v = ((arr3[stim_idx, t, :] if arr3 is not None and stim_idx < arr3.shape[0] else 0.0) -
                     (arr4[stim_idx, t, :] if arr4 is not None and stim_idx < arr4.shape[0] else 0.0))
                update_direction_quiver(q, u, v, max_mag)
            else:
                stim_scatter.set_array(zero_vec)
                update_direction_quiver(q, zero_vec, zero_vec, max_mag)

        return fig, update, total_frames

    init_plot_for_direction.num_stimuli = 1
    init_plot_for_direction.uses_stimuli_index = False
    return init_plot_for_direction
