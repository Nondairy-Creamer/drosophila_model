"""
Shared utilities for movie scripts.

Provides:
  - Movie rendering helpers: masked_rgba, write_movie, ylim_from_data, get_fov,
    compute_filtered.
  - Flow helpers: synthesize_two_eye_flow_from_odometry, subtract_rotation_flow.
  - The diamond-layout cell-response movie: make_scatter_diamond_sequence_factory
    + make_movie.
  - Re-exports from analysis_utilities (fig_to_rgb, get_flow_key,
    make_flow_comparison_movie, make_stimulus_flow_movie and the per-eye flow
    rendering helpers prepare_eye_quiver, setup_eye_axis, init_eye_quiver,
    update_eye_quiver, add_eye_colorwheel).
"""
from pathlib import Path

import numpy as np
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
import imageio as iio_v2

from drosophila_model.analysis.model_analysis.analysis_utilities import (
    fig_to_rgb,
    get_flow_key,
    make_flow_comparison_movie,
    make_stimulus_flow_movie,
    # Eye-view single-source-of-truth helpers
    prepare_eye_quiver,
    setup_eye_axis,
    init_eye_quiver,
    update_eye_quiver,
    add_eye_colorwheel,
)
from drosophila_model.fly_eye.utils import column_hex_centers, hex_vertices

__all__ = [
    # defined here
    "masked_rgba",
    "synthesize_two_eye_flow_from_odometry",
    "write_movie",
    "ylim_from_data",
    "get_fov",
    "compute_filtered",
    "subtract_rotation_flow",
    "make_scatter_diamond_sequence_factory",
    "make_movie",
    # re-exported from analysis_utilities
    "fig_to_rgb",
    "get_flow_key",
    "make_flow_comparison_movie",
    "make_stimulus_flow_movie",
    "prepare_eye_quiver",
    "setup_eye_axis",
    "init_eye_quiver",
    "update_eye_quiver",
    "add_eye_colorwheel",
]

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


def synthesize_two_eye_flow_from_odometry(rotation_deg_s, translation, phi, theta,
                                          eye_offset=None):
    """Predicted two-eye optic flow (degrees) from per-frame odometry at depth Z=1.

    Used by the no-/with-model two-eye movie scripts as a fallback when a
    stimulus carries ground-truth ego-motion but no ground-truth optic flow
    (e.g. the drone rigs, whose stored ``optic_flow`` is all-NaN). The left eye
    is the sagittal mirror of the right (``phi -> -phi``; the eye offset's
    ``y`` flips, see ``OpticLobeRegion.mirror_camera_offset``).

    Args:
        rotation_deg_s: (T, 3) observer rotation rate in deg/s.
        translation: (T, 3) observer translation rate.
        phi, theta: (N,) right-eye column angles (radians).
        eye_offset: optional length-3 right-eye camera offset
            (``StimulusData.eye_offset``, same length unit as ``translation``).
            Each eye then translates at ``translation + omega x offset``, the
            rig geometry the two-eye ego-motion solver assumes. ``None`` puts
            both eyes at the body centre.

    Returns:
        ``(flow_right, flow_left)``, each (T, N, 2) in degrees.
    """
    import torch
    from drosophila_model.model_definition.ego_motion import generate_optic_flow
    from drosophila_model.model_definition.regions import OpticLobeRegion

    phi_right_t = torch.tensor(phi, dtype=torch.float32)
    theta_right_t = torch.tensor(theta, dtype=torch.float32)
    phi_left_t = torch.tensor(-phi, dtype=torch.float32)
    omega_t = torch.tensor(np.asarray(rotation_deg_s) * np.pi / 180.0, dtype=torch.float32)
    trans_t = torch.tensor(np.asarray(translation), dtype=torch.float32)

    offset_right = offset_left = None
    if eye_offset is not None:
        offsets = OpticLobeRegion.mirror_camera_offset(eye_offset)
        offset_right = torch.tensor(offsets['right'], dtype=torch.float32)
        offset_left = torch.tensor(offsets['left'], dtype=torch.float32)

    ur_r, uu_r = generate_optic_flow(phi_right_t, theta_right_t, omega_t, trans_t, 1.0,
                                     offset=offset_right)
    ur_l, uu_l = generate_optic_flow(phi_left_t, theta_right_t, omega_t, trans_t, 1.0,
                                     offset=offset_left)
    flow_right = np.stack([ur_r.numpy(), uu_r.numpy()], axis=-1) * (180.0 / np.pi)
    flow_left = np.stack([ur_l.numpy(), uu_l.numpy()], axis=-1) * (180.0 / np.pi)
    return flow_right, flow_left


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
    """Subtract rotation-induced flow, returning translation-only angular flow.

    The rotational flow is :func:`ego_motion.generate_optic_flow` with zero
    translation (depth-independent), at the right-eye columns and at their
    sagittal mirror (``phi -> -phi``) for the left eye. It is linear in
    ``omega``, so ``omega`` must be in the angular unit of the predicted flow.

    Args:
        pred_flow_right, pred_flow_left: (T, N, 2) predicted angular flow.
        omega: (T, 3) rotation rate.
        column_phi_theta: (N, 2) right-eye column (phi, theta) in radians.
    """
    import torch
    from drosophila_model.model_definition.ego_motion import generate_optic_flow

    # torch.tensor copies: column_phi_theta may be a read-only (parquet) array.
    phi_theta = torch.tensor(np.array(column_phi_theta), dtype=torch.float64)
    omega_t = torch.tensor(np.array(omega), dtype=torch.float64)
    no_translation = torch.zeros_like(omega_t)

    outputs = []
    for pred_flow, phi in ((pred_flow_right, phi_theta[:, 0]),
                           (pred_flow_left, -phi_theta[:, 0])):
        d_right, d_up = generate_optic_flow(phi, phi_theta[:, 1], omega_t,
                                            no_translation, 1.0)
        out = pred_flow.copy()
        out -= torch.stack([d_right, d_up], dim=-1).numpy()
        outputs.append(out)
    return outputs[0], outputs[1]


def _hex_polys_centered_to_extent(column_p_q, target_x, target_z):
    """
    Build hexagon polygons for ``column_p_q`` (the fly_eye hex-mosaic layout,
    see :func:`fly_eye.utils.column_hex_centers`), centered at the origin and
    scaled to match the extent spanned by ``target_x``/``target_z``. Returns
    ``(verts, scale)`` where ``verts`` is an ``(N, 6, 2)`` array suitable for
    ``matplotlib.collections.PolyCollection``.
    """
    x, y = column_hex_centers(column_p_q)
    x = x - x.mean()
    y = y - y.mean()

    target_span = max(np.ptp(target_x), np.ptp(target_z), 1e-9)
    raw_span = max(np.ptp(x), np.ptp(y), 1e-9)
    s = target_span / raw_span

    verts = hex_vertices(x * s, y * s, size=s)  # size is radius in pixel units
    return verts, s


def make_scatter_diamond_sequence_factory(responses_dict, cell_x, cell_z, offset=2.6, scale=1.0, size_scale=1.0, stim_cmap='viridis', stim_scale_frac=0.5, stim_size_frac=0.2, stim_p_q=None):
    """
    Diamond layout of four cell-type responses around the stimulus.
      - responses_dict: dict with 5 keys (in insertion order):
          ['<right>', '<left>', '<up>', '<down>', 'stimulus']
        Each value is a numpy array shaped [num_stimuli, num_time, num_columns].
      - The four non-'stimulus' keys are plotted simultaneously in a diamond:
          order (in insertion): Right at (+offset, 0), Left at (-offset, 0),
          Up at (0, +offset), Down at (0, -offset).
        'stimulus' is plotted at the center, scaled by ``stim_scale_frac`` with
        markers scaled by ``stim_size_frac``.
      - For each stimulus index, we render its full time series, followed by 25 blank frames.
      - If ``stim_p_q`` (an ``(N, 2)`` hex-grid coord array) is provided, the
        center stimulus is rendered as a hexagonal mosaic via the fly_eye hex
        layout instead of a scatter (cell panels remain scatters).
    The returned init callable conforms to make_movie: init(d, di) -> (fig, update(frame_t), num_frames).
    This factory concatenates ALL stimuli back-to-back into a single movie (with gaps),
    so the direction index ``di`` is ignored.
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

    return init_plot_for_direction


def make_movie(init_plot_for_direction, directions, fig_folder, frame_rate, save_name=''):
    """
    Write one movie per direction label to ``{fig_folder}/{save_name}_{direction}.mp4``.

    ``init_plot_for_direction(d, di) -> (fig, update(frame_t), num_frames)``,
    e.g. from :func:`make_scatter_diamond_sequence_factory`.
    """
    for di, d in enumerate(directions):
        fig, update, num_frames = init_plot_for_direction(d, di)
        write_movie(fig, update, num_frames, Path(fig_folder) / f"{save_name}_{d}.mp4",
                    frame_rate)
