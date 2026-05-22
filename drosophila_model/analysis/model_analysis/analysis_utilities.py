"""
Shared utilities for analysis scripts.

Provides:
  - Matplotlib helpers (fig_to_rgb, centered_norm)
  - Data prep (prepare_xt, input_conv_stimulus_timecourse, interp_nan_column)
  - Hex grid (hex_grid_points)
  - Model prediction (predict_optic_flow, run_model_on_stimuli)
  - Movie creation (make_flow_comparison_movie)
"""
import argparse
import sys
from copy import deepcopy
from pathlib import Path

# Add parent directory to path for imports

import numpy as np
import pandas as pd
import torch

from drosophila_model.config import load_config
from drosophila_model.model_definition import neural_models as nm
from drosophila_model.utilities import stimulus_utilities as su


def get_flow_key(model):
    """Return the output key for the first flow (eye-asymmetric) readout."""
    return next(
        (k for k, m in model._readout_modules.items() if m.predicts_flow),
        None,
    )


# ---------------------------------------------------------------------------
# Matplotlib helpers
# ---------------------------------------------------------------------------


def fig_to_rgb(fig) -> np.ndarray:
    """Convert a matplotlib figure to an RGB numpy array *(H, W, 3)*."""
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return rgba[..., :3].copy()


def centered_norm(data: np.ndarray) -> tuple[float, float]:
    """Return symmetric *(vmin, vmax)* centered around zero."""
    max_abs = float(np.max(np.abs(data)))
    if max_abs == 0:
        max_abs = 1.0
    return -max_abs, max_abs


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------


def prepare_xt(
    activity: np.ndarray, phi: np.ndarray
) -> tuple[np.ndarray, list[float]]:
    """Sort columns by *phi* and return *(sorted_activity, extent)* for XT plots.

    Columns containing any NaN are dropped.
    """
    valid_cols = np.isfinite(activity).all(axis=0)
    if not np.any(valid_cols):
        raise ValueError("No valid columns found for XT plot.")
    activity_valid = activity[:, valid_cols]
    phi_valid = phi[valid_cols]
    sort_inds = np.argsort(phi_valid)
    activity_sorted = activity_valid[:, sort_inds]
    phi_sorted = phi_valid[sort_inds]
    extent = [
        float(phi_sorted.min()),
        float(phi_sorted.max()),
        0,
        activity_sorted.shape[0],
    ]
    return activity_sorted, extent


def input_conv_stimulus_timecourse(model, inputs: torch.Tensor) -> torch.Tensor:
    """Apply :meth:`~neural_models.BaseRnn.apply_input_conv` at each time step.

    Matches the input path inside :meth:`~neural_models.BaseRnn.predict` when
    ``use_input_conv`` is True (spatial pre-filter, local contrast norm, temporal LRU).

    If ``model.regions['optic_lobe'].use_input_conv`` is False, returns *inputs* unchanged.

    Parameters
    ----------
    model
        A model with optional ``use_input_conv`` and ``apply_input_conv``.
    inputs
        Shape *(batch, time, num_columns)* on the model device/dtype.

    Returns
    -------
    torch.Tensor
        Same shape as *inputs* — drive fed to ``forward`` after filtering each step.
    """
    if not getattr(model.regions['optic_lobe'], "use_input_conv", False):
        return inputs
    if inputs.ndim != 3:
        raise ValueError(f"expected inputs (B, T, C), got shape {tuple(inputs.shape)}")
    b, t_steps, c = inputs.shape
    out = torch.empty_like(inputs)
    state = None
    with torch.no_grad():
        for t in range(t_steps):
            y, state = model.regions['optic_lobe'].apply_input_conv(inputs[:, t, :], state)
            out[:, t, :] = y
    return out


def interp_nan_column(
    time_by_neurons: np.ndarray,
    cell_phi: np.ndarray,
    cell_theta: np.ndarray,
    distance_deg: float = 10.0,
) -> np.ndarray:
    """Interpolate NaN columns from nearby non-NaN neighbours.

    Parameters
    ----------
    time_by_neurons : array, shape ``(time, neurons)`` or ``(stim, time, neurons)``
        If 2-D, a leading singleton dimension is added temporarily.
    cell_phi, cell_theta : 1-D arrays of angular positions (radians).
    distance_deg : angular neighbourhood radius in degrees.

    Returns
    -------
    Interpolated array with the same shape as the input.
    """
    squeeze = False
    if time_by_neurons.ndim == 2:
        time_by_neurons = time_by_neurons[None, :, :]
        squeeze = True

    distance = distance_deg * np.pi / 180
    num_neurons = time_by_neurons.shape[2]
    nan_neurons = np.any(np.isnan(time_by_neurons), axis=(0, 1))

    for i in range(num_neurons):
        if nan_neurons[i]:
            close_phi = np.abs(cell_phi - cell_phi[i]) < distance
            close_theta = np.abs(cell_theta - cell_theta[i]) < distance
            close_neurons = close_phi & close_theta & ~nan_neurons
            time_by_neurons[:, :, i] = np.mean(
                time_by_neurons[:, :, close_neurons], axis=-1
            )

    if squeeze:
        time_by_neurons = time_by_neurons[0]
    return time_by_neurons


# ---------------------------------------------------------------------------
# Hex grid helpers
# ---------------------------------------------------------------------------


def hex_grid_points(axial_coords: np.ndarray) -> np.ndarray:
    """Convert axial *(q, r)* hex coordinates to flat-top *(x, y)* points."""
    q = axial_coords[:, 0].astype(np.float64)
    r = axial_coords[:, 1].astype(np.float64)
    grid_x = 1.5 * q
    grid_y = np.sqrt(3.0) * (r + 0.5 * q)
    return np.stack([grid_x, grid_y], axis=1)


# ---------------------------------------------------------------------------
# Model prediction helpers
# ---------------------------------------------------------------------------


def _apply_config_overrides(config: dict, overrides: dict | None) -> dict:
    if not overrides:
        return config
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key] = {**config[key], **value}
        else:
            config[key] = value
    return config


def predict_optic_flow(
    model: nm.BaseRnn, stimulus: torch.Tensor
) -> torch.Tensor:
    """Run the model and return predicted optic flow."""
    flow_key = get_flow_key(model)
    if flow_key is None:
        raise ValueError("Model has no flow readout configured. Cannot predict optic flow.")
    with torch.no_grad():
        outputs = model.predict(stimulus, input_neurons=["L1", "L2", "L3"])['outputs']['optic_lobe']
        return outputs[flow_key]


def run_model_on_stimuli(
    model_path: str | Path = "models/best_model",
    device: str | None = None,
    stimulus_types: list[str] | None = None,
    stimulus_specs: list[dict] | None = None,
    max_stimuli: int | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Generate stimuli, run a model, and collect responses.

    Parameters
    ----------
    max_stimuli : int or None
        If given, keep at most this many stimuli per stimulus type.

    Returns a dict keyed by stimulus name with values containing
    ``stimulus``, ``optic_flow``, ``pred_flow``, ``column_xyz``, and ``dt``.
    """
    model_path = Path(model_path)
    model_folder = model_path.parent if model_path.is_file() else model_path
    config = load_config(model_folder / "config.toml")
    model = nm.BaseRnn.load(model_path, device=device)

    column_phi_theta = model.regions['optic_lobe'].column_phi_theta
    column_xyz = model.regions['optic_lobe'].get_column_xyz()

    if stimulus_specs is None:
        if stimulus_types is None:
            stimulus_types = ["sintel", "flow360", "flyview", "random_sine"]
        stimulus_specs = []
        stimulus_map = {
            "sintel": ("sintel", None),
            "flow360": ("flow_360", None),
            "flyview": ("flyview", None),
            "random_sine": ("random_sine", "column_phi_theta"),
        }
        for stim_name in stimulus_types:
            if stim_name not in stimulus_map:
                raise ValueError(f"Unsupported stimulus type: {stim_name}")
            stim_type, column_info = stimulus_map[stim_name]
            stimulus_specs.append(
                {
                    "name": stim_name,
                    "stimulus_type": stim_type,
                    "column_info": column_info,
                }
            )

    responses: dict[str, dict[str, torch.Tensor]] = {}
    for spec in stimulus_specs:
        stim_config = deepcopy(config)
        stim_config["model"]["device"] = str(model.device)
        stim_config["stimulus"]["pre_stim_time"] = 0.0
        stim_config = _apply_config_overrides(
            stim_config, spec.get("config_overrides")
        )
        column_info = spec.get("column_info")
        if column_info == "column_phi_theta":
            column_info = column_phi_theta
        stimulus_kwargs = spec.get("stimulus_kwargs", {})
        stimulus, optic_flow, _ = su.get_stimulus_one_eye(
            spec["stimulus_type"],
            stim_config,
            column_info,
            **stimulus_kwargs,
        )
        if stimulus.shape[0] > 1:
            stimulus, optic_flow = stimulus[1:], optic_flow[1:]

        if max_stimuli is not None and stimulus.shape[0] > max_stimuli:
            stimulus = stimulus[:max_stimuli]
            optic_flow = optic_flow[:max_stimuli]

        pred_flow = predict_optic_flow(model, stimulus)
        entry: dict[str, torch.Tensor] = {
            "stimulus": stimulus,
            "optic_flow": optic_flow,
            "pred_flow": pred_flow,
            "column_xyz": torch.tensor(
                column_xyz, device=model.device, dtype=model.dtype
            ),
            "dt": torch.tensor(model.dt, device=model.device, dtype=model.dtype),
        }

        if model.regions['optic_lobe'].use_input_conv:
            with torch.no_grad():
                stim_dev = stimulus.to(device=model.device, dtype=model.dtype)
                filtered_frames = []
                for si in range(stim_dev.shape[0]):
                    filter_state = None
                    for t in range(stim_dev.shape[1]):
                        y_t, filter_state = model.regions['optic_lobe'].apply_input_conv(
                            stim_dev[si : si + 1, t, :], filter_state
                        )
                        filtered_frames.append(y_t)
                entry["filtered_stimulus"] = torch.stack(filtered_frames).reshape(
                    stim_dev.shape
                ).cpu()

        responses[spec["name"]] = entry

    return responses


# NOTE: `generate_optic_flow`, `load_viewing_directions`, and the
# `cross_matrix` helper formerly lived here. They moved to
# `drosophila_model.utilities.ego_motion` (canonical home — sister to
# `estimate_from_flow`). Import them from there.


# ---------------------------------------------------------------------------
# Flow coordinate helpers
# ---------------------------------------------------------------------------


def angular_flow_to_cartesian_xz(
    flow_u: np.ndarray,
    flow_v: np.ndarray,
    cell_x: np.ndarray,
    cell_y: np.ndarray,
    cell_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert angular optic flow ``(u_right, u_up)`` to Cartesian ``(v_x, v_z)``.

    The angular flow lives in the local tangent frame on the sphere:

        v_3d = -u_right * e_phi - u_up * e_theta

    Projecting onto the x-z plotting plane gives:

        v_x = u_right * sin(phi) - u_up * cos(theta) * cos(phi)
        v_z = u_up * sin(theta)

    Parameters
    ----------
    flow_u : (..., N) rightward angular flow component
    flow_v : (..., N) upward angular flow component
    cell_x, cell_y, cell_z : (N,) Cartesian coordinates on the unit sphere
    """
    sin_theta = np.sqrt(np.maximum(cell_x ** 2 + cell_y ** 2, 1e-12))
    cos_theta = cell_z
    # Avoid division by zero at the poles
    sin_phi = np.where(sin_theta > 1e-8, cell_y / sin_theta, 0.0)
    cos_phi = np.where(sin_theta > 1e-8, cell_x / sin_theta, 1.0)

    v_x = flow_u * sin_phi - flow_v * cos_theta * cos_phi
    v_z = flow_v * sin_theta

    return v_x, v_z


def flow_direction_colors(u, v, max_mag=None, flip_lr=False):
    """RGBA colors encoding direction as hue and amplitude as saturation.

    Parameters
    ----------
    u, v : array-like, same shape
        Cartesian flow components (e.g. vx, vz from quiver).
    max_mag : float or None
        Saturation normalisation ceiling.  ``None`` → 99th percentile.
    flip_lr : bool
        If True, map hue as if ``u`` were negated. Use when the panel's x-axis
        is NOT horizontally flipped but the shared colorwheel was drawn with
        ``flip_lr=True`` (or vice versa), so that on-screen arrow direction
        agrees with the colorwheel legend.

    Returns
    -------
    colors : (*shape, 4) RGBA array
    max_mag : float — the normalisation value actually used
    """
    from matplotlib.colors import hsv_to_rgb

    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    u_for_hue = -u if flip_lr else u
    angle = np.arctan2(v, u_for_hue)
    hue = (angle / (2 * np.pi)) % 1.0

    mag = np.hypot(u, v)
    if max_mag is None:
        finite = mag[np.isfinite(mag)]
        max_mag = float(np.percentile(finite, 99)) if finite.size > 0 and finite.max() > 0 else 1.0
    sat = np.clip(mag / max(max_mag, 1e-9), 0, 1)

    hsv = np.stack([hue, sat, np.ones_like(hue)], axis=-1)
    rgb = hsv_to_rgb(hsv)
    return np.concatenate([rgb, np.ones(rgb.shape[:-1] + (1,))], axis=-1), max_mag


def update_direction_quiver(q, u, v, max_mag, flip_lr=False):
    """Update a direction-colored quiver's geometry and colors.

    ``flip_lr`` only affects the color hue mapping, not arrow geometry.
    """
    q.set_UVC(u, v)
    colors, _ = flow_direction_colors(u, v, max_mag, flip_lr=flip_lr)
    q.set_facecolor(colors)
    q.set_edgecolor(colors)


def add_direction_colorwheel(fig, rect=None, flip_lr=False):
    """Add a small HSV colour wheel showing the direction-to-hue mapping.

    Parameters
    ----------
    flip_lr : bool
        If True, mirror the hue pattern left–right (equivalently θ → π − θ in hue
        space) and swap east/west tick glyphs. Use when flow panels use a
        horizontally flipped x-axis (e.g. ``setup_eye_axes`` right eye) so the
        legend matches on-screen arrow directions.
    """
    from matplotlib.colors import hsv_to_rgb

    if rect is None:
        rect = [0.87, 0.87, 0.11, 0.11]
    ax_cw = fig.add_axes(rect, projection='polar')
    n = 256
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    width = 2 * np.pi / n
    if flip_lr:
        hues = np.mod((np.pi - theta) / (2 * np.pi), 1.0)
        tick_labels = ['\u2190', '\u2191', '\u2192', '\u2193']  # ← ↑ → ↓
    else:
        hues = (theta / (2 * np.pi)) % 1.0
        tick_labels = ['\u2192', '\u2191', '\u2190', '\u2193']  # → ↑ ← ↓
    colors = hsv_to_rgb(np.stack([hues, np.ones(n), np.ones(n)], axis=-1))
    ax_cw.bar(theta, 1, width=width, bottom=0.0, color=colors,
              edgecolor='none', linewidth=0)
    ax_cw.set_yticks([])
    ax_cw.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    ax_cw.set_xticklabels(tick_labels, fontsize=8)
    ax_cw.tick_params(pad=1)
    ax_cw.grid(False)
    ax_cw.spines['polar'].set_visible(False)
    return ax_cw


# ---------------------------------------------------------------------------
# Eye-view helpers (single source of truth for per-eye flow rendering)
# ---------------------------------------------------------------------------
#
# Physical model
# --------------
# The figure observer sits at the origin of the fly's body frame and looks
# toward the eye being shown:
#
#   Right eye panel: observer looks toward -Y. Screen-right = -X (posterior),
#       screen-left = +X (anterior). Implementation: x-axis xlim is flipped;
#       cell positions used for projection are right-eye (cell_x, cell_y, cell_z).
#   Left eye panel:  observer looks toward +Y. Screen-right = +X (anterior),
#       screen-left  = -X (posterior). Implementation: normal x-axis; cell
#       positions used for projection are left-eye (cell_x, -cell_y, cell_z),
#       because the left eye's columns physically live at phi_l = -phi_r.
#
# Color hue (screen-relative): in every panel hue 0 ("→") means the arrow
# points right ON SCREEN. Because the right-eye axis is flipped, the right-eye
# color call uses ``flip_lr=True`` and the left-eye uses ``flip_lr=False``.
# A single shared "→ ↑ ← ↓" wheel (``flip_lr=False``) per figure suffices.
# ---------------------------------------------------------------------------


_VALID_EYES = ("right", "left")


def _check_eye(eye):
    if eye not in _VALID_EYES:
        raise ValueError(f"eye must be one of {_VALID_EYES}, got {eye!r}")


def eye_cell_y(cell_y, eye):
    """Return the eye-correct ``cell_y`` for projecting flow.

    Right eye: ``cell_y`` unchanged. Left eye: ``-cell_y`` (columns mirrored
    across the y axis since ``phi_l = -phi_r``).
    """
    _check_eye(eye)
    return cell_y if eye == "right" else -cell_y


def eye_color_flip(eye):
    """Whether color hue should be flipped left/right for this eye panel.

    True for the right eye (whose x-axis is flipped on screen), False for the
    left eye. Use as the ``flip_lr`` argument to
    :func:`flow_direction_colors` / :func:`update_direction_quiver`.
    """
    _check_eye(eye)
    return eye == "right"


def project_eye_flow(angular_flow, cell_x, cell_y, cell_z, eye):
    """Project an eye's angular flow ``(u_right, u_up)`` to Cartesian ``(v_x, v_z)``.

    Wraps :func:`angular_flow_to_cartesian_xz` and applies the eye-correct
    ``cell_y`` mirroring so callers never need to remember the convention.

    Parameters
    ----------
    angular_flow : (..., N, 2) array — channel 0 is u_right, channel 1 is u_up.
    cell_x, cell_y, cell_z : (N,) right-eye Cartesian coordinates on the unit
        sphere (as returned by ``model.regions['optic_lobe'].get_column_xyz()``).
    eye : 'right' or 'left'.
    """
    _check_eye(eye)
    return angular_flow_to_cartesian_xz(
        angular_flow[..., 0], angular_flow[..., 1],
        cell_x, eye_cell_y(cell_y, eye), cell_z,
    )


def prepare_eye_quiver(angular_flow, cell_x, cell_y, cell_z, eye):
    """Project flow for an eye and return everything needed for quiver rendering.

    Returns
    -------
    vx, vz : (T, N) Cartesian flow components, NaN replaced with 0.
    mag : (T, N) magnitude, NaN replaced with 0.
    flow_norm : matplotlib Normalize for colormaps (0 .. 99th-percentile mag).
    arrow_scale : float, suitable for ``quiver(scale=1/arrow_scale, ...)``.
    """
    import matplotlib.colors as mcolors

    vx, vz = project_eye_flow(angular_flow, cell_x, cell_y, cell_z, eye)
    mag = np.hypot(vx, vz)
    finite_mag = mag[np.isfinite(mag)]
    max_mag = (float(np.percentile(finite_mag, 99))
               if finite_mag.size > 0 and finite_mag.max() > 0 else 1.0)
    flow_norm = mcolors.Normalize(vmin=0.0, vmax=max_mag)
    arrow_scale = 0.2 / max_mag if max_mag > 0 else 1.0

    vx = np.nan_to_num(vx, nan=0.0)
    vz = np.nan_to_num(vz, nan=0.0)
    mag = np.nan_to_num(mag, nan=0.0)
    return vx, vz, mag, flow_norm, arrow_scale


def setup_eye_axis(ax, cell_x, cell_z, eye, x_margin=None, z_margin=None):
    """Configure a single eye panel's xlim/ylim/aspect.

    The right eye gets a flipped x-axis so screen-right = -X (posterior);
    the left eye uses a normal x-axis so screen-right = +X (anterior).
    """
    _check_eye(eye)
    if x_margin is None:
        x_margin = 0.1 * (cell_x.max() - cell_x.min())
    if z_margin is None:
        z_margin = 0.1 * (cell_z.max() - cell_z.min())
    if eye == "right":
        ax.set_xlim(cell_x.max() + x_margin, cell_x.min() - x_margin)
    else:
        ax.set_xlim(cell_x.min() - x_margin, cell_x.max() + x_margin)
    ax.set_ylim(cell_z.min() - z_margin, cell_z.max() + z_margin)
    ax.set_aspect("equal")


def setup_eye_axes(ax_left, ax_right, cell_x, cell_z, x_margin=None, z_margin=None):
    """Configure a paired (left, right) eye panel. Thin wrapper around
    :func:`setup_eye_axis`."""
    setup_eye_axis(ax_left, cell_x, cell_z, "left",
                   x_margin=x_margin, z_margin=z_margin)
    setup_eye_axis(ax_right, cell_x, cell_z, "right",
                   x_margin=x_margin, z_margin=z_margin)


def eye_quiver_colors(vx, vz, max_mag, eye):
    """Color hues for an eye panel's quiver, screen-direction-consistent."""
    return flow_direction_colors(vx, vz, max_mag, flip_lr=eye_color_flip(eye))


def init_eye_quiver(ax, cell_x, cell_z, vx0, vz0, max_mag, eye, **quiver_kw):
    """Create a quiver on ``ax`` with eye-correct color hues.

    ``quiver_kw`` is forwarded to ``ax.quiver`` (e.g. ``angles``,
    ``scale_units``, ``scale``, ``width``).
    """
    q = ax.quiver(cell_x, cell_z, vx0, vz0, **quiver_kw)
    colors, _ = eye_quiver_colors(vx0, vz0, max_mag, eye)
    q.set_facecolor(colors)
    q.set_edgecolor(colors)
    return q


def update_eye_quiver(q, vx, vz, max_mag, eye):
    """Update a quiver's geometry and eye-correct colors."""
    update_direction_quiver(q, vx, vz, max_mag, flip_lr=eye_color_flip(eye))


def add_eye_colorwheel(fig, rect=None):
    """Add the standard screen-direction colorwheel ("→ ↑ ← ↓") for eye panels.

    One wheel per figure is sufficient because :func:`eye_quiver_colors`
    already rotates the per-eye hue mapping so that "→" always points right
    on screen, regardless of which eye panel you're looking at.
    """
    return add_direction_colorwheel(fig, rect=rect, flip_lr=False)


# ---------------------------------------------------------------------------
# Movie creation helpers
# ---------------------------------------------------------------------------


def make_flow_comparison_movie(
    stimulus: np.ndarray,
    true_flow: np.ndarray,
    pred_flow: np.ndarray | None = None,
    cell_x: np.ndarray = None,
    cell_y: np.ndarray = None,
    cell_z: np.ndarray = None,
    output_path: Path = None,
    frame_rate: int = 30,
    stim_name: str = "",
    filtered_stimulus: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    eye: str = "right",
) -> None:
    """Create a single-eye flow comparison movie.

    When *pred_flow* is provided, creates a 2x2 layout (stimulus, true flow,
    filtered stimulus, predicted flow).  When *pred_flow* is ``None``, creates
    a 1x2 layout (stimulus, true flow).

    Parameters
    ----------
    stimulus : (time, columns)
    true_flow : (time, columns, 2) — angular flow ``(u_right, u_up)``
    pred_flow : (time, columns, 2) or None
    cell_x, cell_y, cell_z : (columns,) right-eye Cartesian coordinates on the
        unit sphere (as returned by ``model.regions['optic_lobe'].get_column_xyz()``). The eye-correct
        ``cell_y`` mirroring is handled internally based on ``eye``.
    output_path : Path for .mp4 output
    frame_rate : frames per second
    stim_name : label shown in the stimulus panel title
    filtered_stimulus : (time, columns) or None — stimulus after the model's
        input filter.  Shown in the bottom-left panel when provided.
    mask : (time, columns) bool or None — True = valid, False = masked
        (shown yellow).
    eye : 'right' or 'left' — which eye this single-eye view represents.
        Controls the axis flip and color hue convention.
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import matplotlib.colors as mcolors
    import imageio as iio_v2

    num_time, _ = stimulus.shape
    stim_min, stim_max = np.nanmin(stimulus), np.nanmax(stimulus)
    stim_norm = mcolors.Normalize(vmin=stim_min, vmax=stim_max)

    _MASK_COLOR = np.array([1.0, 1.0, 0.0, 1.0])
    gray_cmap = plt.cm.gray

    def _masked_rgba(values, mask_t, cmap, norm):
        rgba = cmap(norm(values))
        if mask_t is not None:
            rgba[~mask_t] = _MASK_COLOR
        return rgba

    true_flow_vx, true_flow_vz, _, true_norm, true_arrow_scale = prepare_eye_quiver(
        true_flow, cell_x, cell_y, cell_z, eye)
    true_max_mag = true_norm.vmax

    has_pred = pred_flow is not None
    has_filtered = filtered_stimulus is not None
    if has_pred:
        pred_flow_vx, pred_flow_vz, _, pred_norm, pred_arrow_scale = prepare_eye_quiver(
            pred_flow, cell_x, cell_y, cell_z, eye)
        pred_max_mag = pred_norm.vmax
        fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.24), dpi=100)
        ax_stim, ax_true = axes[0, 0], axes[0, 1]
        ax_bottom_left = axes[1, 0]
        ax_pred = axes[1, 1]
    else:
        fig, (ax_stim, ax_true) = plt.subplots(
            1, 2, figsize=(12.8, 5.12), dpi=100
        )

    if has_pred and has_filtered:
        filt_min, filt_max = np.nanmin(filtered_stimulus), np.nanmax(filtered_stimulus)
        filt_norm = mcolors.Normalize(vmin=filt_min, vmax=filt_max)
    elif has_pred:
        ax_bottom_left.axis("off")
        ax_bottom_left.set_title("(reserved)")

    plot_axes = [ax_stim, ax_true] + ([ax_pred] if has_pred else [])
    if has_pred and has_filtered:
        plot_axes.append(ax_bottom_left)
    # Anatomical x-axis label depends on the eye (axis-flip orientation).
    if eye == "right":
        xlabel = "x (anterior ← → posterior)"
    else:
        xlabel = "x (posterior ← → anterior)"
    for ax in plot_axes:
        setup_eye_axis(ax, cell_x, cell_z, eye)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("z (ventral ↔ dorsal)")

    mask_0 = mask[0] if mask is not None else None
    sc_stim = ax_stim.scatter(
        cell_x,
        cell_z,
        c=_masked_rgba(stimulus[0, :], mask_0, gray_cmap, stim_norm),
        s=90,
        edgecolors="none",
    )
    title_label = f"Stimulus ({stim_name})" if stim_name else "Stimulus"
    ax_stim.set_title(title_label)
    fig.colorbar(sc_stim, ax=ax_stim, label="intensity")

    sc_filt = None
    if has_pred and has_filtered:
        sc_filt = ax_bottom_left.scatter(
            cell_x,
            cell_z,
            c=_masked_rgba(filtered_stimulus[0, :], mask_0, gray_cmap, filt_norm),
            s=90,
            edgecolors="none",
        )
        ax_bottom_left.set_title("Filtered Stimulus")
        fig.colorbar(sc_filt, ax=ax_bottom_left, label="intensity")

    quiver_kw = dict(angles="xy", scale_units="xy", width=0.003)
    q_true = init_eye_quiver(
        ax_true, cell_x, cell_z,
        true_flow_vx[0, :], true_flow_vz[0, :], true_max_mag, eye,
        scale=1 / true_arrow_scale, **quiver_kw,
    )
    ax_true.set_title("True Optic Flow")

    if has_pred:
        q_pred = init_eye_quiver(
            ax_pred, cell_x, cell_z,
            pred_flow_vx[0, :], pred_flow_vz[0, :], pred_max_mag, eye,
            scale=1 / pred_arrow_scale, **quiver_kw,
        )
        ax_pred.set_title("Predicted Optic Flow")

    add_eye_colorwheel(fig)

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing movie to {output_path}...")

    with iio_v2.get_writer(
        str(output_path), codec="libx264", fps=frame_rate
    ) as writer:
        for t in range(num_time):
            mask_t = mask[t] if mask is not None else None
            sc_stim.set_facecolors(_masked_rgba(stimulus[t, :], mask_t, gray_cmap, stim_norm))
            if sc_filt is not None:
                sc_filt.set_facecolors(_masked_rgba(filtered_stimulus[t, :], mask_t, gray_cmap, filt_norm))

            update_eye_quiver(q_true, true_flow_vx[t, :], true_flow_vz[t, :],
                              true_max_mag, eye)

            if has_pred:
                update_eye_quiver(q_pred, pred_flow_vx[t, :], pred_flow_vz[t, :],
                                  pred_max_mag, eye)

            frame_title = (
                f"Stimulus ({stim_name}) frame {t+1}/{num_time}"
                if stim_name
                else f"Stimulus (frame {t+1}/{num_time})"
            )
            ax_stim.set_title(frame_title)
            writer.append_data(fig_to_rgb(fig))

            if (t + 1) % 20 == 0:
                print(f"  Frame {t+1}/{num_time}")

    plt.close(fig)
    print(f"Done! Movie saved to {output_path}")


def make_stimulus_flow_movie(
    stimulus: np.ndarray,
    true_flow: np.ndarray,
    cell_x: np.ndarray,
    cell_z: np.ndarray,
    output_path: Path,
    frame_rate: int,
    cell_y: np.ndarray = None,
    mask: np.ndarray | None = None,
    eye: str = "right",
) -> None:
    """Create a 1x2 movie: stimulus intensity + true optic flow.

    Thin wrapper around :func:`make_flow_comparison_movie` with no predicted
    flow panel. ``eye`` selects the per-eye axis flip / color hue convention.
    """
    make_flow_comparison_movie(
        stimulus=stimulus,
        true_flow=true_flow,
        pred_flow=None,
        cell_x=cell_x,
        cell_y=cell_y,
        cell_z=cell_z,
        output_path=output_path,
        frame_rate=frame_rate,
        mask=mask,
        eye=eye,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze RNN optic flow predictions."
    )
    parser.add_argument(
        "model_folder",
        type=str,
        help="Path to model folder or model.pt",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the model on (default: cpu)",
    )
    args = parser.parse_args()

    responses = run_model_on_stimuli(args.model_folder, device=args.device)
    print(f"Loaded responses for: {', '.join(responses.keys())}")


if __name__ == "__main__":
    main()
