"""
Shared utilities for analysis scripts.

Provides:
  - Path resolution (resolve_model_folder, resolve_model_paths)
  - Matplotlib helpers (fig_to_rgb, centered_norm)
  - Data prep (prepare_xt, interp_nan_column)
  - Hex grid (hex_grid_points)
  - Model prediction (predict_optic_flow, run_model_on_stimuli)
  - Movie creation (make_flow_comparison_movie)
"""
import argparse
import sys
from copy import deepcopy
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

import data_utilities as du
import neural_models as nm
import stimulus_utilities as su


# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------


def resolve_model_folder(model_path: str | Path) -> Path:
    """Return the model directory from a model path (file or folder)."""
    model_path = Path(model_path)
    if model_path.is_dir():
        return model_path
    return model_path.parent


def resolve_model_paths(model_path: str | Path) -> tuple[Path, Path]:
    """Return *(model_file, model_folder)* from a model path (file or folder).

    Raises ``FileNotFoundError`` if *model.pt* cannot be located.
    """
    model_path = Path(model_path)
    if model_path.is_file():
        return model_path, model_path.parent
    model_pt = model_path / "model.pt"
    if model_pt.exists():
        return model_pt, model_path
    raise FileNotFoundError(f"Could not find model.pt at {model_path}")


# ---------------------------------------------------------------------------
# Matplotlib helpers
# ---------------------------------------------------------------------------


def fig_to_rgb(fig) -> np.ndarray:
    """Convert a matplotlib figure to an RGB numpy array *(H, W, 3)*."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[..., :3].copy()


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
    model: nm.NeuralModel, stimulus: torch.Tensor
) -> torch.Tensor:
    """Run the model and return predicted optic flow."""
    with torch.no_grad():
        if model.use_decoder:
            outputs = model.predict(stimulus, input_neurons=["L1", "L2", "L3"])[0]
            return outputs["decoder"]
        return model.calculate_optic_flow(stimulus)[0]


def run_model_on_stimuli(
    model_path: str | Path = "models/best_model",
    device: str = "cpu",
    stimulus_types: list[str] | None = None,
    stimulus_specs: list[dict] | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Generate stimuli, run a model, and collect responses.

    Returns a dict keyed by stimulus name with values containing
    ``stimulus``, ``optic_flow``, ``pred_flow``, ``column_xyz``, and ``dt``.
    """
    model_pt, model_folder = resolve_model_paths(Path(model_path))
    config = du.load_config(model_folder / "config.toml")
    model = nm.NeuralModel.load(model_pt, device=device)

    column_phi_theta = model.column_phi_theta
    column_xyz = model.get_column_xyz()

    if stimulus_specs is None:
        if stimulus_types is None:
            stimulus_types = ["sintel", "flow360", "random_sine"]
        stimulus_specs = []
        stimulus_map = {
            "sintel": ("sintel", None),
            "flow360": ("flow_360", None),
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
        stimulus, optic_flow = su.get_stimulus(
            spec["stimulus_type"],
            stim_config,
            column_info,
            **stimulus_kwargs,
        )
        if stimulus.shape[0] > 1:
            stimulus, optic_flow = stimulus[1:], optic_flow[1:]

        pred_flow = predict_optic_flow(model, stimulus)
        responses[spec["name"]] = {
            "stimulus": stimulus,
            "optic_flow": optic_flow,
            "pred_flow": pred_flow,
            "column_xyz": torch.tensor(
                column_xyz, device=model.device, dtype=model.dtype
            ),
            "dt": torch.tensor(model.dt, device=model.device, dtype=model.dtype),
        }

    return responses


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
) -> None:
    """Create a flow comparison movie.

    When *pred_flow* is provided, creates a 2x2 layout (stimulus, true flow,
    blank, predicted flow).  When *pred_flow* is ``None``, creates a 1x2
    layout (stimulus, true flow).

    Parameters
    ----------
    stimulus : (time, columns)
    true_flow : (time, columns, 2) — angular flow ``(u_right, u_up)``
    pred_flow : (time, columns, 2) or None
    cell_x, cell_y, cell_z : (columns,) Cartesian coords on unit sphere.
        When *cell_y* is provided the angular flow is properly projected to
        Cartesian ``(v_x, v_z)`` for the quiver arrows.
    output_path : Path for .mp4 output
    frame_rate : frames per second
    stim_name : label shown in the stimulus panel title
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import matplotlib.colors as mcolors
    import imageio as iio_v2

    num_time, _ = stimulus.shape
    stim_min, stim_max = np.nanmin(stimulus), np.nanmax(stimulus)
    stim_norm = mcolors.Normalize(vmin=stim_min, vmax=stim_max)

    # Convert angular flow (u_right, u_up) to Cartesian (v_x, v_z)
    if cell_y is not None:
        true_flow_vx, true_flow_vz = angular_flow_to_cartesian_xz(
            true_flow[:, :, 0], true_flow[:, :, 1], cell_x, cell_y, cell_z,
        )
    else:
        # Legacy fallback (incorrect for non-trivial geometries)
        true_flow_vx = -true_flow[:, :, 0]
        true_flow_vz = true_flow[:, :, 1]

    true_magnitude = np.hypot(true_flow_vx, true_flow_vz)
    max_mag = (
        np.nanpercentile(true_magnitude, 99) if np.any(true_magnitude) else 1.0
    )
    flow_norm = mcolors.Normalize(vmin=0.0, vmax=max_mag)

    has_pred = pred_flow is not None
    if has_pred:
        if cell_y is not None:
            pred_flow_vx, pred_flow_vz = angular_flow_to_cartesian_xz(
                pred_flow[:, :, 0], pred_flow[:, :, 1], cell_x, cell_y, cell_z,
            )
        else:
            pred_flow_vx = -pred_flow[:, :, 0]
            pred_flow_vz = pred_flow[:, :, 1]
        fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.24), dpi=100)
        ax_stim, ax_true = axes[0, 0], axes[0, 1]
        axes[1, 0].axis("off")
        axes[1, 0].set_title("(reserved)")
        ax_pred = axes[1, 1]
    else:
        fig, (ax_stim, ax_true) = plt.subplots(
            1, 2, figsize=(12.8, 5.12), dpi=100
        )

    x_margin = 0.1 * (cell_x.max() - cell_x.min())
    z_margin = 0.1 * (cell_z.max() - cell_z.min())

    plot_axes = [ax_stim, ax_true] + ([ax_pred] if has_pred else [])
    for ax in plot_axes:
        ax.set_xlim(cell_x.max() + x_margin, cell_x.min() - x_margin)
        ax.set_ylim(cell_z.min() - z_margin, cell_z.max() + z_margin)
        ax.set_aspect("equal")
        ax.set_xlabel("x (anterior ← → posterior)")
        ax.set_ylabel("z (ventral ↔ dorsal)")

    sc_stim = ax_stim.scatter(
        cell_x,
        cell_z,
        c=stimulus[0, :],
        cmap="gray",
        norm=stim_norm,
        s=90,
        edgecolors="none",
    )
    title_label = f"Stimulus ({stim_name})" if stim_name else "Stimulus"
    ax_stim.set_title(title_label)
    fig.colorbar(sc_stim, ax=ax_stim, label="intensity")

    arrow_scale = 0.2 / max_mag if max_mag > 0 else 1.0

    q_true = ax_true.quiver(
        cell_x,
        cell_z,
        true_flow_vx[0, :],
        true_flow_vz[0, :],
        true_magnitude[0, :],
        cmap="viridis",
        norm=flow_norm,
        angles="xy",
        scale_units="xy",
        scale=1 / arrow_scale,
        width=0.003,
    )
    ax_true.set_title("True Optic Flow")
    fig.colorbar(q_true, ax=ax_true, label="magnitude")

    if has_pred:
        pred_magnitude = np.hypot(pred_flow_vx, pred_flow_vz)
        q_pred = ax_pred.quiver(
            cell_x,
            cell_z,
            pred_flow_vx[0, :],
            pred_flow_vz[0, :],
            pred_magnitude[0, :],
            cmap="viridis",
            norm=flow_norm,
            angles="xy",
            scale_units="xy",
            scale=1 / arrow_scale,
            width=0.003,
        )
        ax_pred.set_title("Predicted Optic Flow")
        fig.colorbar(q_pred, ax=ax_pred, label="magnitude")

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing movie to {output_path}...")

    with iio_v2.get_writer(
        str(output_path), codec="libx264", fps=frame_rate
    ) as writer:
        for t in range(num_time):
            sc_stim.set_array(stimulus[t, :])

            vx_true_t = true_flow_vx[t, :]
            vz_true_t = true_flow_vz[t, :]
            mag_true_t = np.hypot(vx_true_t, vz_true_t)
            q_true.set_UVC(vx_true_t, vz_true_t, mag_true_t)

            if has_pred:
                vx_pred_t = pred_flow_vx[t, :]
                vz_pred_t = pred_flow_vz[t, :]
                mag_pred_t = np.hypot(vx_pred_t, vz_pred_t)
                q_pred.set_UVC(vx_pred_t, vz_pred_t, mag_pred_t)

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
) -> None:
    """Create a 1x2 movie: stimulus intensity + true optic flow.

    Thin wrapper around :func:`make_flow_comparison_movie` with no predicted
    flow panel.
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
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze RNN optic flow predictions."
    )
    parser.add_argument(
        "model_path",
        type=str,
        nargs="?",
        default="models/best_model",
        help="Path to model folder or model.pt (default: models/best_model)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the model on (default: cpu)",
    )
    args = parser.parse_args()

    responses = run_model_on_stimuli(args.model_path, device=args.device)
    print(f"Loaded responses for: {', '.join(responses.keys())}")


if __name__ == "__main__":
    main()
