"""
Create movies comparing true vs predicted optic flow for each stimulus type.
2x2 layout:
  - Top left: Stimulus intensity (scatter plot)
  - Top right: True optic flow (quiver plot)
  - Bottom left: Blank (reserved)
  - Bottom right: Predicted optic flow (quiver plot)
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import imageio as iio_v2

from analysis_utilities import run_model_on_stimuli
from loss_functions import cos_loss


def _resolve_model_folder(model_path: Path) -> Path:
    if model_path.is_dir():
        return model_path
    return model_path.parent


def fig_to_rgb(fig):
    """Convert matplotlib figure to RGB array."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[..., :3].copy()


def _make_movie_for_stim(
    stim_name: str,
    response: dict,
    output_folder: Path,
    stim_index: int,
) -> None:
    stimulus = response["stimulus"].detach().cpu().numpy()
    true_flow = response["optic_flow"].detach().cpu().numpy()
    pred_flow = response["pred_flow"].detach().cpu().numpy()
    column_xyz = response["column_xyz"].detach().cpu().numpy()
    dt = float(response["dt"].item())

    num_stimuli = stimulus.shape[0]
    if num_stimuli == 0:
        print(f"Skipping {stim_name}: no stimuli available.")
        return
    if stim_index >= num_stimuli:
        stim_index = num_stimuli - 1

    frame_rate = max(1, int((1 / dt) / 2))

    cell_x = column_xyz[:, 0]
    cell_z = column_xyz[:, 2]

    stim_single = stimulus[stim_index, :, :]
    true_flow_single = true_flow[stim_index, :, :, :]
    pred_flow_single = pred_flow[stim_index, :, :, :]
    num_time, _ = stim_single.shape

    stim_min, stim_max = np.nanmin(stim_single), np.nanmax(stim_single)
    stim_norm = mcolors.Normalize(vmin=stim_min, vmax=stim_max)

    true_flow_u = true_flow_single[:, :, 0]
    true_flow_v = true_flow_single[:, :, 1]
    pred_flow_u = pred_flow_single[:, :, 0]
    pred_flow_v = pred_flow_single[:, :, 1]

    true_magnitude = np.hypot(true_flow_u, true_flow_v)
    max_mag = np.nanpercentile(true_magnitude, 99) if np.any(true_magnitude) else 1.0
    flow_norm = mcolors.Normalize(vmin=0.0, vmax=max_mag)

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.24), dpi=100)
    ax_stim = axes[0, 0]
    ax_true = axes[0, 1]
    ax_blank = axes[1, 0]
    ax_pred = axes[1, 1]

    x_margin = 0.1 * (cell_x.max() - cell_x.min())
    z_margin = 0.1 * (cell_z.max() - cell_z.min())

    for ax in [ax_stim, ax_true, ax_pred]:
        ax.set_xlim(cell_x.max() + x_margin, cell_x.min() - x_margin)
        ax.set_ylim(cell_z.min() - z_margin, cell_z.max() + z_margin)
        ax.set_aspect("equal")
        ax.set_xlabel("x (anterior ← → posterior)")
        ax.set_ylabel("z (ventral ↔ dorsal)")

    ax_blank.axis("off")
    ax_blank.set_title("(reserved)")

    sc_stim = ax_stim.scatter(
        cell_x, cell_z, c=stim_single[0, :], cmap="gray", norm=stim_norm, s=90, edgecolors="none"
    )
    ax_stim.set_title(f"Stimulus ({stim_name})")
    fig.colorbar(sc_stim, ax=ax_stim, label="intensity")

    arrow_scale = 0.2 / max_mag if max_mag > 0 else 1.0

    q_true = ax_true.quiver(
        cell_x, cell_z,
        -true_flow_u[0, :], true_flow_v[0, :],
        np.hypot(true_flow_u[0, :], true_flow_v[0, :]),
        cmap="viridis", norm=flow_norm,
        angles="xy", scale_units="xy", scale=1 / arrow_scale,
        width=0.003,
    )
    ax_true.set_title("True Optic Flow")
    fig.colorbar(q_true, ax=ax_true, label="magnitude")

    q_pred = ax_pred.quiver(
        cell_x, cell_z,
        -pred_flow_u[0, :], pred_flow_v[0, :],
        np.hypot(pred_flow_u[0, :], pred_flow_v[0, :]),
        cmap="viridis", norm=flow_norm,
        angles="xy", scale_units="xy", scale=1 / arrow_scale,
        width=0.003,
    )
    ax_pred.set_title("Predicted Optic Flow")
    fig.colorbar(q_pred, ax=ax_pred, label="magnitude")

    plt.tight_layout()

    output_folder.mkdir(parents=True, exist_ok=True)
    out_path = output_folder / f"optic_flow_{stim_name}_stim_{stim_index}.mp4"
    print(f"Writing movie for {stim_name} to {out_path}...")

    with iio_v2.get_writer(str(out_path), codec="libx264", fps=frame_rate) as writer:
        for t in range(num_time):
            sc_stim.set_array(stim_single[t, :])

            u_true_t = true_flow_u[t, :]
            v_true_t = true_flow_v[t, :]
            mag_true_t = np.hypot(u_true_t, v_true_t)
            q_true.set_UVC(-u_true_t, v_true_t, mag_true_t)

            u_pred_t = pred_flow_u[t, :]
            v_pred_t = pred_flow_v[t, :]
            mag_pred_t = np.hypot(u_pred_t, v_pred_t)
            q_pred.set_UVC(-u_pred_t, v_pred_t, mag_pred_t)

            ax_stim.set_title(f"Stimulus ({stim_name}) frame {t+1}/{num_time}")
            writer.append_data(fig_to_rgb(fig))

            if (t + 1) % 20 == 0:
                print(f"  Frame {t+1}/{num_time}")

    plt.close(fig)
    print(f"Done! Movie saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Make optic flow movies for all stimuli.")
    parser.add_argument(
        "model_path",
        type=str,
        nargs="?",
        default="models/best_model",
        help="Path to model folder or model.pt (default: models/best_model)",
    )
    parser.add_argument(
        "--stim-index",
        type=int,
        default=5,
        help="Stimulus index to visualize per stimulus type",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Optional output folder (default: <model_folder>/optic_flow_movie_stimuli)",
    )
    args = parser.parse_args()

    stimulus_specs = [
        {
            "name": "sine_right_4hz",
            "stimulus_type": "directional_sine",
            "column_info": "column_phi_theta",
            "config_overrides": {
                "stimulus": {
                    "stim_time": 2.0,
                },
            },
            "stimulus_kwargs": {
                "sine_direction": "right",
                "sine_temporal_frequency": 8.0,
                "pre_stim_time_override": 1.0,
            },
        },
        {
            "name": "sine_left_4hz",
            "stimulus_type": "directional_sine",
            "column_info": "column_phi_theta",
            "config_overrides": {
                "stimulus": {
                    "stim_time": 2.0,
                },
            },
            "stimulus_kwargs": {
                "sine_direction": "left",
                "sine_temporal_frequency": 8.0,
                "pre_stim_time_override": 1.0,
            },
        },
        {
            "name": "sine_up_4hz",
            "stimulus_type": "directional_sine",
            "column_info": "column_phi_theta",
            "config_overrides": {
                "stimulus": {
                    "stim_time": 2.0,
                },
            },
            "stimulus_kwargs": {
                "sine_direction": "up",
                "sine_temporal_frequency": 8.0,
                "pre_stim_time_override": 1.0,
            },
        },
        {
            "name": "sine_down_4hz",
            "stimulus_type": "directional_sine",
            "column_info": "column_phi_theta",
            "config_overrides": {
                "stimulus": {
                    "stim_time": 2.0,
                },
            },
            "stimulus_kwargs": {
                "sine_direction": "down",
                "sine_temporal_frequency": 8.0,
                "pre_stim_time_override": 1.0,
            },
        },
        {
            "name": "random_sine_4s_warmup",
            "stimulus_type": "random_sine",
            "column_info": "column_phi_theta",
            "config_overrides": {
                "stimulus": {
                    "seed": 100,
                },
            },
            "stimulus_kwargs": {
                "random_sine_duration_s": 4.0,
                "pre_stim_time_override": 1.0,
            },
        },
        {
            "name": "sintel_max",
            "stimulus_type": "sintel",
            "stimulus_kwargs": {
                "use_full_length": True,
                "pre_stim_time_override": 1.0,
            },
        },
        {
            "name": "flow360_max",
            "stimulus_type": "flow_360",
            "stimulus_kwargs": {
                "use_full_length": True,
            },
        },
    ]

    responses = run_model_on_stimuli(args.model_path, stimulus_specs=stimulus_specs)
    model_folder = _resolve_model_folder(Path(args.model_path))
    output_folder = Path(args.output_folder) if args.output_folder else model_folder / "optic_flow_movie_stimuli"

    for stim_name, response in responses.items():
        _make_movie_for_stim(stim_name, response, output_folder, args.stim_index)

    print("Cosine similarity (1 - cos_loss) per stimulus:")
    for stim_name, response in responses.items():
        cosine_sim = 1.0 - cos_loss(response["optic_flow"], response["pred_flow"])
        print(f"  {stim_name}: {cosine_sim.item():.6f}")


if __name__ == "__main__":
    main()
