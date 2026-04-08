"""
Run a trained model on any freshly generated one-eye stimulus and visualize
true vs predicted optic flow.

Generates a stimulus using get_stimulus_one_eye (supports all stimulus types
in stimulus_utilities.py), runs the model, and renders a movie:

  Row 0: Stimulus intensity (scatter plot)
  Row 1: True optic flow (quiver plot)
  Row 2: Predicted optic flow (quiver plot)

Usage:
    python analysis/run_model_on_stimulus.py models/my_model/ --stim-type random_sine
    python analysis/run_model_on_stimulus.py models/my_model/ --stim-type static_sine --config path/to/config.toml
    python analysis/run_model_on_stimulus.py models/my_model/ --stim-type edge --stim-index 0 3 5
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

import data_utilities as du
import neural_models as nm
import stimulus_utilities as su

from analysis_utilities import (
    fig_to_rgb,
    angular_flow_to_cartesian_xz,
    get_flow_key,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run model on a freshly generated one-eye stimulus and visualize flow."
    )
    parser.add_argument("model_folder", type=str, help="Path to model folder or model.pt")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.toml (default: <model_folder>/config.toml)")
    parser.add_argument("--stim-type", type=str, default="random_sine",
                        help="Stimulus type (any type supported by get_stimulus_one_eye)")
    parser.add_argument("--stim-index", type=int, nargs="+", default=[0],
                        help="Stimulus indices to visualize (space-separated)")
    parser.add_argument("--fps", type=int, default=None,
                        help="Output movie frame rate (default: half the model rate)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum number of frames to render")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: <model_folder>/stimulus_flow_movie)")
    args = parser.parse_args()

    model_path = Path(args.model_folder)
    model_folder = model_path.parent if model_path.is_file() else model_path

    if args.config:
        print(f"Loading config from {args.config}...")
        config = du.load_config(args.config)
    else:
        print("Loading config from config.toml...")
        config = du.load_config()

    print(f"Loading model from {model_path}...")
    model = nm.NeuralModel.load(model_path)
    model.eval()

    device = model.device
    dtype = model.dtype
    model_dt = model.dt
    column_phi_theta = model.column_phi_theta

    # Override config device/dtype to match the model's actual device
    # (the saved config may say "cuda" but we may be running on CPU)
    config['model']['device'] = str(device)
    config['model']['dtype'] = str(dtype).replace('torch.', '')

    frame_rate = args.fps or max(1, int((1 / model_dt) / 2))

    print(f"Generating stimulus (type={args.stim_type})...")
    stimulus, optic_flow, train_mask = su.get_stimulus_one_eye(
        args.stim_type, config, column_phi_theta,
    )
    stimulus_np = stimulus.cpu().numpy()
    optic_flow_np = optic_flow.cpu().numpy()
    train_mask_np = train_mask.cpu().numpy()

    n_examples = stimulus_np.shape[0]
    print(f"  Generated {n_examples} stimulus examples, shape {stimulus_np.shape}")

    column_xyz = model.get_column_xyz()
    cell_x, cell_y, cell_z = column_xyz[:, 0], column_xyz[:, 1], column_xyz[:, 2]

    flow_key = get_flow_key(model)
    if flow_key is None:
        print("WARNING: No flow readout found in model. Using first readout.")
        flow_key = next(iter(model._readout_modules.keys()))

    output_dir = Path(args.output_dir) if args.output_dir else model_folder / "stimulus_flow_movie"
    output_dir.mkdir(parents=True, exist_ok=True)

    for stim_index in args.stim_index:
        if stim_index >= n_examples:
            print(f"Skipping stim {stim_index}: only {n_examples} examples available")
            continue

        stim_input = stimulus[stim_index:stim_index + 1].to(device=device, dtype=dtype)
        true_flow = optic_flow_np[stim_index]
        mask = train_mask_np[stim_index]

        num_frames = stim_input.shape[1]
        if args.max_frames is not None:
            num_frames = min(num_frames, args.max_frames)
            stim_input = stim_input[:, :num_frames]
            true_flow = true_flow[:num_frames]
            mask = mask[:num_frames]

        print(f"\nRunning model on stimulus {stim_index} ({num_frames} frames)...")
        with torch.no_grad():
            result = model.predict(stim_input)
            pred_flow = result["outputs"][flow_key][0].cpu().numpy()

        pred_flow = pred_flow[:num_frames]
        stim_np = stimulus_np[stim_index, :num_frames]

        out_path = output_dir / f"{args.stim_type}_stim_{stim_index}.mp4"
        print(f"  Creating movie -> {out_path}")
        _make_one_eye_flow_movie(
            stimulus=stim_np,
            true_flow=true_flow,
            pred_flow=pred_flow,
            cell_x=cell_x,
            cell_y=cell_y,
            cell_z=cell_z,
            output_path=out_path,
            frame_rate=frame_rate,
            mask=mask,
            stim_type=args.stim_type,
        )


def _make_one_eye_flow_movie(
    stimulus: np.ndarray,
    true_flow: np.ndarray,
    pred_flow: np.ndarray,
    cell_x: np.ndarray,
    cell_y: np.ndarray,
    cell_z: np.ndarray,
    output_path: Path,
    frame_rate: int,
    mask: np.ndarray | None = None,
    stim_type: str = "",
) -> None:
    """Render a 3-row movie: stimulus, true flow, predicted flow."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import matplotlib.colors as mcolors
    import imageio as iio

    num_frames = stimulus.shape[0]

    _MASK_COLOR = np.array([1.0, 1.0, 0.0, 1.0])
    gray_cmap = plt.cm.gray

    stim_min, stim_max = np.nanmin(stimulus), np.nanmax(stimulus)
    stim_norm = mcolors.Normalize(vmin=stim_min, vmax=stim_max)

    def _masked_rgba(values, m):
        rgba = gray_cmap(stim_norm(values))
        if m is not None:
            rgba[~m] = _MASK_COLOR
        return rgba

    # Project true and predicted flow to Cartesian (x, z) for quiver
    true_vx, true_vz = angular_flow_to_cartesian_xz(
        true_flow[:, :, 0], true_flow[:, :, 1], cell_x, cell_y, cell_z)
    pred_vx, pred_vz = angular_flow_to_cartesian_xz(
        pred_flow[:, :, 0], pred_flow[:, :, 1], cell_x, cell_y, cell_z)

    true_mag = np.hypot(true_vx, true_vz)
    pred_mag = np.hypot(pred_vx, pred_vz)

    all_mag = np.concatenate([true_mag.ravel(), pred_mag.ravel()])
    finite_mag = all_mag[np.isfinite(all_mag)]
    max_mag = (np.percentile(finite_mag, 99)
               if finite_mag.size > 0 and finite_mag.max() > 0 else 1.0)
    flow_norm = mcolors.Normalize(vmin=0.0, vmax=max_mag)
    arrow_scale = 0.2 / max_mag if max_mag > 0 else 1.0

    true_vx = np.nan_to_num(true_vx, nan=0.0)
    true_vz = np.nan_to_num(true_vz, nan=0.0)
    pred_vx = np.nan_to_num(pred_vx, nan=0.0)
    pred_vz = np.nan_to_num(pred_vz, nan=0.0)
    true_mag = np.nan_to_num(true_mag, nan=0.0)
    pred_mag = np.nan_to_num(pred_mag, nan=0.0)

    fig, axes = plt.subplots(3, 1, figsize=(8, 16), dpi=100)
    ax_stim, ax_true, ax_pred = axes

    x_margin = 0.1 * (cell_x.max() - cell_x.min())
    z_margin = 0.1 * (cell_z.max() - cell_z.min())

    for ax in axes:
        ax.set_xlim(cell_x.min() - x_margin, cell_x.max() + x_margin)
        ax.set_ylim(cell_z.min() - z_margin, cell_z.max() + z_margin)
        ax.set_aspect("equal")

    mask_0 = mask[0] if mask is not None else None
    sc_stim = ax_stim.scatter(
        cell_x, cell_z, c=_masked_rgba(stimulus[0], mask_0), s=90, edgecolors="none")
    ax_stim.set_title(f"Stimulus ({stim_type})")

    quiver_kw = dict(
        cmap="viridis", norm=flow_norm, angles="xy",
        scale_units="xy", scale=1 / arrow_scale, width=0.003,
    )
    q_true = ax_true.quiver(
        cell_x, cell_z, true_vx[0], true_vz[0], true_mag[0], **quiver_kw)
    ax_true.set_title("True Flow")
    fig.colorbar(q_true, ax=ax_true, label="magnitude")

    q_pred = ax_pred.quiver(
        cell_x, cell_z, pred_vx[0], pred_vz[0], pred_mag[0], **quiver_kw)
    ax_pred.set_title("Predicted Flow")
    fig.colorbar(q_pred, ax=ax_pred, label="magnitude")

    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"  Writing {num_frames} frames at {frame_rate} fps...")

    with iio.get_writer(str(output_path), codec="libx264", fps=frame_rate) as writer:
        for t in range(num_frames):
            mask_t = mask[t] if mask is not None else None
            sc_stim.set_facecolors(_masked_rgba(stimulus[t], mask_t))

            q_true.set_UVC(true_vx[t], true_vz[t], true_mag[t])
            q_pred.set_UVC(pred_vx[t], pred_vz[t], pred_mag[t])

            ax_stim.set_title(f"Stimulus ({stim_type}) — frame {t + 1}/{num_frames}")

            writer.append_data(fig_to_rgb(fig))
            if (t + 1) % 50 == 0:
                print(f"    Frame {t + 1}/{num_frames}")

    plt.close(fig)
    print(f"  Done! -> {output_path}")


if __name__ == "__main__":
    main()
