"""
Run a trained model on any freshly generated two-eye stimulus and visualize
true vs predicted optic flow for both eyes.

Generates a stimulus using get_stimulus_two_eyes (supports all two-eye
stimulus types in stimulus_utilities.py), runs the model, and renders a movie:

  Row 0: Left eye stimulus        |  Right eye stimulus
  Row 1: Left eye true flow       |  Right eye true flow
  Row 2: Left eye predicted flow  |  Right eye predicted flow

If ground-truth flow is unavailable (all NaN, e.g. drone_view), predicted
flow from odometry (rotation + translation, constant depth Z=1) is used
for the "true flow" row.

Usage:
    python analysis/run_model_on_stimulus_two_eyes.py models/my_model/ --stim-type random_pano
    python analysis/run_model_on_stimulus_two_eyes.py models/my_model/ --stim-type drone_view --config path/to/config.toml
    python analysis/run_model_on_stimulus_two_eyes.py models/my_model/ --stim-type edge --stim-index 0 3 5
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
        description="Run model on a freshly generated two-eye stimulus and visualize flow."
    )
    parser.add_argument("model_folder", type=str, help="Path to model folder or model.pt")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.toml (default: <model_folder>/config.toml)")
    parser.add_argument("--stim-type", type=str, default="random_pano",
                        help="Stimulus type (any type supported by get_stimulus_two_eyes)")
    parser.add_argument("--stim-index", type=int, nargs="+", default=[0],
                        help="Stimulus indices to visualize (space-separated)")
    parser.add_argument("--fps", type=int, default=None,
                        help="Output movie frame rate (default: half the model rate)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum number of frames to render")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: <model_folder>/stimulus_flow_movie_two_eyes)")
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
    N = column_phi_theta.shape[0]

    # Override config device/dtype to match the model's actual device
    # (the saved config may say "cuda" but we may be running on CPU)
    config['model']['device'] = str(device)
    config['model']['dtype'] = str(dtype).replace('torch.', '')

    frame_rate = args.fps or max(1, int((1 / model_dt) / 2))

    print(f"Generating two-eye stimulus (type={args.stim_type})...")
    stimulus, optic_flow, observer_rotation, observer_translation, train_mask = (
        su.get_stimulus_two_eyes(args.stim_type, config, column_phi_theta)
    )
    stimulus_np = stimulus.cpu().numpy()
    optic_flow_np = optic_flow.cpu().numpy()
    observer_rotation_np = observer_rotation.cpu().numpy()
    observer_translation_np = observer_translation.cpu().numpy()
    train_mask_np = train_mask.cpu().numpy()

    n_examples = stimulus_np.shape[0]
    print(f"  Generated {n_examples} stimulus examples, shape {stimulus_np.shape}")

    column_xyz = model.get_column_xyz()
    cell_x, cell_y, cell_z = column_xyz[:, 0], column_xyz[:, 1], column_xyz[:, 2]

    flow_key = get_flow_key(model)
    if flow_key is None:
        print("WARNING: No flow readout found in model. Using first readout.")
        flow_key = next(iter(model._readout_modules.keys()))

    output_dir = (Path(args.output_dir) if args.output_dir
                  else model_folder / "stimulus_flow_movie_two_eyes")
    output_dir.mkdir(parents=True, exist_ok=True)

    pre_stim_steps = int(config["stimulus"].get("pre_stim_time", 0) / model_dt)

    phi = column_phi_theta[:, 0]
    theta = column_phi_theta[:, 1]
    phi_right_t = torch.tensor(phi, dtype=torch.float32)
    theta_right_t = torch.tensor(theta, dtype=torch.float32)
    phi_left_t = torch.tensor(-phi, dtype=torch.float32)

    for stim_index in args.stim_index:
        if stim_index >= n_examples:
            print(f"Skipping stim {stim_index}: only {n_examples} examples available")
            continue

        # Split into right / left eye
        stim_right = stimulus_np[stim_index, :, :N]
        stim_left = stimulus_np[stim_index, :, N:]
        flow_right = optic_flow_np[stim_index, :, :N, :]
        flow_left = optic_flow_np[stim_index, :, N:, :]
        mask_right = train_mask_np[stim_index, :, :N]
        mask_left = train_mask_np[stim_index, :, N:]

        # If ground-truth flow is unavailable, synthesize from odometry
        if np.all(np.isnan(flow_right[pre_stim_steps:])):
            print(f"  Stim {stim_index}: ground-truth flow unavailable — "
                  "synthesizing from odometry (Z=1)")
            from analysis.estimate_self_motion import generate_optic_flow
            omega_t = torch.tensor(
                observer_rotation_np[stim_index] * np.pi / 180.0,
                dtype=torch.float32,
            )
            trans_t = torch.tensor(
                observer_translation_np[stim_index], dtype=torch.float32,
            )
            pred_ur_r, pred_uu_r = generate_optic_flow(
                phi_right_t, theta_right_t, omega_t, trans_t, 1.0)
            pred_ur_l, pred_uu_l = generate_optic_flow(
                phi_left_t, theta_right_t, omega_t, trans_t, 1.0)
            flow_right = np.stack(
                [pred_ur_r.numpy(), pred_uu_r.numpy()], axis=-1) * (180.0 / np.pi)
            flow_left = np.stack(
                [pred_ur_l.numpy(), pred_uu_l.numpy()], axis=-1) * (180.0 / np.pi)
            flow_label = "Flow (from odometry, Z=1)"
        else:
            flow_label = "True Flow"

        num_frames = stim_right.shape[0]
        if args.max_frames is not None:
            num_frames = min(num_frames, args.max_frames)
            stim_right = stim_right[:num_frames]
            stim_left = stim_left[:num_frames]
            flow_right = flow_right[:num_frames]
            flow_left = flow_left[:num_frames]
            mask_right = mask_right[:num_frames]
            mask_left = mask_left[:num_frames]

        right_input = torch.tensor(
            stim_right, device=device, dtype=dtype).unsqueeze(0)
        left_input = torch.tensor(
            stim_left, device=device, dtype=dtype).unsqueeze(0)

        print(f"\nRunning model on stimulus {stim_index} ({num_frames} frames, N={N})...")
        with torch.no_grad():
            right_result, left_result = model.predict_two_eyes(
                right_input, left_input)

        pred_flow_right = right_result["outputs"][flow_key][0].cpu().numpy()[:num_frames]
        pred_flow_left = left_result["outputs"][flow_key][0].cpu().numpy()[:num_frames]

        out_path = output_dir / f"{args.stim_type}_stim_{stim_index}.mp4"
        print(f"  Creating movie -> {out_path}")
        _make_two_eye_flow_movie(
            stimulus_left=stim_left,
            stimulus_right=stim_right,
            true_flow_left=flow_left,
            true_flow_right=flow_right,
            pred_flow_left=pred_flow_left,
            pred_flow_right=pred_flow_right,
            cell_x=cell_x,
            cell_y=cell_y,
            cell_z=cell_z,
            output_path=out_path,
            frame_rate=frame_rate,
            mask_left=mask_left,
            mask_right=mask_right,
            stim_type=args.stim_type,
            flow_label=flow_label,
        )


def _make_two_eye_flow_movie(
    stimulus_left: np.ndarray,
    stimulus_right: np.ndarray,
    true_flow_left: np.ndarray,
    true_flow_right: np.ndarray,
    pred_flow_left: np.ndarray,
    pred_flow_right: np.ndarray,
    cell_x: np.ndarray,
    cell_y: np.ndarray,
    cell_z: np.ndarray,
    output_path: Path,
    frame_rate: int,
    mask_left: np.ndarray | None = None,
    mask_right: np.ndarray | None = None,
    stim_type: str = "",
    flow_label: str = "True Flow",
) -> None:
    """Render a 3x2 movie: stimulus / true flow / predicted flow for both eyes."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import matplotlib.colors as mcolors
    import imageio as iio

    num_frames = stimulus_left.shape[0]

    _MASK_COLOR = np.array([1.0, 1.0, 0.0, 1.0])
    gray_cmap = plt.cm.gray

    stim_min = min(np.nanmin(stimulus_left), np.nanmin(stimulus_right))
    stim_max = max(np.nanmax(stimulus_left), np.nanmax(stimulus_right))
    stim_norm = mcolors.Normalize(vmin=stim_min, vmax=stim_max)

    def _masked_rgba(values, m):
        rgba = gray_cmap(stim_norm(values))
        if m is not None:
            rgba[~m] = _MASK_COLOR
        return rgba

    # Project all flows to Cartesian (x, z) for quiver plotting
    # Left eye has phi_left = -phi, so cell_y_left = -cell_y
    cell_y_left = -cell_y

    true_vx_r, true_vz_r = angular_flow_to_cartesian_xz(
        true_flow_right[:, :, 0], true_flow_right[:, :, 1], cell_x, cell_y, cell_z)
    true_vx_l, true_vz_l = angular_flow_to_cartesian_xz(
        true_flow_left[:, :, 0], true_flow_left[:, :, 1], cell_x, cell_y_left, cell_z)
    pred_vx_r, pred_vz_r = angular_flow_to_cartesian_xz(
        pred_flow_right[:, :, 0], pred_flow_right[:, :, 1], cell_x, cell_y, cell_z)
    pred_vx_l, pred_vz_l = angular_flow_to_cartesian_xz(
        pred_flow_left[:, :, 0], pred_flow_left[:, :, 1], cell_x, cell_y_left, cell_z)

    true_mag_r = np.hypot(true_vx_r, true_vz_r)
    true_mag_l = np.hypot(true_vx_l, true_vz_l)
    pred_mag_r = np.hypot(pred_vx_r, pred_vz_r)
    pred_mag_l = np.hypot(pred_vx_l, pred_vz_l)

    all_mag = np.concatenate([
        true_mag_r.ravel(), true_mag_l.ravel(),
        pred_mag_r.ravel(), pred_mag_l.ravel(),
    ])
    finite_mag = all_mag[np.isfinite(all_mag)]
    max_mag = (np.percentile(finite_mag, 99)
               if finite_mag.size > 0 and finite_mag.max() > 0 else 1.0)
    flow_norm = mcolors.Normalize(vmin=0.0, vmax=max_mag)
    arrow_scale = 0.2 / max_mag if max_mag > 0 else 1.0

    for arr in (true_vx_r, true_vz_r, true_vx_l, true_vz_l,
                pred_vx_r, pred_vz_r, pred_vx_l, pred_vz_l,
                true_mag_r, true_mag_l, pred_mag_r, pred_mag_l):
        np.nan_to_num(arr, nan=0.0, copy=False)

    # 3 rows x 2 columns: [stimulus, true flow, predicted flow] x [left, right]
    fig, axes = plt.subplots(3, 2, figsize=(12.8, 16), dpi=100)
    ax_stim_l, ax_stim_r = axes[0]
    ax_true_l, ax_true_r = axes[1]
    ax_pred_l, ax_pred_r = axes[2]

    x_margin = 0.1 * (cell_x.max() - cell_x.min())
    z_margin = 0.1 * (cell_z.max() - cell_z.min())

    def _setup_eye_axes(ax_left, ax_right):
        for ax, flip in [(ax_left, False), (ax_right, True)]:
            if flip:
                ax.set_xlim(cell_x.max() + x_margin, cell_x.min() - x_margin)
            else:
                ax.set_xlim(cell_x.min() - x_margin, cell_x.max() + x_margin)
            ax.set_ylim(cell_z.min() - z_margin, cell_z.max() + z_margin)
            ax.set_aspect("equal")

    _setup_eye_axes(ax_stim_l, ax_stim_r)
    _setup_eye_axes(ax_true_l, ax_true_r)
    _setup_eye_axes(ax_pred_l, ax_pred_r)

    mask_l_0 = mask_left[0] if mask_left is not None else None
    mask_r_0 = mask_right[0] if mask_right is not None else None

    # Row 0: stimulus
    sc_stim_l = ax_stim_l.scatter(
        cell_x, cell_z, c=_masked_rgba(stimulus_left[0], mask_l_0),
        s=90, edgecolors="none")
    sc_stim_r = ax_stim_r.scatter(
        cell_x, cell_z, c=_masked_rgba(stimulus_right[0], mask_r_0),
        s=90, edgecolors="none")
    ax_stim_l.set_title(f"Left Eye ({stim_type})")
    ax_stim_r.set_title(f"Right Eye ({stim_type})")

    quiver_kw = dict(
        cmap="viridis", norm=flow_norm, angles="xy",
        scale_units="xy", scale=1 / arrow_scale, width=0.003,
    )

    # Row 1: true flow
    q_true_l = ax_true_l.quiver(
        cell_x, cell_z, true_vx_l[0], true_vz_l[0], true_mag_l[0], **quiver_kw)
    q_true_r = ax_true_r.quiver(
        cell_x, cell_z, true_vx_r[0], true_vz_r[0], true_mag_r[0], **quiver_kw)
    ax_true_l.set_title(f"Left Eye — {flow_label}")
    ax_true_r.set_title(f"Right Eye — {flow_label}")

    # Row 2: predicted flow
    q_pred_l = ax_pred_l.quiver(
        cell_x, cell_z, pred_vx_l[0], pred_vz_l[0], pred_mag_l[0], **quiver_kw)
    q_pred_r = ax_pred_r.quiver(
        cell_x, cell_z, pred_vx_r[0], pred_vz_r[0], pred_mag_r[0], **quiver_kw)
    ax_pred_l.set_title("Left Eye — Predicted Flow")
    ax_pred_r.set_title("Right Eye — Predicted Flow")

    # Single shared colorbar
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=flow_norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes[1:].ravel().tolist(), label="flow magnitude",
                 shrink=0.6, pad=0.02)

    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"  Writing {num_frames} frames at {frame_rate} fps...")

    with iio.get_writer(str(output_path), codec="libx264", fps=frame_rate) as writer:
        for t in range(num_frames):
            mask_l_t = mask_left[t] if mask_left is not None else None
            mask_r_t = mask_right[t] if mask_right is not None else None

            sc_stim_l.set_facecolors(_masked_rgba(stimulus_left[t], mask_l_t))
            sc_stim_r.set_facecolors(_masked_rgba(stimulus_right[t], mask_r_t))

            q_true_l.set_UVC(true_vx_l[t], true_vz_l[t], true_mag_l[t])
            q_true_r.set_UVC(true_vx_r[t], true_vz_r[t], true_mag_r[t])
            q_pred_l.set_UVC(pred_vx_l[t], pred_vz_l[t], pred_mag_l[t])
            q_pred_r.set_UVC(pred_vx_r[t], pred_vz_r[t], pred_mag_r[t])

            ax_stim_l.set_title(f"Left Eye ({stim_type}) — frame {t + 1}/{num_frames}")
            ax_stim_r.set_title(f"Right Eye ({stim_type}) — frame {t + 1}/{num_frames}")

            writer.append_data(fig_to_rgb(fig))
            if (t + 1) % 50 == 0:
                print(f"    Frame {t + 1}/{num_frames}")

    plt.close(fig)
    print(f"  Done! -> {output_path}")


if __name__ == "__main__":
    main()
