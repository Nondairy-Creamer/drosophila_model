"""
Visualize two-eye stimuli as a movie (inputs + flow + odometry).

Layout (3x2):
  Row 0:  left eye ommatidia    |  right eye ommatidia
  Row 1:  left eye optic flow   |  right eye optic flow
  Row 2:  rotation rate         |  translation rate

If ground truth optic flow is unavailable (all NaN), predicted flow from
odometry (rotation + translation, constant depth Z=1) is shown instead.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch

import data_utilities as du
import stimulus_utilities as su
from analysis.estimate_self_motion import generate_optic_flow


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make a movie showing two-eye stimulus inputs and odometry."
    )
    parser.add_argument(
        "--config", type=str, default="config.toml", help="Path to config file"
    )
    parser.add_argument(
        "--stim-index", type=int, nargs="+", default=[3],
        help="Stimulus indices to visualize (space-separated list)"
    )
    parser.add_argument(
        "--fps", type=int, default=100, help="Output movie frame rate"
    )
    parser.add_argument(
        "--stim-time", type=float, default=None,
        help="Override stim_time from config (seconds)"
    )
    parser.add_argument(
        "--stim-type", type=str, default="drone_view",
        help="Stimulus type (default: drone_view)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="figures/two_eyes",
        help="Output directory for the movie"
    )
    args = parser.parse_args()

    config = du.load_config(args.config)
    connectome_folder = Path(config["data"]["connectome_folder"])
    output_folder = Path(args.output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)

    stimulus_type = args.stim_type if args.stim_type else config["stimulus"]["stimulus_type"]
    frame_rate = args.fps

    if args.stim_time is not None:
        config["stimulus"]["stim_time"] = args.stim_time

    column_info_df = pd.read_parquet(connectome_folder / "optic_lobes_column_info.parquet")
    column_phi_theta = column_info_df[["phi", "theta"]].to_numpy()

    print("Loading two-eye stimulus and odometry...")
    stimulus, optic_flow, observer_rotation, observer_translation, train_mask = (
        su.get_stimulus_two_eyes(stimulus_type, config, column_phi_theta)
    )
    stimulus = stimulus.numpy()
    observer_rotation = observer_rotation.numpy()
    observer_translation = observer_translation.numpy()
    train_mask = train_mask.numpy()

    optic_flow = optic_flow.numpy()

    N = column_phi_theta.shape[0]
    phi = column_phi_theta[:, 0]
    theta = column_phi_theta[:, 1]
    cell_x = np.sin(theta) * np.cos(phi)
    cell_y = np.sin(theta) * np.sin(phi)
    cell_z = np.cos(theta)

    model_dt = config["model"]["dt"]
    pre_stim_steps = int(config["stimulus"].get("pre_stim_time", 0) / model_dt)

    print(f"Stimulus shape: {stimulus.shape}")
    print(f"Rotation shape: {observer_rotation.shape}")

    # Predict optic flow from odometry (rotation + translation, depth=1)
    phi_right_t = torch.tensor(phi, dtype=torch.float32)
    theta_right_t = torch.tensor(theta, dtype=torch.float32)
    phi_left_t = torch.tensor(-phi, dtype=torch.float32)

    for stim_index in args.stim_index:
        if stim_index >= stimulus.shape[0]:
            print(f"Skipping stim {stim_index}: only {stimulus.shape[0]} examples available")
            continue

        out_path = output_folder / f"two_eyes_stim_{stim_index}.mp4"
        print(f"\nCreating movie for stimulus {stim_index} -> {out_path}")

        stim_right = stimulus[stim_index, :, :N]
        stim_left = stimulus[stim_index, :, N:]
        flow_right = optic_flow[stim_index, :, :N, :]
        flow_left = optic_flow[stim_index, :, N:, :]
        mask_right = train_mask[stim_index, :, :N]
        mask_left = train_mask[stim_index, :, N:]

        # If ground truth flow is unavailable (all NaN after the pre-stim interval),
        # replace with predicted flow from odometry (Z=1)
        if np.all(np.isnan(flow_right[pre_stim_steps:])):
            print("  Ground truth flow unavailable — using predicted flow from odometry (Z=1)")
            omega_t = torch.tensor(
                observer_rotation[stim_index] * np.pi / 180.0, dtype=torch.float32
            )
            trans_t = torch.tensor(
                observer_translation[stim_index], dtype=torch.float32
            )

            pred_ur_r, pred_uu_r = generate_optic_flow(
                phi_right_t, theta_right_t, omega_t, trans_t, 1.0
            )
            pred_ur_l, pred_uu_l = generate_optic_flow(
                phi_left_t, theta_right_t, omega_t, trans_t, 1.0
            )
            pred_ur_l = -pred_ur_l  # flip u_right for left eye

            flow_right = np.stack([pred_ur_r.numpy(), pred_uu_r.numpy()], axis=-1)
            flow_left = np.stack([pred_ur_l.numpy(), pred_uu_l.numpy()], axis=-1)
            flow_right *= 180.0 / np.pi  # rad/s -> deg/s
            flow_left *= 180.0 / np.pi
            flow_label = "Predicted Flow (Z=1)"
        else:
            flow_label = "Flow"

        _make_two_eyes_input_movie(
            sampled_right=stim_right,
            sampled_left=stim_left,
            flow_right=flow_right,
            flow_left=flow_left,
            rotation_rate=observer_rotation[stim_index],
            translation_rate=observer_translation[stim_index],
            flow_label=flow_label,
            cell_x=cell_x,
            cell_y=cell_y,
            cell_z=cell_z,
            output_path=out_path,
            frame_rate=frame_rate,
            model_dt=model_dt,
            mask_right=mask_right,
            mask_left=mask_left,
        )


def _make_two_eyes_input_movie(
    sampled_right: np.ndarray,
    sampled_left: np.ndarray,
    flow_right: np.ndarray,
    flow_left: np.ndarray,
    rotation_rate: np.ndarray,
    translation_rate: np.ndarray,
    cell_x: np.ndarray,
    cell_y: np.ndarray,
    cell_z: np.ndarray,
    output_path: Path,
    frame_rate: int,
    model_dt: float,
    flow_label: str = "Flow",
    trail_frames: int = 60,
    mask_right: np.ndarray | None = None,
    mask_left: np.ndarray | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import matplotlib.colors as mcolors
    import imageio as iio

    from analysis_utilities import fig_to_rgb, angular_flow_to_cartesian_xz

    num_frames = sampled_right.shape[0]

    vmin = min(np.nanmin(sampled_left), np.nanmin(sampled_right))
    vmax = max(np.nanmax(sampled_left), np.nanmax(sampled_right))
    scatter_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    gray_cmap = plt.cm.gray
    _MASK_COLOR = np.array([1.0, 1.0, 0.0, 1.0])

    def _masked_rgba(values, mask):
        rgba = gray_cmap(scatter_norm(values))
        if mask is not None:
            rgba[~mask] = _MASK_COLOR
        return rgba

    # --- Optic flow projection ---
    flow_r_vx, flow_r_vz = angular_flow_to_cartesian_xz(
        flow_right[:, :, 0], flow_right[:, :, 1], cell_x, cell_y, cell_z)
    flow_l_vx, flow_l_vz = angular_flow_to_cartesian_xz(
        flow_left[:, :, 0], flow_left[:, :, 1], cell_x, cell_y, cell_z)
    mag_r = np.hypot(flow_r_vx, flow_r_vz)
    mag_l = np.hypot(flow_l_vx, flow_l_vz)
    finite_mag = np.concatenate([mag_r[np.isfinite(mag_r)], mag_l[np.isfinite(mag_l)]])
    flow_max_mag = np.percentile(finite_mag, 99) if finite_mag.size > 0 and finite_mag.max() > 0 else 1.0
    flow_norm = mcolors.Normalize(vmin=0.0, vmax=flow_max_mag)
    arrow_scale = 0.2 / flow_max_mag if flow_max_mag > 0 else 1.0
    flow_r_vx = np.nan_to_num(flow_r_vx, nan=0.0)
    flow_r_vz = np.nan_to_num(flow_r_vz, nan=0.0)
    flow_l_vx = np.nan_to_num(flow_l_vx, nan=0.0)
    flow_l_vz = np.nan_to_num(flow_l_vz, nan=0.0)
    mag_r = np.nan_to_num(mag_r, nan=0.0)
    mag_l = np.nan_to_num(mag_l, nan=0.0)

    # --- Figure layout: 3 rows x 2 cols ---
    fig, axes = plt.subplots(3, 2, figsize=(12.8, 14), dpi=100,
                             gridspec_kw={"height_ratios": [3, 3, 2]})
    ax_sam_l, ax_sam_r = axes[0, 0], axes[0, 1]
    ax_flow_l, ax_flow_r = axes[1, 0], axes[1, 1]
    ax_rot_rate, ax_trans_rate = axes[2, 0], axes[2, 1]

    x_margin = 0.1 * (cell_x.max() - cell_x.min())
    z_margin = 0.1 * (cell_z.max() - cell_z.min())

    for ax, flip in [(ax_sam_r, True), (ax_sam_l, False),
                     (ax_flow_r, True), (ax_flow_l, False)]:
        if flip:
            ax.set_xlim(cell_x.max() + x_margin, cell_x.min() - x_margin)
        else:
            ax.set_xlim(cell_x.min() - x_margin, cell_x.max() + x_margin)
        ax.set_ylim(cell_z.min() - z_margin, cell_z.max() + z_margin)
        ax.set_aspect("equal")

    mask_r_0 = mask_right[0] if mask_right is not None else None
    mask_l_0 = mask_left[0] if mask_left is not None else None

    sc_r = ax_sam_r.scatter(cell_x, cell_z,
                            c=_masked_rgba(sampled_right[0], mask_r_0),
                            s=90, edgecolors="none")
    sc_l = ax_sam_l.scatter(cell_x, cell_z,
                            c=_masked_rgba(sampled_left[0], mask_l_0),
                            s=90, edgecolors="none")
    ax_sam_r.set_title("Right Eye")
    ax_sam_l.set_title("Left Eye")

    # --- Flow quiver plots ---
    q_r = ax_flow_r.quiver(
        cell_x, cell_z, flow_r_vx[0], flow_r_vz[0], mag_r[0],
        cmap="viridis", norm=flow_norm, angles="xy",
        scale_units="xy", scale=1 / arrow_scale, width=0.003,
    )
    ax_flow_r.set_title(f"Right Eye {flow_label}")
    fig.colorbar(q_r, ax=ax_flow_r, label="magnitude")

    q_l = ax_flow_l.quiver(
        cell_x, cell_z, flow_l_vx[0], flow_l_vz[0], mag_l[0],
        cmap="viridis", norm=flow_norm, angles="xy",
        scale_units="xy", scale=1 / arrow_scale, width=0.003,
    )
    ax_flow_l.set_title(f"Left Eye {flow_label}")
    fig.colorbar(q_l, ax=ax_flow_l, label="magnitude")

    # --- Odometry line plots ---
    rot_labels = ["roll", "pitch", "yaw"]
    trans_labels = ["x", "y", "z"]
    rot_colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    trans_colors = ["#e41a1c", "#377eb8", "#4daf4a"]

    rot_rate_lines = []
    for label, color in zip(rot_labels, rot_colors):
        line, = ax_rot_rate.plot([], [], color=color, label=label, linewidth=1.5)
        rot_rate_lines.append(line)
    ax_rot_rate.legend(loc="upper left", fontsize=8)
    ax_rot_rate.set_ylabel("deg/s")
    ax_rot_rate.set_xlabel("time (s)")
    ax_rot_rate.set_title("Rotation Rate")

    trans_rate_lines = []
    for label, color in zip(trans_labels, trans_colors):
        line, = ax_trans_rate.plot([], [], color=color, label=label, linewidth=1.5)
        trans_rate_lines.append(line)
    ax_trans_rate.legend(loc="upper left", fontsize=8)
    ax_trans_rate.set_ylabel("velocity")
    ax_trans_rate.set_xlabel("time (s)")
    ax_trans_rate.set_title("Translation Rate")

    def _fixed_ylim(data, margin_frac=0.1, min_margin=0.01):
        finite = data[np.isfinite(data)] if np.issubdtype(data.dtype, np.floating) else data.ravel()
        if finite.size == 0:
            return -1, 1
        lo, hi = finite.min(), finite.max()
        margin = max(margin_frac * (hi - lo), min_margin)
        return lo - margin, hi + margin

    ax_rot_rate.set_ylim(*_fixed_ylim(rotation_rate[:num_frames], min_margin=1.0))
    ax_trans_rate.set_ylim(*_fixed_ylim(translation_rate[:num_frames], min_margin=0.01))

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {num_frames} frames to {output_path}...")

    times = np.arange(num_frames) * model_dt

    with iio.get_writer(str(output_path), codec="libx264", fps=frame_rate) as writer:
        for t in range(num_frames):
            mask_r_t = mask_right[t] if mask_right is not None else None
            mask_l_t = mask_left[t] if mask_left is not None else None

            sc_r.set_facecolors(_masked_rgba(sampled_right[t], mask_r_t))
            sc_l.set_facecolors(_masked_rgba(sampled_left[t], mask_l_t))

            time_s = t * model_dt
            ax_sam_r.set_title(f"Right Eye (t={time_s:.2f}s)")
            ax_sam_l.set_title(f"Left Eye (t={time_s:.2f}s)")

            q_r.set_UVC(flow_r_vx[t], flow_r_vz[t], np.hypot(flow_r_vx[t], flow_r_vz[t]))
            q_l.set_UVC(flow_l_vx[t], flow_l_vz[t], np.hypot(flow_l_vx[t], flow_l_vz[t]))

            t_start = max(0, t - trail_frames + 1)
            t_slice = slice(t_start, t + 1)
            t_times = times[t_slice]

            for i, line in enumerate(rot_rate_lines):
                line.set_data(t_times, rotation_rate[t_slice, i])
            ax_rot_rate.set_xlim(t_times[0], t_times[0] + trail_frames * model_dt)

            for i, line in enumerate(trans_rate_lines):
                line.set_data(t_times, translation_rate[t_slice, i])
            ax_trans_rate.set_xlim(t_times[0], t_times[0] + trail_frames * model_dt)

            writer.append_data(fig_to_rgb(fig))
            if (t + 1) % 50 == 0:
                print(f"  Frame {t+1}/{num_frames}")

    plt.close(fig)
    print(f"Done! Movie saved to {output_path}")


if __name__ == "__main__":
    main()
