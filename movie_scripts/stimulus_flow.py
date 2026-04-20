"""
Create a movie of stimulus intensity with true optic flow. No model required.

Supports both one-eye and two-eye stimuli via the --two-eyes flag.

One-eye layout (1x2):
  Left: Stimulus intensity (scatter)
  Right: True optic flow (quiver)

Two-eye layout (3x2):
  Row 0: Left eye ommatidia    | Right eye ommatidia
  Row 1: Left eye optic flow   | Right eye optic flow
  Row 2: Rotation rate         | Translation rate

If ground-truth flow is unavailable (all NaN), predicted flow from odometry
(rotation + translation, constant depth Z=1) is shown instead.

Usage:
    python movie_scripts/stimulus_flow.py --stim-type flyview
    python movie_scripts/stimulus_flow.py --two-eyes --stim-type drone_view
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch

import data_utilities as du
import stimulus_utilities as su

from movie_utilities import (
    masked_rgba,
    setup_eye_axes,
    prepare_flow_quiver,
    write_movie,
    ylim_from_data,
    fig_to_rgb,
    make_stimulus_flow_movie,
    flow_direction_colors,
    update_direction_quiver,
    add_direction_colorwheel,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make a movie showing stimulus and true optic flow (no model needed)."
    )
    parser.add_argument("--config", type=str, default="config.toml", help="Path to config file")
    parser.add_argument("--two-eyes", action="store_true",
                        help="Use two-eye stimulus")
    parser.add_argument("--stim-index", type=int, nargs="+", default=None,
                        help="Stimulus indices to visualize (default: [5] for 1-eye, [3] for 2-eye)")
    parser.add_argument("--fps", type=int, default=None,
                        help="Output movie frame rate (default: 1/dt from config)")
    parser.add_argument("--stim-type", type=str, default=None,
                        help="Stimulus type (default: flyview for 1-eye, drone_view for 2-eye)")
    parser.add_argument("--stim-time", type=float, default=None,
                        help="Override stim_time from config (seconds)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for the movie")
    args = parser.parse_args()

    if args.stim_type is None:
        args.stim_type = "drone_view" if args.two_eyes else "flyview"
    if args.stim_index is None:
        args.stim_index = [3] if args.two_eyes else [5]

    config = du.load_config(args.config)
    connectome_folder = Path(config['data']['connectome_folder'])

    if args.stim_time is not None:
        config["stimulus"]["stim_time"] = args.stim_time

    # expanding_dot: training config often requests 50 light + 50 dark; movies only
    # need the requested --stim-index rows — trim batch so load is ~1 trial, not 100.
    if args.stim_type == "expanding_dot" and args.stim_index is not None:
        su.trim_expanding_dot_config_for_max_index(config, max(args.stim_index))

    output_folder = Path(args.output_dir) if args.output_dir else (
        Path("figures/two_eyes") if args.two_eyes else Path("figures/movie"))
    output_folder.mkdir(parents=True, exist_ok=True)

    frame_rate = args.fps if args.fps else int(round(1.0 / config['model']['dt']))

    column_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_column_info.parquet')
    column_phi_theta = column_info_df[['phi', 'theta']].to_numpy()

    phi = column_phi_theta[:, 0]
    theta = column_phi_theta[:, 1]
    cell_x = np.sin(theta) * np.cos(phi)
    cell_y = np.sin(theta) * np.sin(phi)
    cell_z = np.cos(theta)

    if args.two_eyes:
        _run_two_eyes(args, config, column_phi_theta, output_folder, frame_rate,
                      cell_x, cell_y, cell_z)
    else:
        _run_one_eye(args, config, column_phi_theta, output_folder, frame_rate,
                     cell_x, cell_y, cell_z)


def _run_one_eye(args, config, column_phi_theta, output_folder, frame_rate,
                 cell_x, cell_y, cell_z):
    print("Loading stimulus and optic flow...")
    stimulus, optic_flow, train_mask = su.get_stimulus_one_eye(
        args.stim_type, config, column_phi_theta)
    stimulus = stimulus.cpu().numpy()
    optic_flow = optic_flow.cpu().numpy()
    train_mask = train_mask.cpu().numpy()

    print(f"Stimulus shape: {stimulus.shape}")
    print(f"Optic flow shape: {optic_flow.shape}")

    for stim_index in args.stim_index:
        print(f"\nCreating movie for stimulus {stim_index} with {stimulus.shape[1]} frames...")
        out_path = output_folder / f'stimulus_flow_stim_{stim_index}.mp4'
        make_stimulus_flow_movie(
            stimulus[stim_index], optic_flow[stim_index],
            cell_x, cell_z, out_path, frame_rate,
            cell_y=cell_y, mask=train_mask[stim_index],
        )


def _run_two_eyes(args, config, column_phi_theta, output_folder, frame_rate,
                  cell_x, cell_y, cell_z):
    from analysis.analysis_utilities import generate_optic_flow

    N = column_phi_theta.shape[0]
    cell_y_left = -cell_y
    model_dt = config["model"]["dt"]
    pre_stim_steps = int(config["stimulus"].get("pre_stim_time", 0) / model_dt)

    print("Loading two-eye stimulus and odometry...")
    stimulus, optic_flow, observer_rotation, observer_translation, train_mask = (
        su.get_stimulus_two_eyes(args.stim_type, config, column_phi_theta)
    )
    stimulus = stimulus.numpy()
    observer_rotation = observer_rotation.numpy()
    observer_translation = observer_translation.numpy()
    train_mask = train_mask.numpy()
    optic_flow = optic_flow.numpy()

    print(f"Stimulus shape: {stimulus.shape}")

    phi = column_phi_theta[:, 0]
    theta = column_phi_theta[:, 1]
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

        if np.all(np.isnan(flow_right[pre_stim_steps:])):
            print("  Ground truth flow unavailable — using predicted flow from odometry (Z=1)")
            omega_t = torch.tensor(
                observer_rotation[stim_index] * np.pi / 180.0, dtype=torch.float32)
            trans_t = torch.tensor(
                observer_translation[stim_index], dtype=torch.float32)
            pred_ur_r, pred_uu_r = generate_optic_flow(phi_right_t, theta_right_t, omega_t, trans_t, 1.0)
            pred_ur_l, pred_uu_l = generate_optic_flow(phi_left_t, theta_right_t, omega_t, trans_t, 1.0)
            flow_right = np.stack([pred_ur_r.numpy(), pred_uu_r.numpy()], axis=-1) * (180.0 / np.pi)
            flow_left = np.stack([pred_ur_l.numpy(), pred_uu_l.numpy()], axis=-1) * (180.0 / np.pi)
            flow_label = "Predicted Flow (Z=1)"
        else:
            flow_label = "Flow"

        _make_two_eyes_input_movie(
            sampled_right=stim_right, sampled_left=stim_left,
            flow_right=flow_right, flow_left=flow_left,
            rotation_rate=observer_rotation[stim_index],
            translation_rate=observer_translation[stim_index],
            flow_label=flow_label,
            cell_x=cell_x, cell_y=cell_y, cell_y_left=cell_y_left, cell_z=cell_z,
            output_path=out_path, frame_rate=frame_rate, model_dt=model_dt,
            mask_right=mask_right, mask_left=mask_left,
        )


def _make_two_eyes_input_movie(
    sampled_right, sampled_left,
    flow_right, flow_left,
    rotation_rate, translation_rate,
    cell_x, cell_y, cell_y_left, cell_z,
    output_path, frame_rate, model_dt,
    flow_label="Flow", trail_frames=60,
    mask_right=None, mask_left=None,
):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import matplotlib.colors as mcolors

    num_frames = sampled_right.shape[0]

    stim_min = min(np.nanmin(sampled_left), np.nanmin(sampled_right))
    stim_max = max(np.nanmax(sampled_left), np.nanmax(sampled_right))
    stim_norm = mcolors.Normalize(vmin=stim_min, vmax=stim_max)
    gray_cmap = plt.cm.gray

    flow_r_vx, flow_r_vz, mag_r, _, _ = prepare_flow_quiver(flow_right, cell_x, cell_y, cell_z)
    flow_l_vx, flow_l_vz, mag_l, _, _ = prepare_flow_quiver(flow_left, cell_x, cell_y_left, cell_z)

    all_vx = np.concatenate([flow_r_vx.ravel(), flow_l_vx.ravel()])
    all_vz = np.concatenate([flow_r_vz.ravel(), flow_l_vz.ravel()])
    _, flow_max_mag = flow_direction_colors(all_vx, all_vz)
    arrow_scale = 0.2 / flow_max_mag if flow_max_mag > 0 else 1.0

    fig, axes = plt.subplots(3, 2, figsize=(12.8, 14), dpi=100,
                             gridspec_kw={"height_ratios": [3, 3, 2]})
    ax_sam_l, ax_sam_r = axes[0, 0], axes[0, 1]
    ax_flow_l, ax_flow_r = axes[1, 0], axes[1, 1]
    ax_rot_rate, ax_trans_rate = axes[2, 0], axes[2, 1]

    setup_eye_axes(ax_sam_l, ax_sam_r, cell_x, cell_z)
    setup_eye_axes(ax_flow_l, ax_flow_r, cell_x, cell_z)

    mask_r_0 = mask_right[0] if mask_right is not None else None
    mask_l_0 = mask_left[0] if mask_left is not None else None

    sc_r = ax_sam_r.scatter(cell_x, cell_z,
        c=masked_rgba(sampled_right[0], mask_r_0, gray_cmap, stim_norm), s=90, edgecolors="none")
    sc_l = ax_sam_l.scatter(cell_x, cell_z,
        c=masked_rgba(sampled_left[0], mask_l_0, gray_cmap, stim_norm), s=90, edgecolors="none")
    ax_sam_r.set_title("Right Eye")
    ax_sam_l.set_title("Left Eye")

    quiver_kw = dict(angles="xy", scale_units="xy", scale=1 / arrow_scale, width=0.003)
    colors_r0, _ = flow_direction_colors(flow_r_vx[0], flow_r_vz[0], flow_max_mag)
    q_r = ax_flow_r.quiver(cell_x, cell_z, flow_r_vx[0], flow_r_vz[0], **quiver_kw)
    q_r.set_facecolor(colors_r0)
    q_r.set_edgecolor(colors_r0)
    ax_flow_r.set_title(f"Right Eye {flow_label}")

    colors_l0, _ = flow_direction_colors(flow_l_vx[0], flow_l_vz[0], flow_max_mag, flip_lr=True)
    q_l = ax_flow_l.quiver(cell_x, cell_z, flow_l_vx[0], flow_l_vz[0], **quiver_kw)
    q_l.set_facecolor(colors_l0)
    q_l.set_edgecolor(colors_l0)
    ax_flow_l.set_title(f"Left Eye {flow_label}")

    add_direction_colorwheel(fig, flip_lr=True)

    rot_labels = ["roll", "pitch", "yaw"]
    trans_labels = ["x", "y", "z"]
    colors = ["#e41a1c", "#377eb8", "#4daf4a"]

    rot_rate_lines = []
    for label, color in zip(rot_labels, colors):
        line, = ax_rot_rate.plot([], [], color=color, label=label, linewidth=1.5)
        rot_rate_lines.append(line)
    ax_rot_rate.legend(loc="upper left", fontsize=8)
    ax_rot_rate.set_ylabel("deg/s")
    ax_rot_rate.set_xlabel("time (s)")
    ax_rot_rate.set_title("Rotation Rate")

    trans_rate_lines = []
    for label, color in zip(trans_labels, colors):
        line, = ax_trans_rate.plot([], [], color=color, label=label, linewidth=1.5)
        trans_rate_lines.append(line)
    ax_trans_rate.legend(loc="upper left", fontsize=8)
    ax_trans_rate.set_ylabel("velocity")
    ax_trans_rate.set_xlabel("time (s)")
    ax_trans_rate.set_title("Translation Rate")

    ax_rot_rate.set_ylim(*ylim_from_data(rotation_rate[:num_frames], min_margin=1.0))
    ax_trans_rate.set_ylim(*ylim_from_data(translation_rate[:num_frames], min_margin=0.01))
    plt.tight_layout()

    times = np.arange(num_frames) * model_dt

    def update(t):
        mask_r_t = mask_right[t] if mask_right is not None else None
        mask_l_t = mask_left[t] if mask_left is not None else None
        sc_r.set_facecolors(masked_rgba(sampled_right[t], mask_r_t, gray_cmap, stim_norm))
        sc_l.set_facecolors(masked_rgba(sampled_left[t], mask_l_t, gray_cmap, stim_norm))

        time_s = t * model_dt
        ax_sam_r.set_title(f"Right Eye (t={time_s:.2f}s)")
        ax_sam_l.set_title(f"Left Eye (t={time_s:.2f}s)")

        update_direction_quiver(q_r, flow_r_vx[t], flow_r_vz[t], flow_max_mag)
        update_direction_quiver(q_l, flow_l_vx[t], flow_l_vz[t], flow_max_mag, flip_lr=True)

        t_start = max(0, t - trail_frames + 1)
        t_slice = slice(t_start, t + 1)
        t_times = times[t_slice]

        for i, line in enumerate(rot_rate_lines):
            line.set_data(t_times, rotation_rate[t_slice, i])
        ax_rot_rate.set_xlim(t_times[0], t_times[0] + trail_frames * model_dt)

        for i, line in enumerate(trans_rate_lines):
            line.set_data(t_times, translation_rate[t_slice, i])
        ax_trans_rate.set_xlim(t_times[0], t_times[0] + trail_frames * model_dt)

    write_movie(fig, update, num_frames, output_path, frame_rate)


if __name__ == "__main__":
    main()
