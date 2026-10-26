"""
Create a movie of stimulus intensity with true optic flow. No model required.

Defaults to two-eye stimuli; pass ``--no-two-eyes`` for a single eye.

One-eye layout (1x2):
  Left: Stimulus intensity (scatter)
  Right: True optic flow (quiver)

Two-eye layout (3x2):
  Row 0: Left eye ommatidia    | Right eye ommatidia
  Row 1: Left eye optic flow   | Right eye optic flow
  Row 2: Rotation rate         | Translation rate

If ground-truth flow is unavailable (all NaN), predicted flow from odometry
(rotation + translation, constant depth Z=1) is shown instead.

Columns outside the field of view are drawn yellow. Movies are written to
``figures/stimulus_flow/`` (relative to the repo root) unless --output-dir is set.

Usage (from the repo root):
    python -m drosophila_model.analysis.movie_scripts.stimulus_flow --stim-type random_pano
    python -m drosophila_model.analysis.movie_scripts.stimulus_flow --no-two-eyes --stim-type flyview
"""
import argparse
from pathlib import Path


import numpy as np
import pandas as pd

from drosophila_model.config import get_data_folder, load_config
from drosophila_model.data_handling import stimulus_generators as su

from drosophila_model.analysis.movie_scripts.movie_utilities import (
    masked_rgba,
    write_movie,
    ylim_from_data,
    make_stimulus_flow_movie,
    synthesize_two_eye_flow_from_odometry,
    prepare_eye_quiver,
    setup_eye_axis,
    init_eye_quiver,
    update_eye_quiver,
    add_eye_colorwheel,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make a movie showing stimulus and true optic flow (no model needed)."
    )
    parser.add_argument("--config-path", "--config", dest="config_path", type=str,
                        default="config.toml",
                        help="Path to config.toml (default: ./config.toml)")
    parser.add_argument("--data-config-path", type=str, default="config_data.toml",
                        help="Path to the stimulus-generation config overlay "
                             "(default: ./config_data.toml)")
    parser.add_argument("--two-eyes", action=argparse.BooleanOptionalAction, default=True,
                        help="Use two-eye stimulus (default: two eyes; "
                             "pass --no-two-eyes for a single eye)")
    parser.add_argument("--stim-index", type=int, nargs="+", default=None,
                        help="Stimulus indices to visualize (default: [5] for 1-eye, [3] for 2-eye)")
    parser.add_argument("--fps", type=int, default=None,
                        help="Output movie frame rate (default: 1/dt from config)")
    parser.add_argument("--stim-type", type=str, default=None,
                        help="Stimulus type (default: flyview)")
    parser.add_argument("--stim-time", type=float, default=None,
                        help="Override stim_time from config (seconds)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for the movie (default: figures/stimulus_flow)")
    args = parser.parse_args()

    if args.stim_type is None:
        args.stim_type = "flyview"
    if args.stim_type == "directional_sine":
        parser.error("--stim-type directional_sine needs a sine direction, which this "
                     "script does not take")
    if args.stim_index is None:
        args.stim_index = [3] if args.two_eyes else [5]

    config = load_config(args.config_path, args.data_config_path)
    # No model is involved: generate the stimulus on the CPU.
    config['model']['device'] = 'cpu'
    data_config = config['data']
    data_folder = get_data_folder(config)

    if args.stim_time is not None:
        data_config["shared"]["stim_time"] = args.stim_time

    # expanding_dot: training config often requests 50 light + 50 dark; movies only
    # need the requested --stim-index rows — trim batch so load is ~1 trial, not 100.
    if args.stim_type == "expanding_dot":
        su.trim_expanding_dot_config_for_max_index(data_config, max(args.stim_index))

    output_folder = Path(args.output_dir) if args.output_dir else Path("figures/stimulus_flow")
    output_folder.mkdir(parents=True, exist_ok=True)

    frame_rate = args.fps if args.fps else int(round(1.0 / config['model']['dt']))

    column_info_df = pd.read_parquet(data_folder / 'connectome_data' / 'baked' / 'optic_lobe_columns.parquet')
    column_phi_theta = column_info_df[['phi', 'theta']].to_numpy()

    phi = column_phi_theta[:, 0]
    theta = column_phi_theta[:, 1]
    cell_x = np.sin(theta) * np.cos(phi)
    cell_y = np.sin(theta) * np.sin(phi)
    cell_z = np.cos(theta)

    if args.two_eyes:
        _run_two_eyes(args, config, data_config, column_phi_theta, output_folder, frame_rate,
                      cell_x, cell_y, cell_z)
    else:
        _run_one_eye(args, config, data_config, column_phi_theta, output_folder, frame_rate,
                     cell_x, cell_y, cell_z)


def _display_mask(train_mask, config, data_config):
    """Field-of-view mask for display from an ``(examples, T, columns)`` train mask.

    The leading pre-stimulus frames are excluded from the loss
    (``train_mask`` False) but are on screen; show them with the field of view
    of the first stimulus frame instead of painting every column as masked.
    """
    pre_stim_steps = int(round(data_config['shared']['pre_stim_time'] / config['model']['dt']))
    mask = train_mask.cpu().numpy().astype(bool)
    mask[:, :pre_stim_steps] = mask[:, pre_stim_steps:pre_stim_steps + 1]
    return mask


def _run_one_eye(args, config, data_config, column_phi_theta, output_folder, frame_rate,
                 cell_x, cell_y, cell_z):
    print("Loading stimulus and optic flow...")
    data = su.get_stimulus(
        args.stim_type, config, column_phi_theta, num_eyes=1, data_config=data_config)
    stimulus = data.stimulus.cpu().numpy()
    optic_flow = data.optic_flow.cpu().numpy()
    train_mask = _display_mask(data.train_mask, config, data_config)

    print(f"Stimulus shape: {stimulus.shape}")
    print(f"Optic flow shape: {optic_flow.shape}")

    for stim_index in args.stim_index:
        if stim_index >= stimulus.shape[0]:
            print(f"Skipping stim {stim_index}: only {stimulus.shape[0]} examples available")
            continue

        print(f"\nCreating movie for stimulus {stim_index} with {stimulus.shape[1]} frames...")
        out_path = output_folder / f'{args.stim_type}_one_eye_stim_{stim_index}.mp4'
        make_stimulus_flow_movie(
            stimulus[stim_index], optic_flow[stim_index],
            cell_x, cell_z, out_path, frame_rate,
            cell_y=cell_y, mask=train_mask[stim_index], eye="right",
        )


def _run_two_eyes(args, config, data_config, column_phi_theta, output_folder, frame_rate,
                  cell_x, cell_y, cell_z):
    N = column_phi_theta.shape[0]
    model_dt = config["model"]["dt"]
    pre_stim_steps = int(round(data_config["shared"]["pre_stim_time"] / model_dt))

    print("Loading two-eye stimulus and odometry...")
    data = su.get_stimulus(args.stim_type, config, column_phi_theta, num_eyes=2,
                           data_config=data_config)
    stimulus = data.stimulus.cpu().numpy()
    observer_rotation = data.rotation.cpu().numpy()
    observer_translation = data.translation.cpu().numpy()
    train_mask = _display_mask(data.train_mask, config, data_config)
    optic_flow = data.optic_flow.cpu().numpy()

    print(f"Stimulus shape: {stimulus.shape}")

    phi = column_phi_theta[:, 0]
    theta = column_phi_theta[:, 1]

    for stim_index in args.stim_index:
        if stim_index >= stimulus.shape[0]:
            print(f"Skipping stim {stim_index}: only {stimulus.shape[0]} examples available")
            continue

        out_path = output_folder / f"{args.stim_type}_two_eyes_stim_{stim_index}.mp4"
        print(f"\nCreating movie for stimulus {stim_index} -> {out_path}")

        stim_right = stimulus[stim_index, :, :N]
        stim_left = stimulus[stim_index, :, N:]
        flow_right = optic_flow[stim_index, :, :N, :]
        flow_left = optic_flow[stim_index, :, N:, :]
        mask_right = train_mask[stim_index, :, :N]
        mask_left = train_mask[stim_index, :, N:]

        if np.all(np.isnan(flow_right[pre_stim_steps:])):
            print("  Ground truth flow unavailable - using predicted flow from odometry (Z=1)")
            flow_right, flow_left = synthesize_two_eye_flow_from_odometry(
                observer_rotation[stim_index], observer_translation[stim_index], phi, theta,
                eye_offset=data.eye_offset)
            flow_label = "Predicted Flow (Z=1)"
        else:
            flow_label = "Flow"

        _make_two_eyes_input_movie(
            sampled_right=stim_right, sampled_left=stim_left,
            flow_right=flow_right, flow_left=flow_left,
            rotation_rate=observer_rotation[stim_index],
            translation_rate=observer_translation[stim_index],
            flow_label=flow_label,
            cell_x=cell_x, cell_y=cell_y, cell_z=cell_z,
            output_path=out_path, frame_rate=frame_rate, model_dt=model_dt,
            mask_right=mask_right, mask_left=mask_left,
        )


def _make_two_eyes_input_movie(
    sampled_right, sampled_left,
    flow_right, flow_left,
    rotation_rate, translation_rate,
    cell_x, cell_y, cell_z,
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

    flow_r_vx, flow_r_vz, _, _, _ = prepare_eye_quiver(flow_right, cell_x, cell_y, cell_z, "right")
    flow_l_vx, flow_l_vz, _, _, _ = prepare_eye_quiver(flow_left, cell_x, cell_y, cell_z, "left")

    all_mag = np.hypot(
        np.concatenate([flow_r_vx.ravel(), flow_l_vx.ravel()]),
        np.concatenate([flow_r_vz.ravel(), flow_l_vz.ravel()]),
    )
    finite_mag = all_mag[np.isfinite(all_mag)]
    flow_max_mag = (float(np.percentile(finite_mag, 99))
                    if finite_mag.size > 0 and finite_mag.max() > 0 else 1.0)
    arrow_scale = 0.2 / flow_max_mag if flow_max_mag > 0 else 1.0

    # 1280x1408 px: both multiples of 16, so the H.264 writer need not resize.
    fig, axes = plt.subplots(3, 2, figsize=(12.8, 14.08), dpi=100,
                             gridspec_kw={"height_ratios": [3, 3, 2]})
    ax_sam_l, ax_sam_r = axes[0, 0], axes[0, 1]
    ax_flow_l, ax_flow_r = axes[1, 0], axes[1, 1]
    ax_rot_rate, ax_trans_rate = axes[2, 0], axes[2, 1]

    for ax in (ax_sam_l, ax_flow_l):
        setup_eye_axis(ax, cell_x, cell_z, "left")
    for ax in (ax_sam_r, ax_flow_r):
        setup_eye_axis(ax, cell_x, cell_z, "right")

    mask_r_0 = mask_right[0] if mask_right is not None else None
    mask_l_0 = mask_left[0] if mask_left is not None else None

    sc_r = ax_sam_r.scatter(cell_x, cell_z,
        c=masked_rgba(sampled_right[0], mask_r_0, gray_cmap, stim_norm), s=90, edgecolors="none")
    sc_l = ax_sam_l.scatter(cell_x, cell_z,
        c=masked_rgba(sampled_left[0], mask_l_0, gray_cmap, stim_norm), s=90, edgecolors="none")
    ax_sam_r.set_title("Right Eye")
    ax_sam_l.set_title("Left Eye")

    quiver_kw = dict(angles="xy", scale_units="xy", scale=1 / arrow_scale, width=0.003)
    q_r = init_eye_quiver(ax_flow_r, cell_x, cell_z, flow_r_vx[0], flow_r_vz[0], flow_max_mag, "right", **quiver_kw)
    ax_flow_r.set_title(f"Right Eye {flow_label}")

    q_l = init_eye_quiver(ax_flow_l, cell_x, cell_z, flow_l_vx[0], flow_l_vz[0], flow_max_mag, "left", **quiver_kw)
    ax_flow_l.set_title(f"Left Eye {flow_label}")

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
    # After tight_layout: the fixed-position polar wheel is not a subplot.
    add_eye_colorwheel(fig)

    times = np.arange(num_frames) * model_dt

    def update(t):
        mask_r_t = mask_right[t] if mask_right is not None else None
        mask_l_t = mask_left[t] if mask_left is not None else None
        sc_r.set_facecolors(masked_rgba(sampled_right[t], mask_r_t, gray_cmap, stim_norm))
        sc_l.set_facecolors(masked_rgba(sampled_left[t], mask_l_t, gray_cmap, stim_norm))

        time_s = t * model_dt
        ax_sam_r.set_title(f"Right Eye (t={time_s:.2f}s)")
        ax_sam_l.set_title(f"Left Eye (t={time_s:.2f}s)")

        update_eye_quiver(q_r, flow_r_vx[t], flow_r_vz[t], flow_max_mag, "right")
        update_eye_quiver(q_l, flow_l_vx[t], flow_l_vz[t], flow_max_mag, "left")

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
