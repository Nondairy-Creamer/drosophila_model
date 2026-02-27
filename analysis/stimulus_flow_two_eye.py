"""
Visualize two-eye stimuli (e.g. copter_view) as a movie.

Layout (4x2):
  Row 0:  raw left camera frame  |  raw right camera frame
  Row 1:  left eye ommatidia     |  right eye ommatidia
  Row 2:  rotation position      |  translation position
  Row 3:  rotation rate          |  translation rate
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make a movie showing two-eye stimulus with raw video and sampled ommatidia."
    )
    parser.add_argument("--config", type=str, default="config.toml", help="Path to config file")
    parser.add_argument("--stim-index", type=int, nargs="+", default=[0],
                        help="Clip indices to visualize (into the list of processed clips)")
    parser.add_argument("--fps", type=int, default=30, help="Output movie frame rate")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum number of frames to render (default: all)")
    parser.add_argument("--output-dir", type=str, default="figures/two_eye",
                        help="Output directory for the movie")
    args = parser.parse_args()

    config = du.load_config(args.config)
    connectome_folder = Path(config["data"]["connectome_folder"])
    output_folder = Path(args.output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)

    column_info_df = pd.read_parquet(connectome_folder / "optic_lobes_column_info.parquet")
    column_phi_theta = column_info_df[["phi", "theta"]].to_numpy()

    model_dt = config["model"]["dt"]
    copter_data_path = connectome_folder / "copter_data.pt"
    copter_video_dir = connectome_folder / "copter_videos"

    print("Loading preprocessed copter clips...")
    copter_clips = su.get_copter_stimulus(
        model_dt, data_path=str(copter_data_path),
    )
    print(f"Found {len(copter_clips)} clips")

    phi = column_phi_theta[:, 0]  # negative for right eye
    theta = column_phi_theta[:, 1]
    cell_x = np.sin(theta) * np.cos(phi)
    cell_z = np.cos(theta)

    for clip_idx in args.stim_index:
        if clip_idx >= len(copter_clips):
            print(f"Skipping clip {clip_idx}: only {len(copter_clips)} clips available")
            continue

        sampled = copter_clips[clip_idx]
        clip_id = sampled["clip_id"]
        video_dt = sampled["video_dt"]

        from process_copter import process_clip
        raw = process_clip(clip_id, copter_video_dir)
        if raw is None:
            print(f"Skipping clip {clip_idx}: could not load raw video")
            continue

        num_frames_raw = raw["left"].shape[0]
        num_frames_sampled = sampled["right"].shape[0]
        num_frames = min(num_frames_raw, num_frames_sampled)
        if args.max_frames is not None:
            num_frames = min(num_frames, args.max_frames)

        sampled_times = np.arange(num_frames_sampled) * model_dt
        raw_times = np.arange(num_frames_raw) * video_dt

        out_path = output_folder / f"two_eye_clip_{clip_idx}.mp4"
        print(f"\nClip {clip_idx}: {num_frames_raw} raw frames, "
              f"{num_frames_sampled} sampled frames -> {out_path}")

        in_fov_right = sampled["in_fov_right"].numpy()
        in_fov_left = sampled["in_fov_left"].numpy()

        def _to_numpy(x):
            return x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

        rot_pos = _to_numpy(sampled["rotation"])
        trans_pos = _to_numpy(sampled["translation"])
        rot_rate = _to_numpy(sampled["rotation_rate"])
        trans_rate = _to_numpy(sampled["translation_rate"])

        _make_two_eye_movie(
            raw_left=raw["left"].numpy(),
            raw_right=raw["right"].numpy(),
            sampled_left=sampled["left"],
            sampled_right=sampled["right"],
            rotation=rot_pos,
            translation=trans_pos,
            rotation_rate=rot_rate,
            translation_rate=trans_rate,
            cell_x=cell_x,
            cell_z=cell_z,
            output_path=out_path,
            frame_rate=args.fps,
            num_frames=num_frames,
            video_dt=video_dt,
            model_dt=model_dt,
            in_fov_left=in_fov_left,
            in_fov_right=in_fov_right,
        )


def _make_two_eye_movie(
    raw_left: np.ndarray,
    raw_right: np.ndarray,
    sampled_left: np.ndarray,
    sampled_right: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    rotation_rate: np.ndarray,
    translation_rate: np.ndarray,
    cell_x: np.ndarray,
    cell_z: np.ndarray,
    output_path: Path,
    frame_rate: int,
    num_frames: int,
    video_dt: float,
    model_dt: float,
    trail_frames: int = 60,
    in_fov_left: np.ndarray | None = None,
    in_fov_right: np.ndarray | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import matplotlib.colors as mcolors
    import imageio as iio

    from analysis_utilities import fig_to_rgb

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

    fig, axes = plt.subplots(4, 2, figsize=(12.8, 16), dpi=100,
                             gridspec_kw={"height_ratios": [3, 3, 2, 2]})
    ax_raw_l, ax_raw_r = axes[0, 0], axes[0, 1]
    ax_sam_l, ax_sam_r = axes[1, 0], axes[1, 1]
    ax_rot, ax_trans = axes[2, 0], axes[2, 1]
    ax_rot_rate, ax_trans_rate = axes[3, 0], axes[3, 1]

    # Raw video panels
    im_l = ax_raw_l.imshow(raw_left[0], cmap="gray", vmin=0, vmax=255, aspect="auto")
    im_r = ax_raw_r.imshow(raw_right[0], cmap="gray", vmin=0, vmax=255, aspect="auto")
    ax_raw_l.set_title("Left Camera")
    ax_raw_r.set_title("Right Camera")
    ax_raw_l.axis("off")
    ax_raw_r.axis("off")

    x_margin = 0.1 * (cell_x.max() - cell_x.min())
    z_margin = 0.1 * (cell_z.max() - cell_z.min())

    ax_sam_l.set_xlim(cell_x.min() - x_margin, cell_x.max() + x_margin)
    ax_sam_l.set_ylim(cell_z.min() - z_margin, cell_z.max() + z_margin)
    ax_sam_l.set_aspect("equal")

    ax_sam_r.set_xlim(cell_x.max() + x_margin, cell_x.min() - x_margin)
    ax_sam_r.set_ylim(cell_z.min() - z_margin, cell_z.max() + z_margin)
    ax_sam_r.set_aspect("equal")

    sc_l = ax_sam_l.scatter(cell_x, cell_z, c=_masked_rgba(sampled_left[0], in_fov_left),
                            s=90, edgecolors="none")
    sc_r = ax_sam_r.scatter(cell_x, cell_z, c=_masked_rgba(sampled_right[0], in_fov_right),
                            s=90, edgecolors="none")
    ax_sam_l.set_title("Left Eye (sampled)")
    ax_sam_r.set_title("Right Eye (sampled)")

    # Odometry panels — initialize with empty data
    rot_labels = ["roll", "pitch", "yaw"]
    trans_labels = ["x", "y", "z"]
    rot_colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    trans_colors = ["#e41a1c", "#377eb8", "#4daf4a"]

    rot_lines = []
    for i, (label, color) in enumerate(zip(rot_labels, rot_colors)):
        line, = ax_rot.plot([], [], color=color, label=label, linewidth=1.5)
        rot_lines.append(line)
    ax_rot.legend(loc="upper left", fontsize=8)
    ax_rot.set_ylabel("deg")
    ax_rot.set_xlabel("time (s)")
    ax_rot.set_title("Rotation")

    trans_lines = []
    for i, (label, color) in enumerate(zip(trans_labels, trans_colors)):
        line, = ax_trans.plot([], [], color=color, label=label, linewidth=1.5)
        trans_lines.append(line)
    ax_trans.legend(loc="upper left", fontsize=8)
    ax_trans.set_ylabel("position")
    ax_trans.set_xlabel("time (s)")
    ax_trans.set_title("Translation")

    rot_rate_lines = []
    for i, (label, color) in enumerate(zip(rot_labels, rot_colors)):
        line, = ax_rot_rate.plot([], [], color=color, label=label, linewidth=1.5)
        rot_rate_lines.append(line)
    ax_rot_rate.legend(loc="upper left", fontsize=8)
    ax_rot_rate.set_ylabel("deg/s")
    ax_rot_rate.set_xlabel("time (s)")
    ax_rot_rate.set_title("Rotation Rate")

    trans_rate_lines = []
    for i, (label, color) in enumerate(zip(trans_labels, trans_colors)):
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

    ax_rot.set_ylim(*_fixed_ylim(rotation[:num_frames], min_margin=1.0))
    ax_trans.set_ylim(*_fixed_ylim(translation[:num_frames], min_margin=0.01))
    ax_rot_rate.set_ylim(*_fixed_ylim(rotation_rate[:num_frames], min_margin=1.0))
    ax_trans_rate.set_ylim(*_fixed_ylim(translation_rate[:num_frames], min_margin=0.01))

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {num_frames} frames to {output_path}...")

    times = np.arange(num_frames) * model_dt

    with iio.get_writer(str(output_path), codec="libx264", fps=frame_rate) as writer:
        for t in range(num_frames):
            raw_t = min(int(round(t * model_dt / video_dt)), raw_left.shape[0] - 1)

            im_l.set_data(raw_left[raw_t])
            im_r.set_data(raw_right[raw_t])
            sc_l.set_facecolors(_masked_rgba(sampled_left[t], in_fov_left))
            sc_r.set_facecolors(_masked_rgba(sampled_right[t], in_fov_right))

            time_s = t * model_dt
            ax_raw_l.set_title(f"Left Camera (t={time_s:.2f}s)")
            ax_raw_r.set_title(f"Right Camera (t={time_s:.2f}s)")
            ax_sam_l.set_title(f"Left Eye (frame {t+1}/{num_frames})")
            ax_sam_r.set_title(f"Right Eye (frame {t+1}/{num_frames})")

            # Update odometry trailing window
            t_start = max(0, t - trail_frames + 1)
            t_slice = slice(t_start, t + 1)
            t_times = times[t_slice]

            for i, line in enumerate(rot_lines):
                line.set_data(t_times, rotation[t_slice, i])
            ax_rot.set_xlim(t_times[0], t_times[0] + trail_frames * model_dt)

            for i, line in enumerate(trans_lines):
                line.set_data(t_times, translation[t_slice, i])
            ax_trans.set_xlim(t_times[0], t_times[0] + trail_frames * model_dt)

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
