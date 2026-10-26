"""
Visualize processed two-eye drone data as a movie.

Loads the baked ommatidium-sampled clips for --dataset via
``drone_utilities.get_drone_stimulus`` (``<data_folder>/stimulus/baked/<dataset>.pt``,
built by ``process_raw_data.py``) and shows them next to the raw videos, which
are read only for display:
  * small_drone (equirectangular rig, raw videos in stimulus/raw/drone_videos)
  * rpi_drone   (dome rig, raw videos in stimulus/raw/rpi_drone/<recording>/)

The rotation and translation panels show the baked ground-truth rates and
their integrals (rotation as a rotation vector in degrees).

Example:
    python -m drosophila_model.analysis.movie_scripts.visualize_drone_data --dataset rpi_drone --plot-extra

Default layout (2x2):
  Row 0:  raw left camera frame  |  raw right camera frame
  Row 1:  rotation position      |  translation position

With --plot-extra (4x2):
  Row 0:  raw left camera frame  |  raw right camera frame
  Row 1:  left eye ommatidia     |  right eye ommatidia
  Row 2:  rotation position      |  translation position
  Row 3:  rotation rate          |  translation rate
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from drosophila_model.config import get_data_folder, load_config
from drosophila_model.data_handling import drone_utilities as du
from drosophila_model.analysis.movie_scripts.movie_utilities import (
    masked_rgba,
    write_movie,
    ylim_from_data,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make a movie showing two-eye stimulus with raw video and sampled ommatidia."
    )
    parser.add_argument("--config-path", "--config", dest="config_path", type=str,
                        default="config.toml", help="Path to config file")
    parser.add_argument("--dataset", type=str, default="small_drone",
                        choices=["small_drone", "rpi_drone"],
                        help="Which baked drone dataset to visualize")
    parser.add_argument("--stim-index", type=int, nargs="+", default=[0],
                        help="Clip indices to visualize (into the list of processed clips)")
    parser.add_argument("--fps", type=float, default=None,
                        help="Output movie frame rate (default: native video rate)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum number of frames to render (default: all)")
    parser.add_argument("--output-dir", type=str, default="figures/visualize_drone_data",
                        help="Output directory for the movie")
    parser.add_argument("--plot-extra", action="store_true", default=False,
                        help="Show sampled ommatidia and rotation/translation rates")
    args = parser.parse_args()

    config = load_config(args.config_path)
    data_folder = get_data_folder(config)
    output_folder = Path(args.output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)

    column_info_df = pd.read_parquet(data_folder / "connectome_data" / "baked" / "optic_lobe_columns.parquet")
    column_phi_theta = column_info_df[["phi", "theta"]].to_numpy()

    model_dt = config["model"]["dt"]
    raw_dir = data_folder / "stimulus" / "raw"

    print(f"Loading baked drone clips (dataset={args.dataset})...")
    drone_clips = du.get_drone_stimulus(
        model_dt, dataset=args.dataset, data_folder=data_folder,
        sample_phi=column_phi_theta[:, 0], sample_theta=column_phi_theta[:, 1],
    )
    print(f"Found {len(drone_clips)} clips")

    phi = column_phi_theta[:, 0]  # negative for right eye
    theta = column_phi_theta[:, 1]
    cell_x = np.sin(theta) * np.cos(phi)
    cell_z = np.cos(theta)

    for clip_idx in args.stim_index:
        if clip_idx >= len(drone_clips):
            print(f"Skipping clip {clip_idx}: only {len(drone_clips)} clips available")
            continue

        sampled = drone_clips[clip_idx]
        clip_id = sampled["clip_id"]
        video_dt = sampled["video_dt"]

        num_frames_sampled = sampled["right"].shape[0]
        num_frames = num_frames_sampled
        raw_cap = None
        if args.max_frames is not None:
            num_frames = min(num_frames, args.max_frames)
            # Decode only the raw frames the movie shows (see update()).
            raw_cap = int(round((num_frames - 1) * model_dt / video_dt)) + 1

        if args.dataset == "rpi_drone":
            rec_dir = raw_dir / "rpi_drone" / str(clip_id)
            raw_left = du.load_video_grayscale(rec_dir / "left.mp4", max_frames=raw_cap)
            raw_right = du.load_video_grayscale(rec_dir / "right.mp4", max_frames=raw_cap)
        else:
            raw_left = du.load_video_grayscale(
                raw_dir / "drone_videos" / f"left_{clip_id}.mp4", max_frames=raw_cap)
            raw_right = du.load_video_grayscale(
                raw_dir / "drone_videos" / f"right_{clip_id}.mp4", max_frames=raw_cap)

        num_frames_raw = raw_left.shape[0]

        out_path = output_folder / f"{args.dataset}_clip_{clip_idx}.mp4"
        print(f"\nClip {clip_idx}: {num_frames_raw} raw frames decoded, "
              f"{num_frames_sampled} sampled frames -> {out_path}")

        fps = args.fps if args.fps is not None else round(1.0 / video_dt)

        _make_two_eyes_movie(
            raw_left=raw_left,
            raw_right=raw_right,
            sampled_left=sampled["left"],
            sampled_right=sampled["right"],
            rotation=np.asarray(sampled["rotation"]),
            translation=np.asarray(sampled["translation"]),
            rotation_rate=np.asarray(sampled["rotation_rate"]),
            translation_rate=np.asarray(sampled["translation_rate"]),
            cell_x=cell_x,
            cell_z=cell_z,
            output_path=out_path,
            frame_rate=fps,
            num_frames=num_frames,
            video_dt=video_dt,
            model_dt=model_dt,
            in_fov_left=np.asarray(sampled["in_fov_left"]),
            in_fov_right=np.asarray(sampled["in_fov_right"]),
            plot_extra=args.plot_extra,
        )


def _make_two_eyes_movie(
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
    plot_extra: bool = False,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import matplotlib.colors as mcolors

    rot_labels = ["roll", "pitch", "yaw"]
    trans_labels = ["x", "y", "z"]
    rot_colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    trans_colors = ["#e41a1c", "#377eb8", "#4daf4a"]

    if plot_extra:
        # 4x2: raw video, sampled ommatidia, rotation/translation pos, rates
        fig, axes = plt.subplots(4, 2, figsize=(12.8, 16), dpi=100,
                                 gridspec_kw={"height_ratios": [3, 3, 2, 2]})
        ax_raw_l, ax_raw_r = axes[0, 0], axes[0, 1]
        ax_sam_l, ax_sam_r = axes[1, 0], axes[1, 1]
        ax_rot, ax_trans = axes[2, 0], axes[2, 1]
        ax_rot_rate, ax_trans_rate = axes[3, 0], axes[3, 1]
    else:
        # 2x2: raw video, rotation/translation position only
        fig, axes = plt.subplots(2, 2, figsize=(12.8, 8), dpi=100,
                                 gridspec_kw={"height_ratios": [3, 2]})
        ax_raw_l, ax_raw_r = axes[0, 0], axes[0, 1]
        ax_rot, ax_trans = axes[1, 0], axes[1, 1]

    # Raw video panels
    im_l = ax_raw_l.imshow(raw_left[0], cmap="gray", vmin=0, vmax=255, aspect="auto")
    im_r = ax_raw_r.imshow(raw_right[0], cmap="gray", vmin=0, vmax=255, aspect="auto")
    ax_raw_l.set_title("Left Camera")
    ax_raw_r.set_title("Right Camera")
    ax_raw_l.axis("off")
    ax_raw_r.axis("off")

    # Sampled ommatidia (only when plot_extra)
    sc_l = sc_r = None
    if plot_extra:
        vmin = min(np.nanmin(sampled_left), np.nanmin(sampled_right))
        vmax = max(np.nanmax(sampled_left), np.nanmax(sampled_right))
        scatter_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        x_margin = 0.1 * (cell_x.max() - cell_x.min())
        z_margin = 0.1 * (cell_z.max() - cell_z.min())

        ax_sam_l.set_xlim(cell_x.min() - x_margin, cell_x.max() + x_margin)
        ax_sam_l.set_ylim(cell_z.min() - z_margin, cell_z.max() + z_margin)
        ax_sam_l.set_aspect("equal")

        ax_sam_r.set_xlim(cell_x.max() + x_margin, cell_x.min() - x_margin)
        ax_sam_r.set_ylim(cell_z.min() - z_margin, cell_z.max() + z_margin)
        ax_sam_r.set_aspect("equal")

        sc_l = ax_sam_l.scatter(cell_x, cell_z,
                                c=masked_rgba(sampled_left[0], in_fov_left, norm=scatter_norm),
                                s=90, edgecolors="none")
        sc_r = ax_sam_r.scatter(cell_x, cell_z,
                                c=masked_rgba(sampled_right[0], in_fov_right, norm=scatter_norm),
                                s=90, edgecolors="none")
        ax_sam_l.set_title("Left Eye (sampled)")
        ax_sam_r.set_title("Right Eye (sampled)")

    # Rotation position (integrated rotation vector)
    rot_lines = []
    for label, color in zip(rot_labels, rot_colors):
        line, = ax_rot.plot([], [], color=color, label=label, linewidth=1.5)
        rot_lines.append(line)
    ax_rot.legend(loc="upper left", fontsize=8)
    ax_rot.set_ylabel("deg")
    ax_rot.set_xlabel("time (s)")
    ax_rot.set_title("Rotation (rotation vector)")
    ax_rot.set_ylim(*ylim_from_data(rotation[:num_frames], min_margin=1.0))

    # Translation position
    trans_lines = []
    for label, color in zip(trans_labels, trans_colors):
        line, = ax_trans.plot([], [], color=color, label=label, linewidth=1.5)
        trans_lines.append(line)
    ax_trans.legend(loc="upper left", fontsize=8)
    ax_trans.set_ylabel("position")
    ax_trans.set_xlabel("time (s)")
    ax_trans.set_title("Translation")
    ax_trans.set_ylim(*ylim_from_data(translation[:num_frames], min_margin=0.01))

    # Rate panels (only when plot_extra)
    rot_rate_lines = []
    trans_rate_lines = []
    if plot_extra:
        for label, color in zip(rot_labels, rot_colors):
            line, = ax_rot_rate.plot([], [], color=color, label=label, linewidth=1.5)
            rot_rate_lines.append(line)
        ax_rot_rate.legend(loc="upper left", fontsize=8)
        ax_rot_rate.set_ylabel("deg/s")
        ax_rot_rate.set_xlabel("time (s)")
        ax_rot_rate.set_title("Rotation Rate")
        ax_rot_rate.set_ylim(*ylim_from_data(rotation_rate[:num_frames], min_margin=1.0))

        for label, color in zip(trans_labels, trans_colors):
            line, = ax_trans_rate.plot([], [], color=color, label=label, linewidth=1.5)
            trans_rate_lines.append(line)
        ax_trans_rate.legend(loc="upper left", fontsize=8)
        ax_trans_rate.set_ylabel("velocity")
        ax_trans_rate.set_xlabel("time (s)")
        ax_trans_rate.set_title("Translation Rate")
        ax_trans_rate.set_ylim(*ylim_from_data(translation_rate[:num_frames], min_margin=0.01))

    plt.tight_layout()

    # One rendered frame per raw video frame (at the nearest model step), so the
    # native-rate movie plays in real time even when video_dt is not a whole
    # number of model steps (rpi_drone: 20 fps video, 30 Hz model).
    num_video_frames = int(np.ceil(num_frames * model_dt / video_dt - 1e-6))
    frame_indices = [t for t in (int(round(k * video_dt / model_dt))
                                 for k in range(num_video_frames))
                     if t < num_frames]
    print(f"Rendering {len(frame_indices)} frames (one per raw video frame)")

    times = np.arange(num_frames) * model_dt

    def update(k):
        t = frame_indices[k]
        raw_t = min(int(round(t * model_dt / video_dt)), raw_left.shape[0] - 1)

        im_l.set_data(raw_left[raw_t])
        im_r.set_data(raw_right[raw_t])

        if plot_extra:
            sc_l.set_facecolors(masked_rgba(sampled_left[t], in_fov_left, norm=scatter_norm))
            sc_r.set_facecolors(masked_rgba(sampled_right[t], in_fov_right, norm=scatter_norm))

        time_s = t * model_dt
        ax_raw_l.set_title(f"Left Camera (t={time_s:.2f}s)")
        ax_raw_r.set_title(f"Right Camera (t={time_s:.2f}s)")

        if plot_extra:
            ax_sam_l.set_title(f"Left Eye (frame {t+1}/{num_frames})")
            ax_sam_r.set_title(f"Right Eye (frame {t+1}/{num_frames})")

        t_start = max(0, t - trail_frames + 1)
        t_slice = slice(t_start, t + 1)
        t_times = times[t_slice]

        for i, line in enumerate(rot_lines):
            line.set_data(t_times, rotation[t_slice, i])
        ax_rot.set_xlim(t_times[0], t_times[0] + trail_frames * model_dt)

        for i, line in enumerate(trans_lines):
            line.set_data(t_times, translation[t_slice, i])
        ax_trans.set_xlim(t_times[0], t_times[0] + trail_frames * model_dt)

        if plot_extra:
            for i, line in enumerate(rot_rate_lines):
                line.set_data(t_times, rotation_rate[t_slice, i])
            ax_rot_rate.set_xlim(t_times[0], t_times[0] + trail_frames * model_dt)

            for i, line in enumerate(trans_rate_lines):
                line.set_data(t_times, translation_rate[t_slice, i])
            ax_trans_rate.set_xlim(t_times[0], t_times[0] + trail_frames * model_dt)

    write_movie(fig, update, len(frame_indices), output_path, frame_rate)


if __name__ == "__main__":
    main()
