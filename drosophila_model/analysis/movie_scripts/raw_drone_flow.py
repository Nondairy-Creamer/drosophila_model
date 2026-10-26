"""
Visualize the legacy small_drone AprilTag pose logs alongside the raw video.

Reads the per-frame ``apriltag_pose_log_<N>_both.csv`` logs that sit next to
the ``left_<N>.mp4`` / ``right_<N>.mp4`` clips in ``stimulus/raw/drone_videos``
and plots the left- and right-camera rotation and translation estimates
unprocessed (no smoothing, no differentiation). This is a raw-data sanity
check only: the modelling pipeline uses the batch-optimised
``drone_poses/pose_<N>.npz`` from ``drone_pose.py`` instead (see
``visualize_drone_data.py`` for the processed equivalent).

Layout (2x2):
  Row 0:  raw left camera  |  raw right camera
  Row 1:  rotation (roll, pitch, yaw)  |  translation (x, y, z)

Usage:
    python -m drosophila_model.analysis.movie_scripts.raw_drone_flow [--clip-id 32] [--fps 30]
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from drosophila_model.config import get_data_folder, load_config
from drosophila_model.data_handling.drone_utilities import load_video_grayscale
from drosophila_model.analysis.movie_scripts.movie_utilities import (
    write_movie,
    ylim_from_data,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize raw drone CSV data as a movie.")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory with left_N.mp4, right_N.mp4, apriltag_pose_log_N_both.csv "
                             "(default: <data_folder>/stimulus/raw/drone_videos)")
    parser.add_argument("--clip-id", type=int, nargs="+", default=None,
                        help="Clip IDs to visualize (default: all found)")
    parser.add_argument("--fps", type=float, default=None,
                        help="Output movie frame rate (default: native video rate)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum number of video frames to render")
    parser.add_argument("--output-dir", type=str, default="figures/raw_drone_flow",
                        help="Output directory for the movie")
    args = parser.parse_args()

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = get_data_folder(load_config()) / 'stimulus' / 'raw' / 'drone_videos'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.clip_id is not None:
        clip_ids = args.clip_id
    else:
        clip_ids = sorted(
            int(p.stem.split("_")[-2])
            for p in data_dir.glob("apriltag_pose_log_*_both.csv")
        )
    if not clip_ids:
        print(f"No apriltag_pose_log_*_both.csv files found in {data_dir}")
        return

    for clip_id in clip_ids:
        csv_path = data_dir / f"apriltag_pose_log_{clip_id}_both.csv"
        if not csv_path.exists():
            print(f"Clip {clip_id}: no {csv_path.name} in {data_dir}, skipping")
            continue
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))

        def col(name):
            return np.array([float(r[name]) for r in rows])

        video_dt = float(rows[1]["t_video_s"]) - float(rows[0]["t_video_s"])

        # Raw rotation and translation from CSV (no smoothing, no differentiation)
        L_rot = np.column_stack([col("L_roll_used_deg"), col("L_pitch_used_deg"), col("L_yaw_used_deg")])
        R_rot = np.column_stack([col("R_roll_used_deg"), col("R_pitch_used_deg"), col("R_yaw_used_deg")])
        L_trans = np.column_stack([col("L_x_used"), col("L_y_used"), col("L_z_used")])
        R_trans = np.column_stack([col("R_x_used"), col("R_y_used"), col("R_z_used")])

        raw_left = load_video_grayscale(data_dir / f"left_{clip_id}.mp4", max_frames=args.max_frames)
        raw_right = load_video_grayscale(data_dir / f"right_{clip_id}.mp4", max_frames=args.max_frames)

        n_frames = min(len(rows), raw_left.shape[0], raw_right.shape[0])
        if args.max_frames is not None:
            n_frames = min(n_frames, args.max_frames)

        fps = args.fps if args.fps is not None else round(1.0 / video_dt)
        out_path = output_dir / f"raw_clip_{clip_id}.mp4"
        print(f"Clip {clip_id}: {n_frames} frames, {fps} fps -> {out_path}")

        _make_raw_movie(
            raw_left[:n_frames], raw_right[:n_frames],
            L_rot[:n_frames], R_rot[:n_frames],
            L_trans[:n_frames], R_trans[:n_frames],
            out_path, fps, video_dt,
        )


def _make_raw_movie(
    raw_left, raw_right,
    L_rot, R_rot, L_trans, R_trans,
    output_path, frame_rate, video_dt,
):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    n_frames = raw_left.shape[0]
    times = np.arange(n_frames) * video_dt
    trail_sec = 5.0
    trail_frames = int(round(trail_sec / video_dt))

    rot_labels = ["roll", "pitch", "yaw"]
    trans_labels = ["x", "y", "z"]
    colors = ["#e41a1c", "#377eb8", "#4daf4a"]

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8), dpi=100,
                             gridspec_kw={"height_ratios": [3, 2]})
    ax_raw_l, ax_raw_r = axes[0]
    ax_rot, ax_trans = axes[1]

    im_l = ax_raw_l.imshow(raw_left[0], cmap="gray", vmin=0, vmax=255, aspect="auto")
    im_r = ax_raw_r.imshow(raw_right[0], cmap="gray", vmin=0, vmax=255, aspect="auto")
    ax_raw_l.set_title("Left Camera")
    ax_raw_r.set_title("Right Camera")
    ax_raw_l.axis("off")
    ax_raw_r.axis("off")

    # Rotation panel — show both L and R as solid/dashed
    rot_lines_L, rot_lines_R = [], []
    for label, color in zip(rot_labels, colors):
        line_l, = ax_rot.plot([], [], color=color, linewidth=1.5, label=f"L_{label}")
        line_r, = ax_rot.plot([], [], color=color, linewidth=1.5, linestyle="--", label=f"R_{label}")
        rot_lines_L.append(line_l)
        rot_lines_R.append(line_r)
    ax_rot.legend(loc="upper left", fontsize=7, ncol=2)
    ax_rot.set_ylabel("deg")
    ax_rot.set_xlabel("time (s)")
    ax_rot.set_title("Rotation (raw CSV)")
    all_rot = np.concatenate([L_rot, R_rot], axis=0)
    ax_rot.set_ylim(*ylim_from_data(all_rot, min_margin=1.0))

    # Translation panel
    trans_lines_L, trans_lines_R = [], []
    for label, color in zip(trans_labels, colors):
        line_l, = ax_trans.plot([], [], color=color, linewidth=1.5, label=f"L_{label}")
        line_r, = ax_trans.plot([], [], color=color, linewidth=1.5, linestyle="--", label=f"R_{label}")
        trans_lines_L.append(line_l)
        trans_lines_R.append(line_r)
    ax_trans.legend(loc="upper left", fontsize=7, ncol=2)
    ax_trans.set_ylabel("position")
    ax_trans.set_xlabel("time (s)")
    ax_trans.set_title("Translation (raw CSV)")
    all_trans = np.concatenate([L_trans, R_trans], axis=0)
    ax_trans.set_ylim(*ylim_from_data(all_trans, min_margin=0.01))

    plt.tight_layout()

    def update(t):
        im_l.set_data(raw_left[t])
        im_r.set_data(raw_right[t])
        ax_raw_l.set_title(f"Left Camera (t={times[t]:.2f}s)")
        ax_raw_r.set_title(f"Right Camera (t={times[t]:.2f}s)")

        t_start = max(0, t - trail_frames + 1)
        t_slice = slice(t_start, t + 1)
        t_times = times[t_slice]

        for i in range(3):
            rot_lines_L[i].set_data(t_times, L_rot[t_slice, i])
            rot_lines_R[i].set_data(t_times, R_rot[t_slice, i])
        ax_rot.set_xlim(t_times[0], t_times[0] + trail_sec)

        for i in range(3):
            trans_lines_L[i].set_data(t_times, L_trans[t_slice, i])
            trans_lines_R[i].set_data(t_times, R_trans[t_slice, i])
        ax_trans.set_xlim(t_times[0], t_times[0] + trail_sec)

    write_movie(fig, update, n_frames, output_path, frame_rate, log_every=100)


if __name__ == "__main__":
    main()
