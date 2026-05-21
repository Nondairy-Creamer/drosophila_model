"""
Visualize raw drone CSV data (translation + rotation) alongside the video.

Layout (2x2):
  Row 0:  raw left camera  |  raw right camera
  Row 1:  rotation (roll, pitch, yaw)  |  translation (x, y, z)

No processing — values are read directly from the apriltag_pose CSV.

Usage:
    python analysis/raw_drone_flow.py [--clip-id 32] [--fps 30]
"""
import argparse
import sys
from pathlib import Path

_repo_root = str(Path(__file__).parent.parent)

import numpy as np

from process_drone import load_video_grayscale


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize raw drone CSV data as a movie.")
    parser.add_argument("--data-dir", type=str, default="../connectome_data/drone_videos",
                        help="Directory with left_N.mp4, right_N.mp4, apriltag_pose_log_N_both.csv")
    parser.add_argument("--clip-id", type=int, nargs="+", default=None,
                        help="Clip IDs to visualize (default: all found)")
    parser.add_argument("--fps", type=float, default=None,
                        help="Output movie frame rate (default: native video rate)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum number of video frames to render")
    parser.add_argument("--output-dir", type=str, default="figures/raw_drone",
                        help="Output directory for the movie")
    args = parser.parse_args()

    import csv
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.clip_id is not None:
        clip_ids = args.clip_id
    else:
        clip_ids = sorted(
            int(p.stem.split("_")[-2])
            for p in data_dir.glob("apriltag_pose_log_*_both.csv")
        )

    for clip_id in clip_ids:
        csv_path = data_dir / f"apriltag_pose_log_{clip_id}_both.csv"
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

        raw_left = load_video_grayscale(data_dir / f"left_{clip_id}.mp4")
        raw_right = load_video_grayscale(data_dir / f"right_{clip_id}.mp4")

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
    import imageio as iio
    from analysis_utilities import fig_to_rgb

    n_frames = raw_left.shape[0]
    times = np.arange(n_frames) * video_dt
    trail_sec = 5.0
    trail_frames = int(round(trail_sec / video_dt))

    rot_labels = ["roll", "pitch", "yaw"]
    trans_labels = ["x", "y", "z"]
    colors = ["#e41a1c", "#377eb8", "#4daf4a"]

    def _ylim(data, margin_frac=0.1, min_margin=0.01):
        lo, hi = np.nanmin(data), np.nanmax(data)
        margin = max(margin_frac * (hi - lo), min_margin)
        return lo - margin, hi + margin

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
    for i, (label, color) in enumerate(zip(rot_labels, colors)):
        line_l, = ax_rot.plot([], [], color=color, linewidth=1.5, label=f"L_{label}")
        line_r, = ax_rot.plot([], [], color=color, linewidth=1.5, linestyle="--", label=f"R_{label}")
        rot_lines_L.append(line_l)
        rot_lines_R.append(line_r)
    ax_rot.legend(loc="upper left", fontsize=7, ncol=2)
    ax_rot.set_ylabel("deg")
    ax_rot.set_xlabel("time (s)")
    ax_rot.set_title("Rotation (raw CSV)")
    all_rot = np.concatenate([L_rot, R_rot], axis=0)
    ax_rot.set_ylim(*_ylim(all_rot, min_margin=1.0))

    # Translation panel
    trans_lines_L, trans_lines_R = [], []
    for i, (label, color) in enumerate(zip(trans_labels, colors)):
        line_l, = ax_trans.plot([], [], color=color, linewidth=1.5, label=f"L_{label}")
        line_r, = ax_trans.plot([], [], color=color, linewidth=1.5, linestyle="--", label=f"R_{label}")
        trans_lines_L.append(line_l)
        trans_lines_R.append(line_r)
    ax_trans.legend(loc="upper left", fontsize=7, ncol=2)
    ax_trans.set_ylabel("position")
    ax_trans.set_xlabel("time (s)")
    ax_trans.set_title("Translation (raw CSV)")
    all_trans = np.concatenate([L_trans, R_trans], axis=0)
    ax_trans.set_ylim(*_ylim(all_trans, min_margin=0.01))

    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"  Writing {n_frames} frames at {frame_rate} fps ...")

    with iio.get_writer(str(output_path), codec="libx264", fps=frame_rate) as writer:
        for t in range(n_frames):
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

            writer.append_data(fig_to_rgb(fig))
            if (t + 1) % 100 == 0:
                print(f"    Frame {t+1}/{n_frames}")

    plt.close(fig)
    print(f"  Done! -> {output_path}")


if __name__ == "__main__":
    main()
