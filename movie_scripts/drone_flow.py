"""
Visualize two-eyes stimuli (e.g. drone_view) as a movie.

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
import sys
from pathlib import Path

_repo_root = str(Path(__file__).parent.parent)
sys.path.insert(0, _repo_root)
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch

import data_utilities as du
import stimulus_utilities as su
from neural_models import (
    integrate_angular_velocity, integrate_translational_velocity,
)
from scipy.spatial.transform import Rotation


def _load_new_pose(
    npz_path: Path,
    num_frames_sampled: int,
    model_dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load velocity estimates from NPZ, resample, and integrate positions.

    Velocities are resampled to the model time grid, then integrated
    using SO(3)-aware integration from neural_models to obtain position
    and rotation.

    Returns:
        (rotation_deg, position_m, rotation_deg_s, velocity_m_s)
        each (num_frames_sampled, 3).
    """
    data = np.load(npz_path)
    fps = float(data["fps"])
    n_video = data["rotation_deg_s"].shape[0]
    video_times = np.arange(n_video) / fps
    sampled_times = np.arange(num_frames_sampled) * model_dt

    def _resample(arr):
        return np.column_stack([
            np.interp(sampled_times, video_times, arr[:, i])
            for i in range(arr.shape[1])
        ])

    rotation_deg_s = _resample(data["rotation_deg_s"])
    velocity_m_s = _resample(data["velocity_m_s"])

    omega_t = torch.tensor(rotation_deg_s, dtype=torch.float64)
    vel_t = torch.tensor(velocity_m_s, dtype=torch.float64)

    rotvec_deg = integrate_angular_velocity(omega_t, model_dt).numpy()
    rotation_deg = Rotation.from_rotvec(np.deg2rad(rotvec_deg)).as_euler(
        'xyz', degrees=True)
    position_m = integrate_translational_velocity(
        vel_t, omega_t, model_dt).numpy()

    return rotation_deg, position_m, rotation_deg_s, velocity_m_s


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make a movie showing two-eye stimulus with raw video and sampled ommatidia."
    )
    parser.add_argument("--config", type=str, default="config.toml", help="Path to config file")
    parser.add_argument("--stim-index", type=int, nargs="+", default=[0],
                        help="Clip indices to visualize (into the list of processed clips)")
    parser.add_argument("--fps", type=float, default=None,
                        help="Output movie frame rate (default: native video rate)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum number of frames to render (default: all)")
    parser.add_argument("--output-dir", type=str, default="figures/two_eyes",
                        help="Output directory for the movie")
    parser.add_argument("--plot-extra", action="store_true", default=False,
                        help="Show sampled ommatidia and rotation/translation rates")
    parser.add_argument("--pose-dir", type=str, default=None,
                        help="Directory with new pose .npz files (default: "
                             "<connectome_folder>/drone_poses)")
    args = parser.parse_args()

    config = du.load_config(args.config)
    connectome_folder = Path(config["data"]["connectome_folder"])
    output_folder = Path(args.output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)

    column_info_df = pd.read_parquet(connectome_folder / "area_extras" / "optic_lobe_columns.parquet")
    column_phi_theta = column_info_df[["phi", "theta"]].to_numpy()

    model_dt = config["model"]["dt"]
    drone_data_path = connectome_folder / "drone_data.pt"
    drone_video_dir = connectome_folder / "drone_videos"
    pose_dir = Path(args.pose_dir) if args.pose_dir else connectome_folder / "drone_poses"

    print("Loading preprocessed drone clips...")
    drone_clips = su.get_drone_stimulus(
        model_dt, data_path=str(drone_data_path),
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

        from drone_analysis.process_drone import load_video_grayscale
        raw_left = load_video_grayscale(drone_video_dir / f"left_{clip_id}.mp4")
        raw_right = load_video_grayscale(drone_video_dir / f"right_{clip_id}.mp4")
        raw = {"left": torch.from_numpy(raw_left), "right": torch.from_numpy(raw_right)}

        num_frames_raw = raw["left"].shape[0]
        num_frames_sampled = sampled["right"].shape[0]
        num_frames = num_frames_sampled
        if args.max_frames is not None:
            num_frames = min(num_frames, args.max_frames)

        sampled_times = np.arange(num_frames_sampled) * model_dt
        raw_times = np.arange(num_frames_raw) * video_dt

        out_path = output_folder / f"two_eyes_clip_{clip_idx}.mp4"
        print(f"\nClip {clip_idx}: {num_frames_raw} raw frames, "
              f"{num_frames_sampled} sampled frames -> {out_path}")

        in_fov_right = sampled["in_fov_right"].numpy()
        in_fov_left = sampled["in_fov_left"].numpy()

        def _to_numpy(x):
            return x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

        pose_npz = pose_dir / f"pose_{clip_id}.npz"
        if pose_npz.exists():
            rot_pos, trans_pos, rot_rate, trans_rate = _load_new_pose(
                pose_npz, num_frames_sampled, model_dt)
            print(f"  Using new pose estimates from {pose_npz}")
        else:
            rot_pos = _to_numpy(sampled["rotation"])
            trans_pos = _to_numpy(sampled["translation"])
            rot_rate = _to_numpy(sampled["rotation_rate"])
            trans_rate = _to_numpy(sampled["translation_rate"])
            print(f"  Using old pose estimates (no {pose_npz})")

        fps = args.fps if args.fps is not None else round(1.0 / video_dt)

        _make_two_eyes_movie(
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
            frame_rate=fps,
            num_frames=num_frames,
            video_dt=video_dt,
            model_dt=model_dt,
            in_fov_left=in_fov_left,
            in_fov_right=in_fov_right,
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
    import imageio as iio

    from analysis_utilities import fig_to_rgb

    rot_labels = ["roll", "pitch", "yaw"]
    trans_labels = ["x", "y", "z"]
    rot_colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    trans_colors = ["#e41a1c", "#377eb8", "#4daf4a"]

    def _fixed_ylim(data, margin_frac=0.1, min_margin=0.01):
        finite = data[np.isfinite(data)] if np.issubdtype(data.dtype, np.floating) else data.ravel()
        if finite.size == 0:
            return -1, 1
        lo, hi = finite.min(), finite.max()
        margin = max(margin_frac * (hi - lo), min_margin)
        return lo - margin, hi + margin

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
        gray_cmap = plt.cm.gray
        _MASK_COLOR = np.array([1.0, 1.0, 0.0, 1.0])

        def _masked_rgba(values, mask):
            rgba = gray_cmap(scatter_norm(values))
            if mask is not None:
                rgba[~mask] = _MASK_COLOR
            return rgba

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

    # Rotation position
    rot_lines = []
    for label, color in zip(rot_labels, rot_colors):
        line, = ax_rot.plot([], [], color=color, label=label, linewidth=1.5)
        rot_lines.append(line)
    ax_rot.legend(loc="upper left", fontsize=8)
    ax_rot.set_ylabel("deg")
    ax_rot.set_xlabel("time (s)")
    ax_rot.set_title("Rotation")
    ax_rot.set_ylim(*_fixed_ylim(rotation[:num_frames], min_margin=1.0))

    # Translation position
    trans_lines = []
    for label, color in zip(trans_labels, trans_colors):
        line, = ax_trans.plot([], [], color=color, label=label, linewidth=1.5)
        trans_lines.append(line)
    ax_trans.legend(loc="upper left", fontsize=8)
    ax_trans.set_ylabel("position")
    ax_trans.set_xlabel("time (s)")
    ax_trans.set_title("Translation")
    ax_trans.set_ylim(*_fixed_ylim(translation[:num_frames], min_margin=0.01))

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
        ax_rot_rate.set_ylim(*_fixed_ylim(rotation_rate[:num_frames], min_margin=1.0))

        for label, color in zip(trans_labels, trans_colors):
            line, = ax_trans_rate.plot([], [], color=color, label=label, linewidth=1.5)
            trans_rate_lines.append(line)
        ax_trans_rate.legend(loc="upper left", fontsize=8)
        ax_trans_rate.set_ylabel("velocity")
        ax_trans_rate.set_xlabel("time (s)")
        ax_trans_rate.set_title("Translation Rate")
        ax_trans_rate.set_ylim(*_fixed_ylim(translation_rate[:num_frames], min_margin=0.01))

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    step = max(1, int(round(video_dt / model_dt)))
    frame_indices = list(range(0, num_frames, step))
    print(f"Writing {len(frame_indices)} frames to {output_path} "
          f"(step={step}, {frame_rate} fps)...")

    times = np.arange(num_frames) * model_dt

    with iio.get_writer(str(output_path), codec="libx264", fps=frame_rate) as writer:
        for t in frame_indices:
            raw_t = min(int(round(t * model_dt / video_dt)), raw_left.shape[0] - 1)

            im_l.set_data(raw_left[raw_t])
            im_r.set_data(raw_right[raw_t])

            if plot_extra:
                sc_l.set_facecolors(_masked_rgba(sampled_left[t], in_fov_left))
                sc_r.set_facecolors(_masked_rgba(sampled_right[t], in_fov_right))

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

            writer.append_data(fig_to_rgb(fig))
            if (t + 1) % 50 == 0:
                print(f"  Frame {t+1}/{num_frames}")

    plt.close(fig)
    print(f"Done! Movie saved to {output_path}")


if __name__ == "__main__":
    main()
