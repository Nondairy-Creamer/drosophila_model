"""
Visualize two-eye stimuli (e.g. copter_view) as a movie.

Layout (2x2):
  Top:    raw left camera frame  |  raw right camera frame
  Bottom: left eye ommatidia     |  right eye ommatidia
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

        _make_two_eye_movie(
            raw_left=raw["left"].numpy(),
            raw_right=raw["right"].numpy(),
            sampled_left=sampled["left"],
            sampled_right=sampled["right"],
            cell_x=cell_x,
            cell_z=cell_z,
            output_path=out_path,
            frame_rate=args.fps,
            num_frames=num_frames,
            video_dt=video_dt,
            model_dt=model_dt,
        )


def _make_two_eye_movie(
    raw_left: np.ndarray,
    raw_right: np.ndarray,
    sampled_left: np.ndarray,
    sampled_right: np.ndarray,
    cell_x: np.ndarray,
    cell_z: np.ndarray,
    output_path: Path,
    frame_rate: int,
    num_frames: int,
    video_dt: float,
    model_dt: float,
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

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.24), dpi=100)
    ax_raw_l, ax_raw_r = axes[0, 0], axes[0, 1]
    ax_sam_l, ax_sam_r = axes[1, 0], axes[1, 1]

    # Raw video panels
    im_l = ax_raw_l.imshow(raw_left[0], cmap="gray", vmin=0, vmax=255, aspect="auto")
    im_r = ax_raw_r.imshow(raw_right[0], cmap="gray", vmin=0, vmax=255, aspect="auto")
    ax_raw_l.set_title("Left Camera")
    ax_raw_r.set_title("Right Camera")
    ax_raw_l.axis("off")
    ax_raw_r.axis("off")

    x_margin = 0.1 * (cell_x.max() - cell_x.min())
    z_margin = 0.1 * (cell_z.max() - cell_z.min())

    # Left eye: anterior (high cell_x) on the right, pointing toward center
    ax_sam_l.set_xlim(cell_x.min() - x_margin, cell_x.max() + x_margin)
    ax_sam_l.set_ylim(cell_z.min() - z_margin, cell_z.max() + z_margin)
    ax_sam_l.set_aspect("equal")

    # Right eye: flip x-axis so anterior points left, toward center
    ax_sam_r.set_xlim(cell_x.max() + x_margin, cell_x.min() - x_margin)
    ax_sam_r.set_ylim(cell_z.min() - z_margin, cell_z.max() + z_margin)
    ax_sam_r.set_aspect("equal")

    sc_l = ax_sam_l.scatter(cell_x, cell_z, c=sampled_left[0], cmap="gray",
                            norm=scatter_norm, s=90, edgecolors="none")
    sc_r = ax_sam_r.scatter(cell_x, cell_z, c=sampled_right[0], cmap="gray",
                            norm=scatter_norm, s=90, edgecolors="none")
    ax_sam_l.set_title("Left Eye (sampled)")
    ax_sam_r.set_title("Right Eye (sampled)")

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {num_frames} frames to {output_path}...")

    with iio.get_writer(str(output_path), codec="libx264", fps=frame_rate) as writer:
        for t in range(num_frames):
            # Find corresponding raw frame for this sampled frame
            raw_t = min(int(round(t * model_dt / video_dt)), raw_left.shape[0] - 1)

            im_l.set_data(raw_left[raw_t])
            im_r.set_data(raw_right[raw_t])
            sc_l.set_array(sampled_left[t])
            sc_r.set_array(sampled_right[t])

            time_s = t * model_dt
            ax_raw_l.set_title(f"Left Camera (t={time_s:.2f}s)")
            ax_raw_r.set_title(f"Right Camera (t={time_s:.2f}s)")
            ax_sam_l.set_title(f"Left Eye (frame {t+1}/{num_frames})")
            ax_sam_r.set_title(f"Right Eye (frame {t+1}/{num_frames})")

            writer.append_data(fig_to_rgb(fig))
            if (t + 1) % 50 == 0:
                print(f"  Frame {t+1}/{num_frames}")

    plt.close(fig)
    print(f"Done! Movie saved to {output_path}")


if __name__ == "__main__":
    main()
