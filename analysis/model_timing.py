"""Benchmark per-frame model inference in a streaming-style loop.

Loads a raw drone video, samples each frame at ommatidium locations, and
runs the model one frame at a time (carrying state forward) to measure
realistic per-frame latency.

Usage:
    python analysis/model_timing.py models/best_drone/
    python analysis/model_timing.py models/best_drone/ --video-dir ../connectome_data/drone_videos --clip-id 32
    python analysis/model_timing.py models/best_drone/ --max-frames 500
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch

import data_utilities as du
import neural_models as nm
from stimulus_utilities import build_ommatidium_sampling_matrix


def load_video_grayscale(path):
    """Load video as grayscale uint8 frames."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    return np.stack(frames)


def sample_image(image_gray, W, in_fov):
    """Sample a single grayscale image at ommatidium locations.

    Args:
        image_gray: (H, W) uint8 grayscale image
        W: sparse (N_ommatidia, H*W) sampling matrix
        in_fov: (N_ommatidia,) bool mask of ommatidia inside camera FOV

    Returns:
        sampled: (1, 1, N_ommatidia) float32 tensor ready for the model.
    """
    flat = image_gray.ravel().astype(np.float32)
    sampled = np.asarray(W.dot(flat)).ravel() / 255.0
    sampled[~in_fov] = 0.5
    return torch.from_numpy(sampled).unsqueeze(0).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark per-frame model inference on raw drone video."
    )
    parser.add_argument("model_folder", type=str, help="Path to model folder or model.pt")
    parser.add_argument("--video-dir", type=str, default=None,
                        help="Directory with drone videos (default: <connectome_folder>/drone_videos)")
    parser.add_argument("--clip-id", type=int, default=32,
                        help="Which clip to use (default: 32)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Cap the number of frames to process")
    parser.add_argument("--max-time", type=int, default=100,
                        help="Limit video to first N time points (default: 100)")
    parser.add_argument("--filter-std", type=float, default=2.5,
                        help="Gaussian filter std in degrees for ommatidium sampling")
    args = parser.parse_args()

    print(f"Loading model from {args.model_folder}...")
    model = nm.NeuralModel.load(args.model_folder)
    device = model.device
    model.eval()

    # Camera FOV limits (must match process_drone.py)
    phi_right_lim = np.array([-30, -150]) * np.pi / 180
    phi_left_lim = np.array([150, 30]) * np.pi / 180
    theta_lim = np.array([45, 135]) * np.pi / 180

    # Ommatidium sampling coordinates from model
    col_phi_theta = np.asarray(model.column_phi_theta)
    sample_phi_right = col_phi_theta[:, 0]
    sample_theta = col_phi_theta[:, 1]
    sample_phi_left = -sample_phi_right

    # Load raw video
    local_config = du.load_config()
    connectome_folder = Path(local_config["data"]["connectome_folder"])
    video_dir = Path(args.video_dir) if args.video_dir else connectome_folder / "drone_videos"
    right_path = video_dir / f"right_{args.clip_id}.mp4"
    left_path = video_dir / f"left_{args.clip_id}.mp4"

    print(f"Loading video clip {args.clip_id} from {video_dir}...")
    right_frames = load_video_grayscale(right_path)
    left_frames = load_video_grayscale(left_path)

    num_frames = min(len(right_frames), len(left_frames))
    if args.max_frames is not None:
        num_frames = min(num_frames, args.max_frames)
    num_frames = min(num_frames, args.max_time)
    right_frames = right_frames[:num_frames]
    left_frames = left_frames[:num_frames]

    image_height, image_width = right_frames.shape[1], right_frames.shape[2]
    print(f"  {num_frames} frames at {image_height}x{image_width}")

    # Build sampling matrices (one-time cost)
    print("Building sampling matrices...")
    W_right = build_ommatidium_sampling_matrix(
        (image_height, image_width), phi_right_lim, theta_lim,
        sample_phi_right, sample_theta, filter_std=args.filter_std,
    )
    W_left = build_ommatidium_sampling_matrix(
        (image_height, image_width), phi_left_lim, theta_lim,
        sample_phi_left, sample_theta, filter_std=args.filter_std,
    )
    in_fov_right = np.asarray(W_right.sum(axis=1)).ravel() > 0
    in_fov_left = np.asarray(W_left.sum(axis=1)).ravel() > 0

    oob = torch.tensor(
        np.concatenate([~in_fov_right, ~in_fov_left]), device=device
    )

    # Streaming-style loop: one frame at a time, carrying state forward
    print(f"Running streaming inference on {num_frames} frames...")
    state_right = None
    state_left = None
    filter_state_right = None
    filter_state_left = None
    cached_mappings = None
    frame_times = []
    sample_times = []

    with torch.no_grad():
        for fi in range(num_frames):
            t_sample = time.perf_counter()
            right_stim = sample_image(right_frames[fi], W_right, in_fov_right).to(device=device, dtype=model.dtype)
            left_stim = sample_image(left_frames[fi], W_left, in_fov_left).to(device=device, dtype=model.dtype)
            sample_times.append(time.perf_counter() - t_sample)

            t0 = time.perf_counter()
            _, _, r_res, l_res = model.estimate_ego_motion_from_image(
                right_stim, left_stim, oob_columns=oob,
                initial_state_right=state_right,
                initial_state_left=state_left,
                initial_filter_state_right=filter_state_right,
                initial_filter_state_left=filter_state_left,
                cached_mappings=cached_mappings,
            )
            frame_times.append(time.perf_counter() - t0)

            if cached_mappings is None:
                cached_mappings = r_res['mappings']

            state_right = r_res['model_state']['v']
            state_left = l_res['model_state']['v']
            filter_state_right = r_res['model_state']['filter_state']
            filter_state_left = l_res['model_state']['filter_state']

            if (fi + 1) % 200 == 0:
                print(f"  {fi + 1}/{num_frames}")

    frame_times_ms = np.array(frame_times) * 1000
    sample_times_ms = np.array(sample_times) * 1000
    print(f"\nPer-frame timing (n={num_frames}):")
    print(f"  sample_image:")
    print(f"    Mean:   {np.mean(sample_times_ms):.4f} ms")
    print(f"    Median: {np.median(sample_times_ms):.4f} ms")
    print(f"  model inference:")
    print(f"    Mean:   {np.mean(frame_times_ms):.4f} ms")
    print(f"    Median: {np.median(frame_times_ms):.4f} ms")
    print(f"    Std:    {np.std(frame_times_ms):.4f} ms")
    print(f"    Min:    {np.min(frame_times_ms):.4f} ms")
    print(f"    Max:    {np.max(frame_times_ms):.4f} ms")
    total_ms = frame_times_ms + sample_times_ms
    print(f"  total (sample + inference):")
    print(f"    Mean:   {np.mean(total_ms):.4f} ms")
    print(f"    Median: {np.median(total_ms):.4f} ms")

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(frame_times_ms, bins=50, edgecolor="black", alpha=0.75)
    ax.axvline(np.mean(frame_times_ms), color="red", linestyle="--",
               label=f"mean = {np.mean(frame_times_ms):.2f} ms")
    ax.axvline(np.median(frame_times_ms), color="orange", linestyle="--",
               label=f"median = {np.median(frame_times_ms):.2f} ms")
    ax.set_xlabel("Time per frame (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Per-frame streaming inference time")
    ax.legend()
    fig.tight_layout()

    hist_path = Path(__file__).parent.parent / "figures" / "timing_hist.png"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    print(f"\nHistogram saved to {hist_path}")


if __name__ == "__main__":
    main()
