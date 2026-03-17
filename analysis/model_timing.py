"""Benchmark per-frame model inference in a streaming-style loop.

Loads stimulus data and runs the model one frame at a time (carrying state
forward) to measure realistic per-frame latency.

Two stimulus sources are supported (tried in order):
  1. Raw drone video  -- samples each frame at ommatidium locations.
  2. Pre-computed cell_inputs.pt from the model folder -- used when raw
     images are not available.

Usage:
    python analysis/model_timing.py models/best_drone/
    python analysis/model_timing.py models/best_drone/ --video-dir ../connectome_data/drone_videos --clip-id 32
    python analysis/model_timing.py models/best_drone/ --max-frames 500
"""
import argparse
import os
import platform
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

import data_utilities as du
import neural_models as nm


def _fmt_bytes(n_bytes):
    """Format byte count as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n_bytes) < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def _peak_rss():
    """Peak RSS of the current process in bytes."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return rss  # already bytes on macOS
    return rss * 1024  # KB on Linux


def _tensor_bytes(t):
    """Total storage bytes of a single tensor."""
    return t.nelement() * t.element_size()


def _model_memory(model):
    """Return (param_bytes, buffer_bytes) for a torch model."""
    param_bytes = sum(_tensor_bytes(p) for p in model.parameters())
    buffer_bytes = sum(_tensor_bytes(b) for b in model.buffers())
    return param_bytes, buffer_bytes


def load_video_grayscale(path, max_frames=None):
    """Load video as grayscale uint8 frames."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    frames = []
    while max_frames is None or len(frames) < max_frames:
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


def _try_load_video(args, model):
    """Attempt to load drone video frames. Returns None on failure."""
    try:
        from stimulus_utilities import build_ommatidium_sampling_matrix
    except ImportError:
        return None

    local_config = du.load_config()
    connectome_folder = Path(local_config["data"]["connectome_folder"])
    video_dir = Path(args.video_dir) if args.video_dir else connectome_folder / "drone_videos"
    right_path = video_dir / f"right_{args.clip_id}.mp4"
    left_path = video_dir / f"left_{args.clip_id}.mp4"

    if not right_path.exists() or not left_path.exists():
        return None

    try:
        import cv2  # noqa: F401
    except ImportError:
        return None

    frame_cap = args.max_time
    if args.max_frames is not None:
        frame_cap = min(frame_cap, args.max_frames)

    print(f"Loading video clip {args.clip_id} from {video_dir}...")
    right_frames = load_video_grayscale(right_path, max_frames=frame_cap)
    left_frames = load_video_grayscale(left_path, max_frames=frame_cap)

    phi_right_lim = np.array([-30, -150]) * np.pi / 180
    phi_left_lim = np.array([150, 30]) * np.pi / 180
    theta_lim = np.array([45, 135]) * np.pi / 180

    col_phi_theta = np.asarray(model.column_phi_theta)
    sample_phi_right = col_phi_theta[:, 0]
    sample_theta = col_phi_theta[:, 1]
    sample_phi_left = -sample_phi_right

    print("Building sampling matrices...")
    W_right = build_ommatidium_sampling_matrix(
        (right_frames.shape[1], right_frames.shape[2]), phi_right_lim, theta_lim,
        sample_phi_right, sample_theta, filter_std=args.filter_std,
    )
    W_left = build_ommatidium_sampling_matrix(
        (left_frames.shape[1], left_frames.shape[2]), phi_left_lim, theta_lim,
        sample_phi_left, sample_theta, filter_std=args.filter_std,
    )
    in_fov_right = np.asarray(W_right.sum(axis=1)).ravel() > 0
    in_fov_left = np.asarray(W_left.sum(axis=1)).ravel() > 0

    return {
        'right_frames': right_frames,
        'left_frames': left_frames,
        'W_right': W_right,
        'W_left': W_left,
        'in_fov_right': in_fov_right,
        'in_fov_left': in_fov_left,
    }


def _run_video_benchmark(video_data, model, device, rss_checkpoints):
    """Run streaming benchmark from raw video frames. Returns timing arrays and extra info."""
    right_frames = video_data['right_frames']
    left_frames = video_data['left_frames']
    W_right, W_left = video_data['W_right'], video_data['W_left']
    in_fov_right, in_fov_left = video_data['in_fov_right'], video_data['in_fov_left']

    num_frames = min(len(right_frames), len(left_frames))
    rss_checkpoints["after video load"] = _peak_rss()

    image_height, image_width = right_frames.shape[1], right_frames.shape[2]
    raw_frame_bytes = int(right_frames[0].nbytes)
    total_video_bytes = int(right_frames.nbytes + left_frames.nbytes)
    print(f"  {num_frames} frames at {image_height}x{image_width}")
    print(f"  Single raw frame: {_fmt_bytes(raw_frame_bytes)} ({image_height}x{image_width}, uint8)")
    print(f"  Total video loaded: {_fmt_bytes(total_video_bytes)} ({num_frames} frames x 2 eyes)")

    rss_checkpoints["after sampling matrices"] = _peak_rss()

    oob = torch.tensor(
        np.concatenate([~in_fov_right, ~in_fov_left]), device=device
    )

    print(f"Running streaming inference on {num_frames} frames...")
    state_right = None
    state_left = None
    filter_state_right = None
    filter_state_left = None
    cached_mappings = None
    frame_times = []
    sample_times = []
    stimulus_bytes = None

    with torch.no_grad():
        for fi in range(num_frames):
            t_sample = time.perf_counter()
            right_stim = sample_image(right_frames[fi], W_right, in_fov_right).to(device=device, dtype=model.dtype)
            left_stim = sample_image(left_frames[fi], W_left, in_fov_left).to(device=device, dtype=model.dtype)
            sample_times.append(time.perf_counter() - t_sample)

            if stimulus_bytes is None:
                stimulus_bytes = _tensor_bytes(right_stim) + _tensor_bytes(left_stim)

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

    return {
        'frame_times': frame_times,
        'sample_times': sample_times,
        'num_frames': num_frames,
        'stimulus_bytes': stimulus_bytes,
        'video_info': {
            'image_height': image_height,
            'image_width': image_width,
            'raw_frame_bytes': raw_frame_bytes,
            'total_video_bytes': total_video_bytes,
            'right_stim_shape': tuple(right_stim.shape),
            'right_stim_dtype': right_stim.dtype,
        },
    }


def _run_cell_inputs_benchmark(cell_inputs_path, model, device, args, rss_checkpoints):
    """Run streaming benchmark from pre-computed cell_inputs.pt."""
    print(f"Loading stimulus from {cell_inputs_path}...")
    data = torch.load(cell_inputs_path, map_location="cpu", weights_only=True)
    stimulus = data['cell_inputs']
    train_mask = data['train_mask']
    rss_checkpoints["after stimulus load"] = _peak_rss()

    n_cols = model.num_columns
    is_two_eye = stimulus.shape[2] == n_cols * 2

    frame_cap = args.max_time
    if args.max_frames is not None:
        frame_cap = min(frame_cap, args.max_frames)
    num_time = min(stimulus.shape[1], frame_cap)
    batch_idx = 0

    if is_two_eye:
        right_stim_all = stimulus[batch_idx:batch_idx+1, :num_time, :n_cols].to(device=device, dtype=model.dtype)
        left_stim_all = stimulus[batch_idx:batch_idx+1, :num_time, n_cols:].to(device=device, dtype=model.dtype)
        oob = ~train_mask.any(dim=1).any(dim=0).to(device=device)
        print(f"  Two-eye stimulus: {stimulus.shape} -> using batch 0, {num_time} time steps")
    else:
        stim_all = stimulus[batch_idx:batch_idx+1, :num_time, :].to(device=device, dtype=model.dtype)
        print(f"  One-eye stimulus: {stimulus.shape} -> using batch 0, {num_time} time steps")

    print(f"Running streaming inference on {num_time} frames (from cell_inputs.pt)...")
    frame_times = []
    cached_mappings = None

    if is_two_eye:
        state_right = None
        state_left = None
        filter_state_right = None
        filter_state_left = None

        with torch.no_grad():
            for t in range(num_time):
                right_frame = right_stim_all[:, t:t+1, :]
                left_frame = left_stim_all[:, t:t+1, :]

                t0 = time.perf_counter()
                _, _, r_res, l_res = model.estimate_ego_motion_from_image(
                    right_frame, left_frame, oob_columns=oob,
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

                if (t + 1) % 200 == 0:
                    print(f"  {t + 1}/{num_time}")

        stimulus_bytes = _tensor_bytes(right_frame) + _tensor_bytes(left_frame)
    else:
        state = None
        filter_state = None

        with torch.no_grad():
            for t in range(num_time):
                stim_frame = stim_all[:, t:t+1, :]

                t0 = time.perf_counter()
                result = model.predict(
                    stim_frame,
                    init_state=state,
                    init_filter_state=filter_state,
                    **(cached_mappings if cached_mappings else {}),
                )
                frame_times.append(time.perf_counter() - t0)

                if cached_mappings is None:
                    cached_mappings = {
                        'input_neurons': result['mappings']['input_mapping'],
                        'output_neurons': result['mappings']['output_mapping'],
                    }

                state = result['model_state']['v']
                filter_state = result['model_state']['filter_state']

                if (t + 1) % 200 == 0:
                    print(f"  {t + 1}/{num_time}")

        stimulus_bytes = _tensor_bytes(stim_frame)

    return {
        'frame_times': frame_times,
        'sample_times': None,
        'num_frames': num_time,
        'stimulus_bytes': stimulus_bytes,
        'video_info': None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark per-frame model inference."
    )
    parser.add_argument("model_folder", type=str, help="Path to model folder or model.pt")
    parser.add_argument("--video-dir", type=str, default=None,
                        help="Directory with drone videos (default: <connectome_folder>/drone_videos)")
    parser.add_argument("--clip-id", type=int, default=32,
                        help="Which clip to use (default: 32)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Cap the number of frames to process")
    parser.add_argument("--max-time", type=int, default=100,
                        help="Limit to first N time points (default: 100)")
    parser.add_argument("--filter-std", type=float, default=2.5,
                        help="Gaussian filter std in degrees for ommatidium sampling")
    args = parser.parse_args()

    rss_checkpoints = {}
    rss_checkpoints["before model load"] = _peak_rss()

    model_path = Path(args.model_folder)
    model_dir = model_path.parent if model_path.is_file() else model_path

    print(f"Loading model from {args.model_folder}...")
    model = nm.NeuralModel.load(args.model_folder)
    device = model.device
    model.eval()
    rss_checkpoints["after model load"] = _peak_rss()

    param_bytes, buffer_bytes = _model_memory(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters:    {n_params:,}  ({_fmt_bytes(param_bytes)})")
    print(f"  Buffers:       {_fmt_bytes(buffer_bytes)}")
    print(f"  Tensors only:  {_fmt_bytes(param_bytes + buffer_bytes)}")

    # --- Load stimulus and run benchmark ---
    video_data = _try_load_video(args, model)

    if video_data is not None:
        results = _run_video_benchmark(video_data, model, device, rss_checkpoints)
    else:
        cell_inputs_path = model_dir / "cell_inputs.pt"
        if not cell_inputs_path.exists():
            print(f"\nError: no drone video found and no cell_inputs.pt in {model_dir}")
            print("Provide --video-dir or ensure cell_inputs.pt exists in the model folder.")
            sys.exit(1)
        print("No raw video found, falling back to cell_inputs.pt...")
        results = _run_cell_inputs_benchmark(cell_inputs_path, model, device, args, rss_checkpoints)

    rss_checkpoints["after inference"] = _peak_rss()

    # --- Report timing ---
    num_frames = results['num_frames']
    frame_times_ms = np.array(results['frame_times']) * 1000

    print(f"\nPer-frame timing (n={num_frames}):")
    if results['sample_times'] is not None:
        sample_times_ms = np.array(results['sample_times']) * 1000
        print(f"  sample_image:")
        print(f"    Mean:   {np.mean(sample_times_ms):.4f} ms")
        print(f"    Median: {np.median(sample_times_ms):.4f} ms")

    print(f"  model inference:")
    print(f"    Mean:   {np.mean(frame_times_ms):.4f} ms")
    print(f"    Median: {np.median(frame_times_ms):.4f} ms")
    print(f"    Std:    {np.std(frame_times_ms):.4f} ms")
    print(f"    Min:    {np.min(frame_times_ms):.4f} ms")
    print(f"    Max:    {np.max(frame_times_ms):.4f} ms")

    if results['sample_times'] is not None:
        total_ms = frame_times_ms + sample_times_ms
        print(f"  total (sample + inference):")
        print(f"    Mean:   {np.mean(total_ms):.4f} ms")
        print(f"    Median: {np.median(total_ms):.4f} ms")

    # --- Memory report ---
    pid = os.getpid()
    peak_rss = _peak_rss()

    print(f"\nMemory breakdown (PID {pid}, device={device}):")
    print(f"  1) Model tensors (params + buffers): {_fmt_bytes(param_bytes + buffer_bytes)}")
    print(f"       Parameters: {_fmt_bytes(param_bytes)}  ({n_params:,} params)")
    print(f"       Buffers:    {_fmt_bytes(buffer_bytes)}")

    video_info = results['video_info']
    if video_info is not None:
        single_frame_pair = 2 * video_info['raw_frame_bytes']
        total_video_bytes = video_info['total_video_bytes']
        ih, iw = video_info['image_height'], video_info['image_width']
        print(f"  2) Single raw frame pair (L+R, grayscale):")
        print(f"       {_fmt_bytes(single_frame_pair)}  (2x {ih}x{iw}, uint8)")

    stimulus_bytes = results['stimulus_bytes']
    if stimulus_bytes is not None:
        print(f"  {'3' if video_info else '2'}) Single stimulus frame (after sampling):")
        print(f"       {_fmt_bytes(stimulus_bytes)}")

    print(f"\n  Peak RSS at checkpoints:")
    for label, rss in rss_checkpoints.items():
        print(f"    {label:.<30s} {_fmt_bytes(rss)}")
    print(f"    {'final':.<30s} {_fmt_bytes(peak_rss)}")

    if video_info is not None:
        estimated_realworld = peak_rss - total_video_bytes + single_frame_pair
        print(f"\n  Estimated real-world memory (single-frame streaming):")
        print(f"    Peak RSS (this run):       {_fmt_bytes(peak_rss)}")
        print(f"    - video buffer:            {_fmt_bytes(total_video_bytes)}")
        print(f"    + single frame (L+R):      {_fmt_bytes(single_frame_pair)}")
        print(f"    = estimated requirement:   {_fmt_bytes(estimated_realworld)}")

    # --- Histogram ---
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
