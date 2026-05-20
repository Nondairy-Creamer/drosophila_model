"""Benchmark model inference timing and peak memory usage.

Usage:
    python analysis/model_timing.py --model-folder models/best_drone/
    python analysis/model_timing.py --model-folder models/best_drone/ --device cpu
    python analysis/model_timing.py --model-folder models/best_drone/ --stimulus-index 3
"""
import argparse
import platform
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

import neural_models as nm


def _peak_rss_bytes():
    """Peak RSS of the current process in bytes."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return rss
    return rss * 1024


def _fmt_bytes(n_bytes):
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n_bytes) < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark model inference.")
    parser.add_argument("model_folder", type=str,
                        help="Path to model folder")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run on (default: model's own device)")
    parser.add_argument("--stimulus-index", type=int, default=5,
                        help="Which stimulus to use (default: 5)")
    args = parser.parse_args()

    model_path = Path(args.model_folder)
    model_folder = model_path.parent if model_path.is_file() else model_path

    mem_before = _peak_rss_bytes()

    # --- Load model ---
    print(f"Loading model from {model_path}...")
    model = nm.BaseRnn.load(model_path)
    if args.device is not None:
        model = model.to(torch.device(args.device))
    device = model.device
    dtype = model.dtype
    model.eval()

    mem_after_model = _peak_rss_bytes()

    # --- Load stimulus ---
    cell_inputs_path = model_folder / "cell_inputs.pt"
    print(f"Loading stimulus from {cell_inputs_path}...")
    cell_inputs_data = torch.load(cell_inputs_path, map_location=device, weights_only=True)
    if isinstance(cell_inputs_data, dict):
        cell_inputs = cell_inputs_data["cell_inputs"].to(device=device, dtype=dtype)
    else:
        cell_inputs = cell_inputs_data.to(device=device, dtype=dtype)

    idx = args.stimulus_index
    n_cols = model.num_columns
    right_stim = cell_inputs[idx : idx + 1, :, :n_cols]
    left_stim = cell_inputs[idx : idx + 1, :, n_cols:]
    n_time = right_stim.shape[1]
    print(f"  Right eye: {right_stim.shape}, Left eye: {left_stim.shape}  (index {idx}, {n_time} steps)")

    mem_after_stimulus = _peak_rss_bytes()

    # --- Warmup (CUDA kernel compilation, cuDNN autotuning, etc.) ---
    print("\nWarmup run...")
    with torch.no_grad():
        model.regions['optic_lobe'].estimate_ego_motion_from_image(
            right_stim[:, :1, :], left_stim[:, :1, :],
        )
        _sync(device)

    # --- Time full forward pass ---
    print(f"\nRunning full forward pass on device={device}...")
    with torch.no_grad():
        _sync(device)
        t0 = time.perf_counter()
        model.regions['optic_lobe'].estimate_ego_motion_from_image(right_stim, left_stim)
        _sync(device)
        full_pass_time = time.perf_counter() - t0
    print(f"  Full pass: {full_pass_time * 1000:.2f} ms ({full_pass_time / n_time * 1000:.4f} ms/frame)")

    # --- Time per-frame streaming loop ---
    print(f"Running per-frame loop ({n_time} steps)...")
    frame_times = []
    with torch.no_grad():
        state_right = None
        state_left = None
        filter_state_right = None
        filter_state_left = None
        cached_mappings = None
        for t in range(n_time):
            right_frame = right_stim[:, t : t + 1, :]
            left_frame = left_stim[:, t : t + 1, :]

            _sync(device)
            t0 = time.perf_counter()
            _, _, r_res, l_res = model.regions['optic_lobe'].estimate_ego_motion_from_image(
                right_frame, left_frame,
                initial_state_right=state_right,
                initial_state_left=state_left,
                initial_filter_state_right=filter_state_right,
                initial_filter_state_left=filter_state_left,
                cached_mappings=cached_mappings,
            )
            _sync(device)
            frame_times.append(time.perf_counter() - t0)

            if cached_mappings is None:
                cached_mappings = r_res["mappings"]
            state_right = r_res["model_state"]["v"]
            state_left = l_res["model_state"]["v"]
            filter_state_right = r_res["model_state"]["filter_state"]
            filter_state_left = l_res["model_state"]["filter_state"]

    frame_times_ms = np.array(frame_times) * 1000
    mem_after_inference = _peak_rss_bytes()

    # --- Report ---
    print(f"\nPer-frame timing ({n_time} frames, device={device}):")
    print(f"  Mean: {np.mean(frame_times_ms):.4f} ms")
    print(f"  Std:  {np.std(frame_times_ms):.4f} ms")

    print(f"\nPeak memory (RSS):")
    print(f"  Before model load:  {_fmt_bytes(mem_before)}")
    print(f"  After model load:   {_fmt_bytes(mem_after_model)}")
    print(f"  After stimulus load:{_fmt_bytes(mem_after_stimulus)}")
    print(f"  After inference:    {_fmt_bytes(mem_after_inference)}")


if __name__ == "__main__":
    main()
