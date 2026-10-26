"""Benchmark model inference timing and peak memory usage.

Times the two-eye ego-motion pipeline on one stimulus of the model's own
training data (``cell_inputs.pt``, the training split).

Usage:
    python -m drosophila_model.analysis.model_analysis.model_timing models/<model>/
    python -m drosophila_model.analysis.model_analysis.model_timing models/<model>/ --device cpu
    python -m drosophila_model.analysis.model_analysis.model_timing models/<model>/ --stimulus-index 3
"""
import argparse
import platform
import sys
import time
from pathlib import Path

try:
    import resource
except ImportError:  # Windows
    resource = None

import numpy as np
import torch

from drosophila_model.config import load_run_config
from drosophila_model.model_definition import neural_models as nm
from drosophila_model.analysis.model_analysis.analysis_utilities import run_eye_offset


def _peak_rss_bytes():
    """Peak resident memory of the current process in bytes, or None if unknown."""
    if resource is not None:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":
            return rss
        return rss * 1024
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class _ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                    "PagefileUsage", "PeakPagefileUsage")
            ]

        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessMemoryCounters), wintypes.DWORD]
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if get_info(handle, ctypes.byref(counters), counters.cb):
            return counters.PeakWorkingSetSize
    return None


def _fmt_bytes(n_bytes):
    if n_bytes is None:
        return "n/a"
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
                        help="Path to model folder or model.pt")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run on (default: the device in config.toml)")
    parser.add_argument("--stimulus-index", type=int, default=5,
                        help="Which stimulus to use (default: 5)")
    args = parser.parse_args()

    model_path = Path(args.model_folder)
    model_folder = model_path.parent if model_path.is_file() else model_path
    eye_offset = run_eye_offset(load_run_config(model_folder))

    mem_before = _peak_rss_bytes()

    # --- Load model ---
    print(f"Loading model from {model_path}...")
    model = nm.BaseRnn.load(model_path, device=args.device)
    device = model.device
    dtype = model.dtype
    model.eval()

    mem_after_model = _peak_rss_bytes()

    # --- Load stimulus ---
    cell_inputs_path = model_folder / "cell_inputs.pt"
    print(f"Loading stimulus from {cell_inputs_path}...")
    cell_inputs_data = torch.load(cell_inputs_path, map_location=device, weights_only=True)
    cell_inputs = cell_inputs_data["cell_inputs"].to(device=device, dtype=dtype)

    idx = args.stimulus_index
    n_cols = model.regions['optic_lobe'].num_columns
    if cell_inputs.shape[2] != 2 * n_cols:
        sys.exit(f"cell_inputs.pt has {cell_inputs.shape[2]} columns, expected "
                 f"{2 * n_cols} (two eyes); model_timing needs a model trained on two-eye data.")
    if not 0 <= idx < cell_inputs.shape[0]:
        sys.exit(f"--stimulus-index {idx} is out of range (0..{cell_inputs.shape[0] - 1}).")
    right_stim = cell_inputs[idx : idx + 1, :, :n_cols]
    left_stim = cell_inputs[idx : idx + 1, :, n_cols:]
    n_time = right_stim.shape[1]
    print(f"  Right eye: {right_stim.shape}, Left eye: {left_stim.shape}  (index {idx}, {n_time} steps)")

    mem_after_stimulus = _peak_rss_bytes()

    # --- Warmup (CUDA kernel compilation, cuDNN autotuning, etc.) ---
    # Also build the cached column mappings here so the per-frame loop below is
    # steady-state (the first call constructs mappings, a one-time cost we don't
    # want polluting the per-step timing).
    print("\nWarmup run...")
    with torch.no_grad():
        _, _, warmup_r_res, _ = model.regions['optic_lobe'].estimate_ego_motion_from_image(
            right_stim[:, :1, :], left_stim[:, :1, :], camera_offset=eye_offset,
        )
        _sync(device)
    warmup_cached_mappings = warmup_r_res["mappings"]

    # --- Time full forward pass ---
    print(f"\nRunning full forward pass on device={device}...")
    with torch.no_grad():
        _sync(device)
        t0 = time.perf_counter()
        model.regions['optic_lobe'].estimate_ego_motion_from_image(
            right_stim, left_stim, camera_offset=eye_offset)
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
        cached_mappings = warmup_cached_mappings
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
                camera_offset=eye_offset,
            )
            _sync(device)
            frame_times.append(time.perf_counter() - t0)

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

    print("\nPeak memory (RSS):")
    print(f"  Before model load:  {_fmt_bytes(mem_before)}")
    print(f"  After model load:   {_fmt_bytes(mem_after_model)}")
    print(f"  After stimulus load:{_fmt_bytes(mem_after_stimulus)}")
    print(f"  After inference:    {_fmt_bytes(mem_after_inference)}")


if __name__ == "__main__":
    main()
