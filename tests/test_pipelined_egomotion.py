"""Smoke test: serial vs pipelined ego-motion match (on a real checkpoint).

Loads the 100hz drone checkpoint, builds a CurrentModelFast (sparse_impl
selected by ``_auto_select_sparse_impl`` for the host: 'csr' on macOS /
x86 Linux, 'scatter' on Linux ARM / Raspberry Pi), and runs both paths
on a short slice of cell inputs.

Run from the repo root with either:
    uv run python tests/test_pipelined_egomotion.py
    uv run tests/test_pipelined_egomotion.py
"""
from __future__ import annotations

import os
import sys
import time

# Make `neural_models` importable when invoked as a script (from the
# repo root or from anywhere with --project), without relying on the
# caller setting PYTHONPATH.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from neural_models import (  # noqa: E402
    CurrentModelFast,
    NeuralModel,
    PipelinedEgoMotionEstimator,
)


def main() -> None:
    torch.manual_seed(0)
    # Resolve relative to the repo root so this works from any cwd.
    ckpt = os.path.join(_REPO_ROOT, '100hz_best_drone_clayton', 'model.pt')
    cell_inputs_path = os.path.join(
        _REPO_ROOT, '100hz_best_drone_clayton', 'cell_inputs.pt',
    )

    print(f"loading {ckpt} ...")
    base = NeuralModel.load(ckpt, device='cpu', dtype='float32', freeze_params=True)
    base.eval()

    print("wrapping in CurrentModelFast (sparse_impl='auto') ...")
    fast = CurrentModelFast.from_trained(base, sparse_impl='auto')
    print(f"  picked sparse_impl = {fast.sparse_impl!r}")

    # Confirm 'coo' and 'scatter' match the chosen backend on a single forward.
    print("\n--- sparse_impl backend cross-check (single recurrent step) ...")
    with torch.no_grad():
        v0 = fast.steady_state.unsqueeze(0).expand(2, -1).clone()
        u0 = torch.zeros(2, fast.num_neurons, dtype=fast.dtype)
        ref = fast(v0, u0)
        for impl in ('coo', 'scatter'):
            other = CurrentModelFast.from_trained(base, sparse_impl=impl)
            out = other(v0, u0)
            d = (out - ref).abs().max().item()
            print(f"  sparse_impl={impl:7s} max abs diff vs '{fast.sparse_impl}' = {d:.3e}")
            del other

    print(f"loading {cell_inputs_path} ...")
    ci = torch.load(cell_inputs_path, map_location='cpu', weights_only=False)
    cell_inputs = ci['cell_inputs']
    full_S, T, total_cols = cell_inputs.shape
    assert total_cols % 2 == 0, "expected an even total column count (right + left)"
    half = total_cols // 2
    right = cell_inputs[:, :, :half].contiguous()
    left = cell_inputs[:, :, half:].contiguous()

    S = 1
    T_test = 20
    right_in = right[:S, :T_test].clone()
    left_in = left[:S, :T_test].clone()
    print(f"  using slice S={S}, T={T_test}, columns/eye={half}")

    print("\n--- running serial path (estimate_ego_motion_from_image) ...")
    t0 = time.perf_counter()
    rot_serial, trans_serial, _, _ = fast.estimate_ego_motion_from_image(
        right_eye_inputs=right_in, left_eye_inputs=left_in,
    )
    serial_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  rotations    : {tuple(rot_serial.shape)} dtype={rot_serial.dtype}")
    print(f"  translations : {tuple(trans_serial.shape)} dtype={trans_serial.dtype}")
    print(f"  total time   : {serial_ms:.1f} ms ({serial_ms/T_test:.2f} ms/frame)")

    print("\n--- running PipelinedEgoMotionEstimator.run_full ...")
    bd = {}
    with PipelinedEgoMotionEstimator(fast) as est:
        t0 = time.perf_counter()
        rot_pipe, trans_pipe = est.run_full(
            right_in, left_in, breakdown_timings=bd,
        )
        pipe_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  rotations    : {tuple(rot_pipe.shape)}")
    print(f"  translations : {tuple(trans_pipe.shape)}")
    print(f"  total time   : {pipe_ms:.1f} ms ({pipe_ms/T_test:.2f} ms/frame)")
    print(f"  speedup      : {serial_ms / pipe_ms:.2f}x")

    print("\n--- per-stage breakdown (cumulative ms over the sequence) ---")
    for k in sorted(bd):
        print(f"  {k:42s} {bd[k]:7.2f} ms")

    print("\n--- numerical comparison ---")
    rot_diff = (rot_serial - rot_pipe).abs()
    trans_diff = (trans_serial - trans_pipe).abs()
    print(f"  rot  max abs diff = {rot_diff.max().item():.3e}  "
          f"mean = {rot_diff.mean().item():.3e}")
    print(f"  trans max abs diff = {trans_diff.max().item():.3e}  "
          f"mean = {trans_diff.mean().item():.3e}")

    rtol = 1e-4
    atol = 1e-4
    rot_close = torch.allclose(rot_serial, rot_pipe, rtol=rtol, atol=atol)
    trans_close = torch.allclose(trans_serial, trans_pipe, rtol=rtol, atol=atol)
    print(f"  allclose(rtol={rtol}, atol={atol}): "
          f"rotation={rot_close}, translation={trans_close}")

    if rot_close and trans_close:
        print("\nOK: pipelined and serial paths agree.")
    else:
        print("\nWARN: pipelined and serial paths disagree at the requested tolerance.")


if __name__ == '__main__':
    main()
