"""
Render T4 and T5 diamond-layout movies (stimulus center + four subtypes) for
constant-angular-velocity natural panoramas — same stimulus pipeline as
``natural_pano_velocity_sweep`` / ``get_constant_velocity_pano``.

Example::

    uv run python analysis/natural_pano_t4t5_movies.py models/hand_tuned \\
        --velocity 60 --direction right --pano-index 0 --zero_center
"""
import argparse
import shutil
import sys
from pathlib import Path


import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

from drosophila_model.model_definition import neural_models as nm

from drosophila_model.analysis.model_analysis.analysis_utilities import interp_nan_column
from drosophila_model.analysis.movie_scripts.movie_utilities import make_movie, make_scatter_diamond_sequence_factory
from drosophila_model.utilities.stimulus_utilities import get_constant_velocity_pano


def _exponential_highpass(block, dt, tau):
    """Subtract a causal exponential lowpass from each column to remove DC.

    First-order IIR lowpass:  y[t] = α·y[t-1] + (1-α)·x[t],  α = exp(-dt/τ)
    Returns:  x[t] - y[t]
    """
    alpha = np.exp(-dt / tau)
    out = np.empty_like(block)
    lp = block[:, 0, :].copy()
    out[:, 0, :] = 0.0
    for t in range(1, block.shape[1]):
        lp = alpha * lp + (1.0 - alpha) * block[:, t, :]
        out[:, t, :] = block[:, t, :] - lp
    return out


def _aligned_column_geometry(model: nm.BaseRnn, n_cols: int):
    """``column_phi_theta`` rows vs ``num_columns`` can differ by one; align for scatter/interp."""
    pt = np.asarray(model.regions['optic_lobe'].column_phi_theta)
    if pt.shape[0] > n_cols:
        pt = pt[:n_cols]
    elif pt.shape[0] < n_cols:
        pt = np.pad(pt, ((0, n_cols - pt.shape[0]), (0, 0)), mode="edge")
    phi, theta = pt[:, 0], pt[:, 1]
    r = 1.0
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    # Match ``make_T4T5_movie``: cell_x = -xyz[:,0], cell_z = xyz[:,1]
    cell_x = -x
    cell_z = y
    return cell_x, cell_z, phi, theta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="T4/T5 movies for fixed-velocity natural panorama rotation."
    )
    parser.add_argument(
        "model_folder",
        type=str,
        help="Path to model folder (model.pt, config.toml)",
    )
    parser.add_argument(
        "--velocity",
        type=float,
        default=60.0,
        help="Rotation speed magnitude in deg/s (default: 60).",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="right",
        choices=("left", "right", "up", "down"),
        help="Rotation axis / cardinal direction (default: right).",
    )
    parser.add_argument(
        "--concat-directions",
        action="store_true",
        help="Append movies for right, left, up, down at the same velocity in one clip.",
    )
    parser.add_argument(
        "--pano-index",
        type=int,
        default=0,
        help="Which panorama index from all_panoramic.pt (default: 0).",
    )
    parser.add_argument(
        "--pre-stim-time",
        type=float,
        default=0.5,
        dest="pre_stim_time",
        help="Static panorama before motion (seconds).",
    )
    parser.add_argument(
        "--stim-time",
        type=float,
        default=1.0,
        dest="stim_time",
        help="Motion duration (seconds).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for initial panorama orientation.",
    )
    parser.add_argument(
        "--zero-center",
        "--zero_center",
        action="store_true",
        dest="zero_center",
        help="Map photoreceptor input from [0,1] to [-1,1] (2x-1).",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        dest="log_scale",
        help="Pass log_scale=True into get_constant_velocity_pano.",
    )
    parser.add_argument(
        "--color-percentile",
        type=float,
        default=99.0,
        dest="color_percentile",
        help="Normalize firing rates by this percentile (default: 99).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: model_folder/movie/natural_pano_t4t5).",
    )
    parser.add_argument(
        "--highpass-tau",
        type=float,
        default=None,
        dest="highpass_tau",
        help="Time constant (seconds) for causal exponential highpass filter "
             "applied per column before the model. Removes DC offset.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not delete the output directory before writing.",
    )

    args = parser.parse_args()

    model_folder = Path(args.model_folder)
    model = nm.BaseRnn.load(model_folder / "model.pt")

    dt = model.dt
    print(f"dt: {dt}")

    if args.concat_directions:
        directions = ["right", "left", "up", "down"]
    else:
        directions = [args.direction]

    pano_indices = np.array([args.pano_index], dtype=int)

    blocks = []
    for di, direction in enumerate(directions):
        block = get_constant_velocity_pano(
            model.regions['optic_lobe'].column_phi_theta[:, 0],
            model.regions['optic_lobe'].column_phi_theta[:, 1],
            direction,
            velocity_deg_per_s=args.velocity,
            dt=dt,
            stim_time=args.stim_time,
            pre_stim_time=args.pre_stim_time,
            seed=args.seed + di,
            pano_indices=pano_indices,
            log_scale=args.log_scale,
        )
        blocks.append(block)

    stimulus_np = np.concatenate(blocks, axis=0)
    if args.zero_center:
        stimulus_np = 2.0 * stimulus_np - 1.0
    if args.highpass_tau is not None:
        print(f"Applying exponential highpass filter (τ={args.highpass_tau} s)")
        stimulus_np = _exponential_highpass(stimulus_np, dt, args.highpass_tau)

    stimulus = torch.tensor(stimulus_np, device=model.device, dtype=model.dtype)

    input_neurons = ["R1-6"]
    output_neurons = [
        "L1",
        "Mi1",
        "Mi9",
        "C3",
        "L2",
        "Tm1",
        "Tm2",
        "Tm4",
        "Tm9",
        "T4a",
        "T4b",
        "T4c",
        "T4d",
        "T5a",
        "T5b",
        "T5c",
        "T5d",
    ]

    with torch.no_grad():
        pred = model.predict(
            stimulus,
            input_neurons=input_neurons,
            output_neurons=output_neurons,
        )["outputs"]["firing_rate"]

    stim_np = stimulus.detach().cpu().numpy()
    n_cols = model.regions['optic_lobe'].num_columns
    cell_x, cell_z, cell_phi, cell_theta = _aligned_column_geometry(model, n_cols)
    if stim_np.shape[2] < n_cols:
        pad_width = n_cols - stim_np.shape[2]
        stim_np = np.pad(stim_np, ((0, 0), (0, 0), (0, pad_width)), mode="edge")
    cp = args.color_percentile

    out_root = (
        Path(args.out_dir)
        if args.out_dir
        else model_folder / "movie" / "natural_pano_t4t5"
    )
    if not args.keep_existing:
        shutil.rmtree(out_root, ignore_errors=True)
    out_root.mkdir(parents=True, exist_ok=True)

    tag = f"v{args.velocity:g}"
    if args.concat_directions:
        tag += "_cardinal4"
    else:
        tag += f"_{args.direction}"
    tag += f"_pano{args.pano_index}"

    frame_rate = max(1, int(round(1.0 / dt)))

    for group_name, measured_cell_types in [
        ("t4", ["T4a", "T4b", "T4c", "T4d"]),
        ("t5", ["T5a", "T5b", "T5c", "T5d"]),
    ]:
        seq_data = {}
        for mct in measured_cell_types:
            cell_resp = pred[mct].detach().cpu().numpy()
            cell_resp = cell_resp / max(np.nanpercentile(cell_resp, cp), 1e-12)
            #cell_resp = interp_nan_column(cell_resp, cell_phi, cell_theta)
            seq_data[mct] = cell_resp

        seq_data["stimulus"] = stim_np

        init_fn = make_scatter_diamond_sequence_factory(
            seq_data, cell_x, cell_z, scale=1.8, size_scale=0.6
        )
        make_movie(
            init_fn,
            ["clip"],
            out_root,
            frame_rate,
            save_name=f"{group_name}_{tag}",
        )

    print(f"Wrote movies under {out_root}:")
    for suffix in ("t4", "t5"):
        p = out_root / f"{suffix}_{tag}_clip.mp4"
        print(f"  {p}")


if __name__ == "__main__":
    main()
