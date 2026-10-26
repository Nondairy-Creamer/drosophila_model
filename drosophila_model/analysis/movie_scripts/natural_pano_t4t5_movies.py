"""
Render T4 and T5 diamond-layout movies (stimulus center + four subtypes) for
constant-angular-velocity natural panoramas — same stimulus pipeline as
``natural_pano_velocity_sweep`` / ``get_constant_velocity_pano``.

The stimulus drives the optic lobe through its configured input neurons (the
region default, L1/L2/L3, unless the model overrides it), as in training.

Example (from the repo root)::

    python -m drosophila_model.analysis.movie_scripts.natural_pano_t4t5_movies \\
        models/<run> --velocity 60 --direction right --pano-index 0 --zero-center
"""
import argparse
from pathlib import Path


import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

from drosophila_model.model_definition import neural_models as nm

from drosophila_model.analysis.model_analysis.analysis_utilities import exponential_highpass
from drosophila_model.analysis.movie_scripts.movie_utilities import make_movie, make_scatter_diamond_sequence_factory
from drosophila_model.data_handling.stimulus_generators import get_constant_velocity_pano

T4_TYPES = ["T4a", "T4b", "T4c", "T4d"]
T5_TYPES = ["T5a", "T5b", "T5c", "T5d"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="T4/T5 movies for fixed-velocity natural panorama rotation."
    )
    parser.add_argument(
        "model_folder",
        type=str,
        help="Path to model folder or model.pt",
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
        help="Map the stimulus from [0,1] to [-1,1] (2x-1).",
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
        help="Output directory (default: <model_folder>/figures/natural_pano_t4t5_movies).",
    )
    parser.add_argument(
        "--highpass-tau",
        type=float,
        default=None,
        dest="highpass_tau",
        help="Time constant (seconds) for causal exponential highpass filter "
             "applied per column before the model. Removes DC offset.",
    )

    args = parser.parse_args()

    model_path = Path(args.model_folder)
    model_folder = model_path.parent if model_path.is_file() else model_path
    model = nm.BaseRnn.load(model_path)

    dt = model.dt

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
        print(f"Applying exponential highpass filter (tau={args.highpass_tau} s)")
        stimulus_np = exponential_highpass(stimulus_np, dt, args.highpass_tau)

    stimulus = torch.tensor(stimulus_np, device=model.device, dtype=model.dtype)

    # input_neurons=None: drive the optic lobe through its configured inputs.
    with torch.no_grad():
        pred = model.predict(
            stimulus,
            input_neurons=None,
            output_neurons=T4_TYPES + T5_TYPES,
        )["outputs"]["optic_lobe"]["firing_rate"]

    # Eye view (x-z plane), as in the other movie scripts.
    column_xyz = model.regions['optic_lobe'].get_column_xyz()
    cell_x, cell_z = -column_xyz[:, 0], column_xyz[:, 2]

    # The diamond factory colours the centre stimulus on a fixed [0, 1] scale;
    # rescale a zero-centred or high-passed stimulus into that range.
    stim_np = stimulus_np
    if args.zero_center or args.highpass_tau is not None:
        lo, hi = np.nanmin(stim_np), np.nanmax(stim_np)
        stim_np = (stim_np - lo) / max(hi - lo, 1e-12)
    cp = args.color_percentile

    out_root = (
        Path(args.out_dir)
        if args.out_dir
        else model_folder / "figures" / "natural_pano_t4t5_movies"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    tag = f"v{args.velocity:g}"
    if args.concat_directions:
        tag += "_cardinal4"
    else:
        tag += f"_{args.direction}"
    tag += f"_pano{args.pano_index}"

    frame_rate = max(1, int(round(1.0 / dt)))

    for group_name, measured_cell_types in [("t4", T4_TYPES), ("t5", T5_TYPES)]:
        seq_data = {}
        for mct in measured_cell_types:
            cell_resp = pred[mct].detach().cpu().numpy()
            cell_resp = cell_resp / max(np.nanpercentile(cell_resp, cp), 1e-12)
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
