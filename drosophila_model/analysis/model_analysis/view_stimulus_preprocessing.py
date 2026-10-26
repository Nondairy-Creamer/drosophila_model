"""
Compare stimulus preprocessing pipelines on the central ommatidium trace.

Generates the same rotating natural panorama and shows how log-scale,
zero-centering, and highpass filtering change the intensity time series
that is fed to the model's input columns (L1-L3).

The model (``--model``: a trained run folder or its model.pt) only supplies
the column geometry and dt. The figure goes to
<model_folder>/figures/view_stimulus_preprocessing unless ``--save`` is given.

Usage (from the repo root):
    python -m drosophila_model.analysis.model_analysis.view_stimulus_preprocessing --model models/<run>
    python -m drosophila_model.analysis.model_analysis.view_stimulus_preprocessing --model models/<run> --velocity 60 --direction left
    python -m drosophila_model.analysis.model_analysis.view_stimulus_preprocessing --model models/<run> --highpass-taus 0.1 0.5 1.0
"""
import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from drosophila_model.model_definition import neural_models as nm
from drosophila_model.data_handling.stimulus_generators import get_constant_velocity_pano
from drosophila_model.analysis.model_analysis.analysis_utilities import (
    central_column_index, exponential_highpass,
)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize stimulus preprocessing on the central ommatidium."
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Trained run folder or its model.pt (used for column geometry "
             "and dt).",
    )
    parser.add_argument(
        "--direction", type=str, default="right",
        choices=["left", "right", "up", "down"],
    )
    parser.add_argument(
        "--velocity", type=float, default=30.0,
        help="Rotation speed in deg/s.",
    )
    parser.add_argument(
        "--stim-time", type=float, default=3.0,
        help="Duration of motion (seconds).",
    )
    parser.add_argument(
        "--pre-stim-time", type=float, default=0.5,
        help="Static panorama before motion (seconds).",
    )
    parser.add_argument(
        "--pano-index", type=int, default=0,
        help="Which panorama from all_panoramic.pt.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--highpass-taus", type=float, nargs="+", default=[0.1, 0.5, 2.0],
        help="Highpass time constants to compare.",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Path to save the figure "
             "(default: <model_folder>/figures/view_stimulus_preprocessing/preprocessing.png).",
    )
    args = parser.parse_args()

    # ── Load model (for geometry and dt only) ─────────────────────────
    print(f"Loading model from {args.model} ...")
    model = nm.BaseRnn.load(args.model)
    sample_phi = model.regions['optic_lobe'].column_phi_theta[:, 0]
    sample_theta = model.regions['optic_lobe'].column_phi_theta[:, 1]
    dt = model.dt

    if args.save:
        save_path = Path(args.save)
    else:
        model_path = Path(args.model)
        model_folder = model_path.parent if model_path.is_file() else model_path
        save_path = (model_folder / "figures" / "view_stimulus_preprocessing"
                     / "preprocessing.png")

    center_idx = central_column_index(
        np.asarray(sample_phi), np.asarray(sample_theta)
    )
    print(
        f"Central ommatidium: column {center_idx} "
        f"(phi={np.degrees(sample_phi[center_idx]):.1f} deg, "
        f"theta={np.degrees(sample_theta[center_idx]):.1f} deg)"
    )

    # ── Generate raw panorama (no preprocessing) ──────────────────────
    raw = get_constant_velocity_pano(
        sample_phi, sample_theta,
        args.direction,
        velocity_deg_per_s=args.velocity,
        dt=dt,
        stim_time=args.stim_time,
        pre_stim_time=args.pre_stim_time,
        seed=args.seed,
        pano_indices=np.array([args.pano_index]),
        log_scale=False,
    )  # (1, T, N)

    raw_log = get_constant_velocity_pano(
        sample_phi, sample_theta,
        args.direction,
        velocity_deg_per_s=args.velocity,
        dt=dt,
        stim_time=args.stim_time,
        pre_stim_time=args.pre_stim_time,
        seed=args.seed,
        pano_indices=np.array([args.pano_index]),
        log_scale=True,
    )

    time_axis = np.arange(raw.shape[1]) * dt - args.pre_stim_time

    # ── Build all variants ────────────────────────────────────────────
    variants = []

    variants.append(("raw", raw[0, :, center_idx]))
    variants.append(("raw + zero-center", 2.0 * raw[0, :, center_idx] - 1.0))
    variants.append(("log-scale", raw_log[0, :, center_idx]))
    variants.append(("log-scale + zero-center", 2.0 * raw_log[0, :, center_idx] - 1.0))

    for tau in args.highpass_taus:
        hp = exponential_highpass(raw, dt, tau)
        variants.append((f"HP τ={tau}s", hp[0, :, center_idx]))

    for tau in args.highpass_taus:
        zc = 2.0 * raw - 1.0
        hp = exponential_highpass(zc, dt, tau)
        variants.append((f"zero-center + HP τ={tau}s", hp[0, :, center_idx]))

    for tau in args.highpass_taus:
        hp = exponential_highpass(raw_log, dt, tau)
        variants.append((f"log + HP τ={tau}s", hp[0, :, center_idx]))

    # ── Plot ──────────────────────────────────────────────────────────
    n = len(variants)
    # Constrained layout keeps the suptitle clear of the top axes, which
    # tight_layout does not for a figure this tall.
    fig, axes = plt.subplots(n, 1, figsize=(12, 1.8 * n), sharex=True, squeeze=False,
                             layout="constrained")

    for i, (label, trace) in enumerate(variants):
        ax = axes[i, 0]
        ax.plot(time_axis, trace, linewidth=0.8)
        ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.set_ylabel(label, fontsize=8)
        ax.tick_params(labelsize=7)

    axes[-1, 0].set_xlabel("Time (s)")
    fig.suptitle(
        f"Central ommatidium (col {center_idx}) — {args.direction} @ "
        f"{args.velocity} deg/s, pano {args.pano_index}",
        fontsize=12, fontweight="bold",
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Figure saved to {save_path}")


if __name__ == "__main__":
    main()
