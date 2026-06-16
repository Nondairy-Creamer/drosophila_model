"""
Compare stimulus preprocessing pipelines on the central ommatidium trace.

Generates the same rotating natural panorama and shows how log-scale,
zero-centering, and highpass filtering change the intensity time series
that reaches the photoreceptors.

Usage:
    python view_stimulus_preprocessing.py
    python view_stimulus_preprocessing.py --velocity 60 --direction left
    python view_stimulus_preprocessing.py --highpass-taus 0.1 0.5 1.0
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from drosophila_model.model_definition import neural_models as nm
from drosophila_model.data_handling.stimulus_generators import get_constant_velocity_pano


DEFAULT_MODEL_PATH = "models/hand_tuned/model.pt"


def _central_column_index(sample_phi, sample_theta):
    """Column closest to forward-viewing (phi=0, theta=pi/2)."""
    ref = np.array([0.0, 1.0, 0.0])
    xyz = np.stack([
        np.sin(sample_theta) * np.cos(sample_phi),
        np.sin(sample_theta) * np.sin(sample_phi),
        np.cos(sample_theta),
    ], axis=1)
    return int(np.argmax(xyz @ ref))


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


def main():
    parser = argparse.ArgumentParser(
        description="Visualize stimulus preprocessing on the central ommatidium."
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL_PATH,
        help="Path to saved model .pt file (used for column geometry and dt).",
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
        help="Save figure to this path instead of showing interactively "
             "(default: <model_folder>/figures/view_stimulus_preprocessing/preprocessing.png).",
    )
    args = parser.parse_args()

    # ── Load model (for geometry and dt only) ─────────────────────────
    print(f"Loading model from {args.model} …")
    model = nm.BaseRnn.load(args.model)
    sample_phi = model.regions['optic_lobe'].column_phi_theta[:, 0]
    sample_theta = model.regions['optic_lobe'].column_phi_theta[:, 1]
    dt = model.dt

    if args.save is None:
        model_path = Path(args.model)
        model_folder = model_path.parent if model_path.is_file() else model_path
        args.save = str(model_folder / "figures" / "view_stimulus_preprocessing"
                        / "preprocessing.png")

    center_idx = _central_column_index(
        np.asarray(sample_phi), np.asarray(sample_theta)
    )
    print(
        f"Central ommatidium: column {center_idx} "
        f"(phi={np.degrees(sample_phi[center_idx]):.1f}°, "
        f"theta={np.degrees(sample_theta[center_idx]):.1f}°)"
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
        hp = _exponential_highpass(raw, dt, tau)
        variants.append((f"HP τ={tau}s", hp[0, :, center_idx]))

    for tau in args.highpass_taus:
        zc = 2.0 * raw - 1.0
        hp = _exponential_highpass(zc, dt, tau)
        variants.append((f"zero-center + HP τ={tau}s", hp[0, :, center_idx]))

    for tau in args.highpass_taus:
        hp = _exponential_highpass(raw_log, dt, tau)
        variants.append((f"log + HP τ={tau}s", hp[0, :, center_idx]))

    # ── Plot ──────────────────────────────────────────────────────────
    n = len(variants)
    fig, axes = plt.subplots(n, 1, figsize=(12, 1.8 * n), sharex=True, squeeze=False)

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
    plt.tight_layout()

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
