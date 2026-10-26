"""
Plot input convolution filter parameters.

Generates (in ``<model_folder>/figures/plot_input_conv_filters/`` or
``--output-dir``):
  - input_conv_filters.png  (pre-filter and local-mean spatial kernels on hex grid)
  - input_conv_lru.png      (LRU mode components + full impulse response)

Usage:
    python -m drosophila_model.analysis.model_analysis.plot_input_conv_filters models/<model>/
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

import torch
from drosophila_model.model_definition import neural_models as nm
from drosophila_model.model_definition import model_layers as ml

from drosophila_model.analysis.model_analysis.analysis_utilities import centered_norm


def compute_lru_impulse_response(lru, num_steps=20, dt=0.01):
    """Compute the impulse response of a TemporalLRU analytically."""
    with torch.no_grad():
        h = lru.impulse_response(num_t=num_steps).cpu().numpy()
    t = np.arange(num_steps) * dt
    return h, t


def _conv_weights(conv):
    """Effective spatial kernel of a HexGridConv (normalised if positive)."""
    weight = conv.weight[0, :, 0].detach()
    if conv.positive_weights:
        weight = weight.exp()
        weight = weight / weight.sum()
    return weight.cpu().numpy()


def _plot_hex_kernel(fig, ax, weights, conv_size, title):
    """Scatter a hex-grid kernel, colour-coded around zero."""
    grid = ml.HexGridConv.pq_to_xy(ml.HexGridConv.hex_grid_pq_offsets(conv_size))
    vmin, vmax = centered_norm(weights)
    scatter = ax.scatter(
        grid[:, 0],
        grid[:, 1],
        c=weights,
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        s=760,
        linewidths=0.6,
        edgecolors="black",
    )
    ax.set_title(title)
    ax.set_xlabel("hex x")
    ax.set_ylabel("hex y")
    ax.set_aspect("equal", adjustable="box")
    x_min, x_max = float(grid[:, 0].min()), float(grid[:, 0].max())
    y_min, y_max = float(grid[:, 1].min()), float(grid[:, 1].max())
    pad = 0.15 * max(x_max - x_min, y_max - y_min, 1.0)
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot input convolution filter parameters.")
    parser.add_argument(
        "model_folder",
        type=str,
        help="Path to model folder or model.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: <model_folder>/figures/plot_input_conv_filters)",
    )
    parser.add_argument(
        "--impulse-steps",
        type=int,
        default=20,
        help="Number of time steps for impulse response plot (default: 20)",
    )
    args = parser.parse_args()

    model_path = Path(args.model_folder)
    model_folder = model_path.parent if model_path.is_file() else model_path
    model = nm.BaseRnn.load(model_path)
    region = model.regions['optic_lobe']
    if not region.use_input_conv:
        raise ValueError("Model does not have input convolution enabled.")

    # use_input_conv builds both the pre-filter and the local-mean conv.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    _plot_hex_kernel(fig, axes[0], _conv_weights(region.input_pre_conv),
                     region.input_pre_conv_size, "pre-filter (spatial)")
    _plot_hex_kernel(fig, axes[1], _conv_weights(region.input_spatial_conv),
                     region.conv_size_x, "local mean (spatial kernel)")
    fig.tight_layout()

    figures_dir = (Path(args.output_dir) if args.output_dir
                   else model_folder / "figures" / "plot_input_conv_filters")
    figures_dir.mkdir(parents=True, exist_ok=True)

    spatial_path = figures_dir / "input_conv_filters.png"
    fig.savefig(spatial_path)
    plt.close(fig)
    print(f"Saved spatial filters to {spatial_path}")

    # --- LRU figure: mode components (left) + full impulse response (right) ---
    num_steps = args.impulse_steps
    dt = model.dt
    t = np.arange(num_steps) * dt

    with torch.no_grad():
        modes, feedthrough = region.temporal_lru.impulse_response_components(num_t=num_steps)
        modes = modes.cpu().numpy()
        feedthrough_val = feedthrough.item()
        h, _ = compute_lru_impulse_response(region.temporal_lru, num_steps=num_steps, dt=dt)

    fig_lru, (ax_comp, ax_full) = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

    for k in range(modes.shape[0]):
        ax_comp.plot(t, modes[k], label=f"mode {k}")
    ax_comp.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax_comp.set_title("LRU mode components")
    ax_comp.set_xlabel("time (s)")
    ax_comp.set_ylabel("amplitude")
    ax_comp.legend(fontsize="small")

    ax_full.plot(t, h)
    ax_full.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax_full.set_title(f"full impulse response (d={feedthrough_val:.3g})")
    ax_full.set_xlabel("time (s)")
    ax_full.set_ylabel("amplitude")

    fig_lru.tight_layout()
    lru_path = figures_dir / "input_conv_lru.png"
    fig_lru.savefig(lru_path)
    plt.close(fig_lru)
    print(f"Saved LRU impulse response to {lru_path}")


if __name__ == "__main__":
    main()
