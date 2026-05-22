"""
Plot input convolution filter parameters.

Generates:
  - input_conv_filters.png  (spatial kernels on hex grid)
  - input_conv_lru.png      (LRU mode components + full impulse response)
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot input convolution filter parameters.")
    parser.add_argument(
        "model_folder",
        type=str,
        help="Path to model folder or model.pt",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional output file path (default: <model_folder>/figures/plot_input_conv_filters/input_conv_filters.png)",
    )
    parser.add_argument(
        "--impulse-steps",
        type=int,
        default=20,
        help="Number of time steps for impulse response plot (default: 100)",
    )
    args = parser.parse_args()

    model_path = Path(args.model_folder)
    model_pt = model_path / "model.pt" if model_path.is_dir() else model_path
    model = nm.BaseRnn.load(model_pt, device="cpu")
    if not model.regions['optic_lobe'].use_input_conv:
        raise ValueError("Model does not have input convolution enabled.")

    raw_weight = model.regions['optic_lobe'].input_spatial_conv.weight[0, :, 0].detach()
    if model.regions['optic_lobe'].input_spatial_conv.positive_weights:
        raw_weight = raw_weight.exp()
        raw_weight = raw_weight / raw_weight.sum()
    conv_xy = raw_weight.cpu().numpy()

    has_pre_conv = hasattr(model, 'input_pre_conv')
    num_plots = 1 + int(has_pre_conv)
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4), dpi=150)
    if num_plots == 1:
        axes = [axes]
    ax_idx = 0

    if has_pre_conv:
        pre_weight = model.regions['optic_lobe'].input_pre_conv.weight[0, :, 0].detach()
        if model.regions['optic_lobe'].input_pre_conv.positive_weights:
            pre_weight = pre_weight.exp()
            pre_weight = pre_weight / pre_weight.sum()
        pre_xy = pre_weight.cpu().numpy()

        pre_offsets = ml.HexGridConv.hex_grid_pq_offsets(model.regions['optic_lobe'].input_pre_conv_size)
        pre_grid = ml.HexGridConv.pq_to_xy(pre_offsets)
        vmin_pre, vmax_pre = centered_norm(pre_xy)
        scatter_pre = axes[ax_idx].scatter(
            pre_grid[:, 0],
            pre_grid[:, 1],
            c=pre_xy,
            cmap="coolwarm",
            vmin=vmin_pre,
            vmax=vmax_pre,
            s=760,
            linewidths=0.6,
            edgecolors="black",
        )
        axes[ax_idx].set_title("pre-filter (spatial)")
        axes[ax_idx].set_xlabel("hex x")
        axes[ax_idx].set_ylabel("hex y")
        axes[ax_idx].set_aspect("equal", adjustable="box")
        x_min, x_max = float(pre_grid[:, 0].min()), float(pre_grid[:, 0].max())
        y_min, y_max = float(pre_grid[:, 1].min()), float(pre_grid[:, 1].max())
        pad = 0.15 * max(x_max - x_min, y_max - y_min, 1.0)
        axes[ax_idx].set_xlim(x_min - pad, x_max + pad)
        axes[ax_idx].set_ylim(y_min - pad, y_max + pad)
        fig.colorbar(scatter_pre, ax=axes[ax_idx], shrink=0.8, pad=0.02)
        ax_idx += 1

    pq_offsets = ml.HexGridConv.hex_grid_pq_offsets(model.regions['optic_lobe'].conv_size_x)
    spatial_weights = conv_xy
    grid_points = ml.HexGridConv.pq_to_xy(pq_offsets)

    vmin_xy, vmax_xy = centered_norm(spatial_weights)
    scatter = axes[ax_idx].scatter(
        grid_points[:, 0],
        grid_points[:, 1],
        c=spatial_weights,
        cmap="coolwarm",
        vmin=vmin_xy,
        vmax=vmax_xy,
        s=760,
        linewidths=0.6,
        edgecolors="black",
    )
    axes[ax_idx].set_title("local mean (spatial kernel)")
    axes[ax_idx].set_xlabel("hex x")
    axes[ax_idx].set_ylabel("hex y")
    axes[ax_idx].set_aspect("equal", adjustable="box")
    x_min, x_max = float(grid_points[:, 0].min()), float(grid_points[:, 0].max())
    y_min, y_max = float(grid_points[:, 1].min()), float(grid_points[:, 1].max())
    pad = 0.15 * max(x_max - x_min, y_max - y_min, 1.0)
    axes[ax_idx].set_xlim(x_min - pad, x_max + pad)
    axes[ax_idx].set_ylim(y_min - pad, y_max + pad)
    fig.colorbar(scatter, ax=axes[ax_idx], shrink=0.8, pad=0.02)

    fig.tight_layout()

    if args.output_path:
        figures_dir = Path(args.output_path).parent
    else:
        model_root = model_path.parent if model_path.is_file() else model_path
        figures_dir = model_root / "figures" / "plot_input_conv_filters"
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
        modes, feedthrough = model.regions['optic_lobe'].temporal_lru.impulse_response_components(num_t=num_steps)
        modes = modes.cpu().numpy()
        feedthrough_val = feedthrough.item()
        h, _ = compute_lru_impulse_response(model.regions['optic_lobe'].temporal_lru, num_steps=num_steps, dt=dt)

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
