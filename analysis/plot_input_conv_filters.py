"""
Plot input convolution filter parameters.

Generates:
  - input_conv_xy (spatial kernel scatter on hex grid)
  - input_conv_t (temporal LRU impulse response)
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

import torch
import neural_models as nm

from analysis_utilities import resolve_model_folder, centered_norm, hex_grid_points


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
        help="Optional output file path (default: <model_folder>/figures/input_conv_filters.png)",
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
    model = nm.NeuralModel.load(model_pt, device="cpu")
    if not model.use_input_conv:
        raise ValueError("Model does not have input convolution enabled.")

    raw_weight = model.input_spatial_conv.weight[0, :, 0].detach()
    if model.input_spatial_conv.positive_weights:
        raw_weight = raw_weight.exp()
        raw_weight = raw_weight / raw_weight.sum()
    conv_xy = raw_weight.cpu().numpy()

    has_pre_conv = hasattr(model, 'input_pre_conv')
    has_ego_conv = hasattr(model, 'ego_motion_conv')
    num_plots = 2 + int(has_pre_conv) + int(has_ego_conv)
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4), dpi=150)
    ax_idx = 0

    if has_pre_conv:
        pre_weight = model.input_pre_conv.weight[0, :, 0].detach()
        if model.input_pre_conv.positive_weights:
            pre_weight = pre_weight.exp()
            pre_weight = pre_weight / pre_weight.sum()
        pre_xy = pre_weight.cpu().numpy()

        pre_axial = nm.NeuralModelConv.hex_grid_axial_coords(model.input_pre_conv_size)
        pre_grid = hex_grid_points(pre_axial)
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

    axial_coords = nm.NeuralModelConv.hex_grid_axial_coords(model.input_conv_size_x)
    spatial_weights = conv_xy
    grid_points = hex_grid_points(axial_coords)

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
    ax_idx += 1

    h, t = compute_lru_impulse_response(model.temporal_lru, num_steps=args.impulse_steps, dt=model.dt)
    axes[ax_idx].plot(t, h)
    axes[ax_idx].set_title("temporal LRU impulse response")
    axes[ax_idx].set_xlabel("time (s)")
    axes[ax_idx].set_ylabel("amplitude")
    axes[ax_idx].axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax_idx += 1

    if has_ego_conv:
        ego_raw = model.ego_motion_conv.weight[0, :, 0].detach()
        if model.ego_motion_conv.positive_weights:
            ego_raw = ego_raw.exp()
            ego_raw = ego_raw / ego_raw.sum()
        ego_xy = ego_raw.cpu().numpy()

        ego_conv_size = model.ego_motion_conv.weight.shape[1]
        for sz in range(1, 20, 2):
            if len(nm.NeuralModelConv.hex_grid_axial_coords(sz)) == ego_conv_size:
                ego_grid_size = sz
                break
        ego_axial = nm.NeuralModelConv.hex_grid_axial_coords(ego_grid_size)
        ego_grid = hex_grid_points(ego_axial)
        vmin_ego, vmax_ego = centered_norm(ego_xy)
        scatter_ego = axes[ax_idx].scatter(
            ego_grid[:, 0],
            ego_grid[:, 1],
            c=ego_xy,
            cmap="coolwarm",
            vmin=vmin_ego,
            vmax=vmax_ego,
            s=760,
            linewidths=0.6,
            edgecolors="black",
        )
        axes[ax_idx].set_title("ego-motion translation (spatial)")
        axes[ax_idx].set_xlabel("hex x")
        axes[ax_idx].set_ylabel("hex y")
        axes[ax_idx].set_aspect("equal", adjustable="box")
        x_min, x_max = float(ego_grid[:, 0].min()), float(ego_grid[:, 0].max())
        y_min, y_max = float(ego_grid[:, 1].min()), float(ego_grid[:, 1].max())
        pad = 0.15 * max(x_max - x_min, y_max - y_min, 1.0)
        axes[ax_idx].set_xlim(x_min - pad, x_max + pad)
        axes[ax_idx].set_ylim(y_min - pad, y_max + pad)
        fig.colorbar(scatter_ego, ax=axes[ax_idx], shrink=0.8, pad=0.02)

    fig.tight_layout()

    output_path = Path(args.output_path) if args.output_path else resolve_model_folder(model_path) / "figures" / "input_conv_filters.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved input convolution filters to {output_path}")


if __name__ == "__main__":
    main()
