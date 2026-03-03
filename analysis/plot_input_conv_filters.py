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


def compute_lru_impulse_response(lru, num_steps=100, dt=0.01):
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
        default=100,
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
    conv_xy = raw_weight.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

    axial_coords = nm.NeuralModelConv.hex_grid_axial_coords(model.input_conv_size_x)
    spatial_weights = conv_xy
    grid_points = hex_grid_points(axial_coords)

    vmin_xy, vmax_xy = centered_norm(spatial_weights)
    scatter = axes[0].scatter(
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
    axes[0].set_title("input_conv_xy (spatial kernel)")
    axes[0].set_xlabel("hex x")
    axes[0].set_ylabel("hex y")
    axes[0].set_aspect("equal", adjustable="box")
    x_min, x_max = float(grid_points[:, 0].min()), float(grid_points[:, 0].max())
    y_min, y_max = float(grid_points[:, 1].min()), float(grid_points[:, 1].max())
    pad = 0.15 * max(x_max - x_min, y_max - y_min, 1.0)
    axes[0].set_xlim(x_min - pad, x_max + pad)
    axes[0].set_ylim(y_min - pad, y_max + pad)
    fig.colorbar(scatter, ax=axes[0], shrink=0.8, pad=0.02)

    h, t = compute_lru_impulse_response(model.temporal_lru, num_steps=args.impulse_steps, dt=model.dt)
    axes[1].plot(t, h)
    axes[1].set_title("temporal LRU impulse response")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("amplitude")
    axes[1].axhline(0.0, color="gray", linewidth=0.8, linestyle="--")

    fig.tight_layout()

    output_path = Path(args.output_path) if args.output_path else resolve_model_folder(model_path) / "figures" / "input_conv_filters.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved input convolution filters to {output_path}")


if __name__ == "__main__":
    main()
