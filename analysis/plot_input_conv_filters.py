"""
Plot input convolution filter parameters.

Generates:
  - input_conv_xy (spatial kernel scatter on hex grid)
  - input_conv_t (temporal kernel line plot)
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

import neural_models as nm


def _resolve_model_folder(model_path: Path) -> Path:
    if model_path.is_dir():
        return model_path
    return model_path.parent


def _centered_norm(data: np.ndarray) -> tuple[float, float]:
    max_abs = float(np.max(np.abs(data)))
    if max_abs == 0:
        max_abs = 1.0
    return -max_abs, max_abs


def _hex_grid_points(axial_coords: np.ndarray) -> np.ndarray:
    """Convert axial (q, r) hex coords to flat-top xy points."""
    q = axial_coords[:, 0].astype(np.float64)
    r = axial_coords[:, 1].astype(np.float64)
    grid_x = 1.5 * q
    grid_y = np.sqrt(3.0) * (r + 0.5 * q)
    return np.stack([grid_x, grid_y], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot input convolution filter parameters.")
    parser.add_argument(
        "model_path",
        type=str,
        nargs="?",
        default="models/best_model",
        help="Path to model folder or model.pt (default: models/best_model)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional output file path (default: <model_folder>/input_conv_filters.png)",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    model_pt = model_path / "model.pt" if model_path.is_dir() else model_path
    model = nm.NeuralModel.load(model_pt, device="cpu")
    if not model.use_input_conv:
        raise ValueError("Model does not have input convolution enabled.")

    conv_xy = model.input_conv_xy.detach().cpu().numpy()
    conv_t = model.input_conv_t.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

    axial_coords = model.input_conv_axial_coords.detach().cpu().numpy()
    radius = model.input_conv_size_x // 2
    q_idx = axial_coords[:, 0] + radius
    r_idx = axial_coords[:, 1] + radius
    spatial_weights = conv_xy[r_idx, q_idx]
    grid_points = _hex_grid_points(axial_coords)

    vmin_xy, vmax_xy = _centered_norm(spatial_weights)
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

    axes[1].plot(conv_t, marker="o")
    axes[1].set_title("input_conv_t (temporal kernel)")
    axes[1].set_xlabel("t index")
    axes[1].set_ylabel("weight")
    axes[1].axhline(0.0, color="gray", linewidth=0.8, linestyle="--")

    fig.tight_layout()

    output_path = Path(args.output_path) if args.output_path else _resolve_model_folder(model_path) / "input_conv_filters.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved input convolution filters to {output_path}")


if __name__ == "__main__":
    main()
