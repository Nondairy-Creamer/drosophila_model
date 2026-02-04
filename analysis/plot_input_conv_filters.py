"""
Plot input convolution filter parameters as heatmaps.

Generates:
  - input_conv_xy (spatial kernel heatmap)
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

    vmin_xy, vmax_xy = _centered_norm(conv_xy)
    im_xy = axes[0].imshow(
        conv_xy,
        cmap="coolwarm",
        vmin=vmin_xy,
        vmax=vmax_xy,
        origin="lower",
        aspect="auto",
    )
    axes[0].set_title("input_conv_xy (spatial kernel)")
    axes[0].set_xlabel("x index")
    axes[0].set_ylabel("y index")
    fig.colorbar(im_xy, ax=axes[0], shrink=0.8, pad=0.02)

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
