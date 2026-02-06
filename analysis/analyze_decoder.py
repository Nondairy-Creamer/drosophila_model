"""
Analyze decoder convolution weights.

Creates:
  1. Bar plot: average |weight| per neuron type across the 8 channels and spatial positions
  2. Hex scatter heatmap: average |weight| at each spatial position in the conv kernel
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

from analysis_utilities import resolve_model_paths, hex_grid_points


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze decoder convolution weights."
    )
    parser.add_argument(
        "model_path",
        type=str,
        nargs="?",
        default="models/best_sine",
        help="Path to model folder or model.pt (default: models/best_sine)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load the model on (default: cpu)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for figures (default: <model_folder>/figures)",
    )
    args = parser.parse_args()

    model_pt, model_folder = resolve_model_paths(Path(args.model_path))
    model = nm.NeuralModel.load(model_pt, device=args.device)

    if not model.use_decoder:
        raise ValueError("Model does not have a decoder.")

    output_dir = Path(args.output_dir) if args.output_dir else model_folder / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract decoder_conv_1 weights: (channels_1, num_nearest_columns, num_decoder_neuron_types)
    conv_1 = model.decoder_conv_1.detach().cpu().numpy()
    neuron_names = model.neurons_to_decode

    # ------------------------------------------------------------------
    # Bar plot: average |weight| per neuron type across 8 channels
    # ------------------------------------------------------------------
    # conv_1 shape: (channels_1=8, num_nearest_columns, num_decoder_neuron_types)
    # For each neuron type, average abs weight over channels and spatial positions
    avg_per_neuron = np.mean(np.abs(conv_1), axis=(0, 1))  # (num_decoder_neuron_types,)

    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    x_pos = np.arange(len(neuron_names))
    ax.bar(x_pos, avg_per_neuron, color="tab:blue", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(neuron_names, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Neuron type")
    ax.set_ylabel("Mean |weight|")
    ax.set_title("Decoder conv_1: average |weight| per neuron (across 8 channels)")
    fig.tight_layout()

    bar_path = output_dir / "decoder_neuron_weights.png"
    fig.savefig(bar_path)
    plt.close(fig)
    print(f"Saved bar plot to {bar_path}")

    # ------------------------------------------------------------------
    # Hex scatter heatmap: average |weight| per spatial position
    # ------------------------------------------------------------------
    # Average across channels and neuron types -> (num_nearest_columns,)
    avg_per_position = np.mean(np.abs(conv_1), axis=(0, 2))

    axial_coords = nm.NeuralModel._hex_grid_axial_coords(model.conv_size)
    grid_points = hex_grid_points(axial_coords)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    sc = ax.scatter(
        grid_points[:, 0],
        grid_points[:, 1],
        c=avg_per_position,
        cmap="viridis",
        s=760,
        linewidths=0.6,
        edgecolors="black",
    )
    fig.colorbar(sc, ax=ax, label="Mean |weight|")
    ax.set_xlabel("hex x")
    ax.set_ylabel("hex y")
    ax.set_title("Decoder conv_1: average |weight| across space")
    ax.set_aspect("equal", adjustable="box")

    x_min, x_max = float(grid_points[:, 0].min()), float(grid_points[:, 0].max())
    y_min, y_max = float(grid_points[:, 1].min()), float(grid_points[:, 1].max())
    pad = 0.15 * max(x_max - x_min, y_max - y_min, 1.0)
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)

    fig.tight_layout()
    hex_path = output_dir / "decoder_spatial_weights.png"
    fig.savefig(hex_path)
    plt.close(fig)
    print(f"Saved hex scatter to {hex_path}")


if __name__ == "__main__":
    main()
