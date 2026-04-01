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

from analysis_utilities import hex_grid_points


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze decoder convolution weights."
    )
    parser.add_argument(
        "model_folder",
        type=str,
        help="Path to model folder or model.pt",
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
    parser.add_argument(
        "--readout",
        type=str,
        default=None,
        help="Name of readout to analyze (default: all hex-grid readouts)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of top-weighted neuron types to plot (default: 30)",
    )
    args = parser.parse_args()

    model_path = Path(args.model_folder)
    model_folder = model_path.parent if model_path.is_file() else model_path
    model = nm.NeuralModel.load(model_path, device=args.device)

    # Collect hex-grid readouts to analyze
    decoders = {}
    for name, mod in model._readout_modules.items():
        if hasattr(mod, 'conv_layers') and not getattr(mod, 'training_only', False):
            decoders[name] = mod
    if not decoders:
        raise ValueError("Model does not have any hex-grid decoder readouts.")

    if args.readout:
        if args.readout not in decoders:
            raise ValueError(f"Readout {args.readout!r} not found. Available: {list(decoders.keys())}")
        decoders = {args.readout: decoders[args.readout]}

    output_dir = Path(args.output_dir) if args.output_dir else model_folder / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    for readout_name, decoder in decoders.items():
        print(f"\n=== Analyzing {readout_name} ===")
        prefix = readout_name.lower()
        analyze_readout(decoder, readout_name, output_dir, prefix, args.top_n)


def analyze_readout(decoder, readout_name, output_dir, prefix, top_n):
    """Analyze a single hex-grid readout decoder."""
    conv_1 = decoder.conv_layers[0].weight.detach().cpu().numpy()
    all_neuron_names = np.array(decoder.neurons_to_decode)

    # Rank neuron types by average |weight| and keep the top N
    avg_per_neuron_all = np.mean(np.abs(conv_1), axis=(0, 1))
    num_to_show = min(top_n, len(all_neuron_names))
    top_indices = np.argsort(avg_per_neuron_all)[::-1][:num_to_show]
    top_indices_sorted = np.sort(top_indices)

    neuron_names = list(all_neuron_names[top_indices_sorted])
    avg_per_neuron = avg_per_neuron_all[top_indices_sorted]
    conv_1_top = conv_1[:, :, top_indices_sorted]

    # ------------------------------------------------------------------
    # Bar plot: average |weight| per neuron type (top N)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    x_pos = np.arange(len(neuron_names))
    ax.bar(x_pos, avg_per_neuron, color="tab:blue", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(neuron_names, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Neuron type")
    ax.set_ylabel("Mean |weight|")
    ax.set_title(f"{readout_name} conv_1: average |weight| per neuron (top {num_to_show} of {len(all_neuron_names)})")
    fig.tight_layout()

    bar_path = output_dir / f"{prefix}_neuron_weights.png"
    fig.savefig(bar_path)
    plt.close(fig)
    print(f"Saved bar plot to {bar_path}")

    # ------------------------------------------------------------------
    # Hex scatter heatmap: average |weight| per spatial position
    # ------------------------------------------------------------------
    avg_per_position = np.mean(np.abs(conv_1_top), axis=(0, 2))

    axial_coords = nm.NeuralModelConv.hex_grid_axial_coords(decoder.conv_sizes[0])
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
    ax.set_title(f"{readout_name} conv_1: average |weight| across space")
    ax.set_aspect("equal", adjustable="box")

    x_min, x_max = float(grid_points[:, 0].min()), float(grid_points[:, 0].max())
    y_min, y_max = float(grid_points[:, 1].min()), float(grid_points[:, 1].max())
    pad = 0.15 * max(x_max - x_min, y_max - y_min, 1.0)
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)

    fig.tight_layout()
    hex_path = output_dir / f"{prefix}_spatial_weights.png"
    fig.savefig(hex_path)
    plt.close(fig)
    print(f"Saved hex scatter to {hex_path}")

    # ------------------------------------------------------------------
    # Per-neuron conv_1 weight maps
    # ------------------------------------------------------------------
    channels_1 = conv_1_top.shape[0]

    # Map channels to grid positions, skipping center (1,1) for 3x3
    positions = [(0, 0), (0, 1), (0, 2),
                 (1, 0),         (1, 2),
                 (2, 0), (2, 1), (2, 2)]

    for neuron_idx, neuron_name in enumerate(neuron_names):
        fig, axes = plt.subplots(3, 3, figsize=(10, 10), dpi=150)

        weights_all_channels = conv_1_top[:, :, neuron_idx]
        vmax = np.max(np.abs(weights_all_channels))
        if vmax == 0:
            vmax = 1.0

        for ch_idx in range(min(channels_1, len(positions))):
            row, col = positions[ch_idx]
            ax = axes[row, col]
            weights = conv_1_top[ch_idx, :, neuron_idx]

            scatter = ax.scatter(
                grid_points[:, 0],
                grid_points[:, 1],
                c=weights,
                cmap="coolwarm",
                vmin=-vmax,
                vmax=vmax,
                s=760,
                linewidths=0.6,
                edgecolors="black",
            )
            ax.set_title(f"Channel {ch_idx}", fontsize=10)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.04)

        axes[1, 1].axis("off")

        for ch_idx in range(channels_1, len(positions)):
            row, col = positions[ch_idx]
            axes[row, col].axis("off")

        fig.suptitle(f"{readout_name} conv_1 weights — {neuron_name}", fontsize=14)
        fig.tight_layout()

        save_path = output_dir / f"{prefix}_conv1_weights_{neuron_name}.png"
        fig.savefig(save_path)
        plt.close(fig)
        print(f"Saved {save_path}")


if __name__ == "__main__":
    main()
