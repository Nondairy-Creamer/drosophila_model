"""
Analyze decoder convolution weights.

Creates per decoder:
  1. Bar plot: average |weight| per neuron type across all channels and spatial positions
  2. Hex scatter heatmap: average |weight| at each spatial position in the conv kernel
  3. Per-neuron weight maps: one subplot per output channel showing spatial weights
"""
import argparse
import math
import sys
from pathlib import Path

# Add parent directory to path for imports

import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from drosophila_model.model_definition import neural_models as nm
from drosophila_model.model_definition import model_layers as ml


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
        help="Output directory for figures (default: <model_folder>/figures/analyze_decoder)",
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
    model = nm.BaseRnn.load(model_path, device=args.device)

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

    output_dir = (Path(args.output_dir) if args.output_dir
                  else model_folder / "figures" / "analyze_decoder")
    output_dir.mkdir(parents=True, exist_ok=True)

    for readout_name, decoder in decoders.items():
        print(f"\n=== Analyzing {readout_name} ===")
        readout_dir = output_dir / readout_name.lower()
        readout_dir.mkdir(parents=True, exist_ok=True)
        analyze_readout(decoder, readout_name, readout_dir, readout_name.lower(), args.top_n)


def _channel_grid_layout(num_channels):
    """Return (nrows, ncols) for a compact subplot grid."""
    ncols = math.ceil(math.sqrt(num_channels))
    nrows = math.ceil(num_channels / ncols)
    return nrows, ncols


def _hex_scatter_size(num_points):
    """Auto-scale scatter dot size so hex grids of different sizes look good."""
    return min(760, 760 * 7 / max(num_points, 1))


def analyze_readout(decoder, readout_name, output_dir, prefix, top_n):
    """Analyze a single hex-grid readout decoder."""
    conv_1 = decoder.conv_layers[0].weight.detach().cpu().numpy()
    all_neuron_names = np.array(decoder.neurons_to_decode)

    # Rank neuron types by average |weight| and keep the top N
    avg_per_neuron_all = np.mean(np.abs(conv_1), axis=(0, 1))
    num_to_show = min(top_n, len(all_neuron_names))
    top_indices = np.argsort(avg_per_neuron_all)[::-1][:num_to_show]

    neuron_names = list(all_neuron_names[top_indices])
    avg_per_neuron = avg_per_neuron_all[top_indices]
    conv_1_top = conv_1[:, :, top_indices]

    pq_offsets = ml.HexGridConv.hex_grid_pq_offsets(decoder.conv_sizes[0])
    grid_points = ml.HexGridConv.pq_to_xy(pq_offsets)
    num_spatial = grid_points.shape[0]
    scatter_s = _hex_scatter_size(num_spatial)

    # ------------------------------------------------------------------
    # Bar plot: average |weight| per neuron type (top N)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * num_to_show), 5), dpi=150)
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

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    sc = ax.scatter(
        grid_points[:, 0],
        grid_points[:, 1],
        c=avg_per_position,
        cmap="viridis",
        s=scatter_s,
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
    # Per-neuron conv_1 weight maps (one subplot per output channel)
    # ------------------------------------------------------------------
    channels_1 = conv_1_top.shape[0]
    nrows, ncols = _channel_grid_layout(channels_1)

    for neuron_idx, neuron_name in enumerate(neuron_names):
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4 * ncols, 4 * nrows), dpi=150,
                                 squeeze=False)

        weights_all_channels = conv_1_top[:, :, neuron_idx]
        vmax = np.max(np.abs(weights_all_channels))
        if vmax == 0:
            vmax = 1.0

        for ch_idx in range(channels_1):
            row, col = divmod(ch_idx, ncols)
            ax = axes[row, col]
            weights = conv_1_top[ch_idx, :, neuron_idx]

            scatter = ax.scatter(
                grid_points[:, 0],
                grid_points[:, 1],
                c=weights,
                cmap="coolwarm",
                vmin=-vmax,
                vmax=vmax,
                s=scatter_s,
                linewidths=0.6,
                edgecolors="black",
            )
            ax.set_title(f"Channel {ch_idx}", fontsize=10)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.04)

        for idx in range(channels_1, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row, col].axis("off")

        fig.suptitle(f"{readout_name} conv_1 weights — {neuron_name}", fontsize=14)
        fig.tight_layout()

        save_path = output_dir / f"{prefix}_conv1_weights_{neuron_name}.png"
        fig.savefig(save_path)
        plt.close(fig)
        print(f"Saved {save_path}")


if __name__ == "__main__":
    main()
