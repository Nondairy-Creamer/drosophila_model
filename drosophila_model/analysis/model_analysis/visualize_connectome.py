"""
Plot the average synapse count between every pair of cell types of a model.

Saves ``<model_folder>/figures/visualize_connectome/connectome_matrix.png``.

Usage:
    python -m drosophila_model.analysis.model_analysis.visualize_connectome models/<model>/
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from drosophila_model.model_definition import neural_models as nm
from drosophila_model.analysis.model_analysis import analysis_utilities as au


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the average synapse count between cell types.")
    parser.add_argument("model_folder", type=str,
                        help="Path to model folder or model.pt")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to load the model on (default: cpu)")
    args = parser.parse_args()

    model_path = Path(args.model_folder)
    model_folder = model_path.parent if model_path.is_file() else model_path
    model = nm.BaseRnn.load(model_path, device=args.device)

    # Compute average connections between cell types
    ave_con = au.compute_average_connections(model)

    # Get cell type labels
    cell_types = model.unique_cell_types

    # Create the figure
    fig, ax = plt.subplots(figsize=(14, 12))

    # Apply signed cube root transform for compression (handles negative values naturally)
    ave_con_transformed = np.cbrt(ave_con)

    # Plot the connection matrix
    vmax = np.abs(ave_con_transformed).max()
    im = ax.imshow(ave_con_transformed, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')

    # Add colorbar with original scale tick labels
    cbar = fig.colorbar(im, ax=ax)
    # Create ticks in transformed space, label with original values
    cbar_ticks = np.linspace(-vmax, vmax, 9)
    cbar_labels = [f'{v**3:.0f}' for v in cbar_ticks]  # Convert back to original scale
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(cbar_labels)
    cbar.set_label('Average synapse count', fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # Set tick labels
    ax.set_xticks(np.arange(len(cell_types)))
    ax.set_yticks(np.arange(len(cell_types)))
    ax.set_xticklabels(cell_types, rotation=90, fontsize=12)
    ax.set_yticklabels(cell_types, fontsize=12)

    # Labels: ave_con[post, pre] so rows=post, cols=pre
    ax.set_xlabel('Presynaptic cell type', fontsize=14)
    ax.set_ylabel('Postsynaptic cell type', fontsize=14)
    ax.set_title(f'Average synapse counts between cell types\n({len(cell_types)} cell types)', fontsize=16)

    fig.tight_layout()
    output_dir = model_folder / 'figures' / 'visualize_connectome'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'connectome_matrix.png'
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved connectome matrix to {output_path}")


if __name__ == "__main__":
    main()
