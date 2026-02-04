"""
Visualize decoder and input convolution columns in phi/theta space.

Creates a 2x2 plot where each panel shows:
  - All columns (gray)
  - Columns included in the convolution for a chosen center column
  - The center column location
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import data_utilities as du
import model_utilities as mu
import neural_models as nm


def _load_model(config: dict, device: str) -> nm.CurrentModel:
    connectome_folder = Path(config['data']['connectome_folder'])

    cell_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_cell_info.parquet')
    synapse_counts_df = pd.read_parquet(connectome_folder / 'optic_lobes_synapse_counts.parquet')
    column_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_column_info.parquet')

    if len(config['model']['use_connectome_subset']) > 0:
        cell_info_df, synapse_counts_df = du.get_connectome_subset(
            cell_info_df, synapse_counts_df, config['model']['use_connectome_subset']
        )

    column_phi_theta = column_info_df[['phi', 'theta']].to_numpy()
    commit_hash = mu.get_git_commit_hash()

    model = nm.CurrentModel(
        synapse_counts_coo=synapse_counts_df[['pre_root_id', 'post_root_id', 'synapse_counts']].to_numpy(),
        root_ids=cell_info_df['root_id'].to_numpy(),
        cell_types=cell_info_df['type'].to_numpy(),
        neurotransmitters=cell_info_df['neurotransmitter'].to_numpy(),
        column_indices=cell_info_df['column_index'].to_numpy(),
        column_phi_theta=column_phi_theta,
        nonlinearity=config['model']['nonlinearity'],
        count_threshold=config['model']['synapse_count_threshold'],
        device=device,
        dtype=config['model']['dtype'],
        use_decoder=True,
        use_input_conv=config['model']['use_input_conv'],
        dt=config['model']['dt'],
        seed=None,
        commit_hash=commit_hash,
        init_new_model=False,
    )

    return model


def _choose_center_indices(column_phi_theta: np.ndarray) -> list[int]:
    num_columns = column_phi_theta.shape[0]
    if num_columns == 0:
        return []

    num_select = min(4, num_columns)
    rng = np.random.default_rng()
    return rng.choice(num_columns, size=num_select, replace=False).tolist()


def _plot_columns(model: nm.CurrentModel, center_indices: list[int], output_path: Path) -> None:
    column_theta_phi = model.column_phi_theta
    phi = column_theta_phi[:, 0]
    theta = column_theta_phi[:, 1]

    nearest_indices = model.nearest_column_indices.detach().cpu().numpy()
    center_color = 'black'
    conv_cmap = plt.get_cmap('viridis')

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=150, sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, center_idx in zip(axes, center_indices):
        conv_indices = nearest_indices[center_idx]
        conv_order = np.arange(conv_indices.size)

        ax.scatter(phi, theta, s=18, color='lightgray', alpha=0.7, edgecolors='none')
        conv_scatter = ax.scatter(
            phi[conv_indices],
            theta[conv_indices],
            s=28,
            c=conv_order,
            cmap=conv_cmap,
            alpha=0.9,
            edgecolors='none'
        )
        ax.scatter(phi[center_idx], theta[center_idx], s=60, color=center_color, edgecolors='none')
        fig.colorbar(conv_scatter, ax=ax, shrink=0.8, pad=0.02, label='Convolution index')

        ax.set_title(f'Center column {center_idx}')
        ax.set_xlabel('phi')
        ax.set_ylabel('theta')
        ax.set_aspect('equal', adjustable='box')

    fig.suptitle('Decoder convolution columns in phi/theta space')
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_input_conv_columns(model: nm.CurrentModel, center_indices: list[int], output_path: Path) -> None:
    column_theta_phi = model.column_phi_theta
    phi = column_theta_phi[:, 0]
    theta = column_theta_phi[:, 1]

    nearest_indices = model.input_nearest_column_indices.detach().cpu().numpy()
    center_color = 'black'
    conv_cmap = plt.get_cmap('viridis')

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=150, sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, center_idx in zip(axes, center_indices):
        conv_indices = nearest_indices[center_idx]
        conv_order = np.arange(conv_indices.size)

        ax.scatter(phi, theta, s=18, color='lightgray', alpha=0.7, edgecolors='none')
        conv_scatter = ax.scatter(
            phi[conv_indices],
            theta[conv_indices],
            s=28,
            c=conv_order,
            cmap=conv_cmap,
            alpha=0.9,
            edgecolors='none'
        )
        ax.scatter(phi[center_idx], theta[center_idx], s=60, color=center_color, edgecolors='none')
        fig.colorbar(conv_scatter, ax=ax, shrink=0.8, pad=0.02, label='Input convolution index')

        ax.set_title(f'Center column {center_idx}')
        ax.set_xlabel('phi')
        ax.set_ylabel('theta')
        ax.set_aspect('equal', adjustable='box')

    fig.suptitle('Input convolution columns in phi/theta space')
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze decoder convolution columns.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.toml",
        help="Path to config.toml (default: config.toml in repo root)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to initialize model on (default: cpu)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/decoder_convolution_columns.png",
        help="Output image path",
    )
    parser.add_argument(
        "--input-output",
        type=str,
        default="figures/input_convolution_columns.png",
        help="Output image path for input convolution columns",
    )
    parser.add_argument(
        "--center-indices",
        type=int,
        nargs=4,
        default=None,
        help="Optional 4 center column indices to plot",
    )
    args = parser.parse_args()

    config = du.load_config(Path(args.config))
    model = _load_model(config, args.device)

    if args.center_indices is None:
        center_indices = _choose_center_indices(model.column_phi_theta)
    else:
        center_indices = list(args.center_indices)

    _plot_columns(model, center_indices, Path(args.output))
    print(f"Saved plot to {args.output}")

    if model.use_input_conv:
        _plot_input_conv_columns(model, center_indices, Path(args.input_output))
        print(f"Saved plot to {args.input_output}")
    else:
        print("Skipping input convolution plot because use_input_conv is disabled.")


if __name__ == "__main__":
    main()
