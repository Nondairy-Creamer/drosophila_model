"""
Visualize decoder and input convolution columns in phi/theta space.

Creates a 2x2 plot where each panel shows:
  - All columns (gray)
  - Columns included in the convolution for a chosen center column
  - The center column location

Also optionally plots the hex grid offsets used for convolution in p,q space.
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


def _load_model(config: dict, device: str) -> nm.NeuralModel:
    connectome_folder = Path(config['data']['connectome_folder'])

    cell_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_cell_info.parquet')
    synapse_counts_df = pd.read_parquet(connectome_folder / 'optic_lobes_synapse_counts.parquet')
    column_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_column_info.parquet')

    if len(config['model']['use_connectome_subset']) > 0:
        cell_info_df, synapse_counts_df = du.get_connectome_subset(
            cell_info_df, synapse_counts_df, config['model']['use_connectome_subset']
        )

    commit_hash = mu.get_git_commit_hash()

    model_type = config['model'].get('model_type')
    ModelClass = getattr(nm, model_type)

    model = ModelClass(
        synapse_counts_coo=synapse_counts_df[['pre_root_id', 'post_root_id', 'synapse_counts']].to_numpy(),
        root_ids=cell_info_df['root_id'].to_numpy(),
        cell_types=cell_info_df['type'].to_numpy(),
        neurotransmitters=cell_info_df['neurotransmitter'].to_numpy(),
        column_indices=cell_info_df['column_index'].to_numpy(),
        column_phi_theta=column_info_df[['phi', 'theta']].to_numpy(),
        column_p_q=column_info_df[['p', 'q']].to_numpy(),
        nonlinearity=config['model']['nonlinearity'],
        count_threshold=config['model']['synapse_count_threshold'],
        use_layer_norm=config['model']['use_layer_norm'],
        device=device,
        dtype=config['model']['dtype'],
        decoder_type=config['model']['decoder_type'],
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


def _plot_columns(model: nm.NeuralModel, center_indices: list[int], output_path: Path) -> None:
    phi = model.column_phi_theta[:, 0]
    theta = model.column_phi_theta[:, 1]

    nearest_indices = model.decoder.conv_layers[0].nearest_column_indices.detach().cpu().numpy()
    center_color = 'black'
    conv_cmap = plt.get_cmap('viridis')

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=150, sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, center_idx in zip(axes, center_indices):
        all_conv_indices = nearest_indices[center_idx]
        valid = all_conv_indices >= 0
        conv_indices = all_conv_indices[valid]
        conv_order = np.arange(all_conv_indices.size)[valid]

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


def _pq_to_xy(pq: np.ndarray) -> np.ndarray:
    """Project p,q hex lattice coordinates to flat x,y space."""
    angle_1 = np.pi / 6
    angle_2 = 2 * np.pi / 3 + angle_1
    basis = np.array([[np.cos(angle_1), np.cos(angle_2)],
                      [np.sin(angle_1), np.sin(angle_2)]])
    return (basis @ pq.T).T


def _plot_conv_grids(model: nm.NeuralModel, output_path: Path) -> None:
    decoder_offsets = nm.NeuralModelConv.hex_grid_pq_offsets(model.decoder.conv_sizes[0])
    grids = [("Decoder grid", decoder_offsets)]
    if model.use_input_conv:
        input_offsets = nm.NeuralModelConv.hex_grid_pq_offsets(model.input_conv_size_x)
        grids.append(("Input grid", input_offsets))

    fig, axes = plt.subplots(
        1,
        len(grids),
        figsize=(6 * len(grids), 5),
        dpi=150,
        sharex=False,
        sharey=False,
    )
    if len(grids) == 1:
        axes = [axes]

    for ax, (title, offsets) in zip(axes, grids):
        xy = _pq_to_xy(offsets.astype(np.float64))
        ax.scatter(xy[:, 0], xy[:, 1], s=50, color='tab:blue', alpha=0.9, edgecolors='none')
        ax.scatter(0, 0, s=80, color='black', edgecolors='none')
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal', adjustable='box')

    fig.suptitle('Hex grid convolution offsets')
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_input_conv_columns(model: nm.NeuralModel, center_indices: list[int], output_path: Path) -> None:
    phi = model.column_phi_theta[:, 0]
    theta = model.column_phi_theta[:, 1]

    nearest_indices = model.input_spatial_conv.nearest_column_indices.detach().cpu().numpy()
    center_color = 'black'
    conv_cmap = plt.get_cmap('viridis')

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=150, sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, center_idx in zip(axes, center_indices):
        all_conv_indices = nearest_indices[center_idx]
        valid = all_conv_indices >= 0
        conv_indices = all_conv_indices[valid]
        conv_order = np.arange(all_conv_indices.size)[valid]

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
    parser.add_argument(
        "--grid-output",
        type=str,
        default="figures/convolution_grid.png",
        help="Output image path for convolution grids",
    )
    args = parser.parse_args()

    config = du.load_config(Path(args.config))

    if config['model']['decoder_type'] == 'none':
        print("Error: decoder_type is 'none' in config. Nothing to analyze.")
        sys.exit(1)

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

    _plot_conv_grids(model, Path(args.grid_output))
    print(f"Saved plot to {args.grid_output}")


if __name__ == "__main__":
    main()
