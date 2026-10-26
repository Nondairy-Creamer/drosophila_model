"""
Spatial layout of the presynaptic inputs to T4a.

Builds a current-based model from the full connectome in the local
config.toml's data folder (its [model.regions] filters and
synapse_count_threshold apply; [model].use_connectome_subset does not) and,
for every T4a neuron, scatters the (phi, theta) offset of each Mi9 / Mi1 /
Mi4 input column relative to the T4a's own column.

Usage (from the repo root):
    python -m drosophila_model.analysis.model_analysis.analyze_connectome

Writes figures/analyze_connectome/presynaptic_inputs_to_T4a.png.
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from drosophila_model.config import get_data_folder, load_config
from drosophila_model.model_definition import connectome_loading as cl
from drosophila_model.model_definition import neural_models as nm
from drosophila_model.model_definition import regions as regions_mod


def main() -> None:
    config = load_config()
    count_threshold = config['model']['synapse_count_threshold']
    region_kwargs = config['model'].get('regions', {})

    # --- Initialize a current-based model from the standardised connectome loader ---
    ModelClass = nm.compose_model_class('CurrentDynamics')
    model = ModelClass(
        connectome=cl.load_connectome(
            region_specs=regions_mod.build_region_specs(region_kwargs),
            data_folder=get_data_folder(config),
            count_threshold=count_threshold,
        ),
        count_threshold=count_threshold,
        region_kwargs=region_kwargs,
        device='cpu',
        init_new_model=False,  # skip set_sigma_max; only connectivity is used
    )

    # Extract data from the model (uses filtered neurons/synapses)
    cell_types = model.cell_types
    column_indices = model.regions['optic_lobe'].column_indices_per_neuron
    column_phi_theta = model.regions['optic_lobe'].column_phi_theta  # (num_columns, 2) with (phi, theta)

    # Edge list after all model preprocessing; columns are [pre, post, value].
    coo = model.synapse_counts_coo
    coo_pre = coo[:, 0].astype(int)
    coo_post = coo[:, 1].astype(int)

    # get all neurons that synapse onto a specific cell and where they are spatially
    chosen_type = 'T4a'
    chosen_type_indexes = np.where(cell_types == chosen_type)[0]
    chosen_presyn = ['Mi9', 'Mi1', 'Mi4']
    chosen_presyn_index = [np.where(cell_types == i)[0] for i in chosen_presyn]
    all_chosen = np.unique(np.concatenate(chosen_presyn_index))

    pre_syn = {i: [] for i in chosen_presyn}
    # loop through each T4a cell and plot the location of its connections
    for cti in chosen_type_indexes:
        # get the column for this neuron
        this_column_idx = column_indices[cti]

        if np.isnan(this_column_idx):
            continue

        this_column_idx = int(this_column_idx)

        # Get the phi, theta position of this column
        column_phi = column_phi_theta[this_column_idx, 0]
        column_theta = column_phi_theta[this_column_idx, 1]
        chosen_type_xy = np.array([[column_phi, column_theta]])

        # Presynaptic partners of this neuron
        pre_indices = coo_pre[coo_post == cti]

        # Filter to only chosen presynaptic types
        in_chosen = np.isin(pre_indices, all_chosen)
        these_connections_indexes = pre_indices[in_chosen]

        for tc in these_connections_indexes:
            xy = np.zeros(2)
            tc_column_idx = column_indices[tc]

            if np.isnan(tc_column_idx):
                continue

            tc_column_idx = int(tc_column_idx)

            xy[0] = column_phi_theta[tc_column_idx, 0]
            xy[1] = column_phi_theta[tc_column_idx, 1]
            tc_type = cell_types[tc]
            pre_syn[tc_type].append(xy[None, :] - chosen_type_xy)

    pre_syn = {i: np.concatenate(pre_syn[i], axis=0) for i in chosen_presyn if pre_syn[i]}
    if not pre_syn:
        raise RuntimeError(
            f"No {chosen_presyn} inputs onto {chosen_type} found in this connectome."
        )

    # Create output directory
    output_dir = Path('figures/analyze_connectome')
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    colors = ['orange', 'blue', 'red']
    for pi, p in enumerate(pre_syn):
        plt.scatter(pre_syn[p][:, 0], pre_syn[p][:, 1], color=colors[pi])
    plt.legend(pre_syn.keys())
    plt.xlabel('delta phi (azimuth)')
    plt.ylabel('delta theta (elevation)')
    plt.title(f'Presynaptic inputs to {chosen_type}')
    save_path = output_dir / f'presynaptic_inputs_to_{chosen_type}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


if __name__ == '__main__':
    main()
