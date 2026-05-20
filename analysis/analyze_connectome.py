import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import pandas as pd
import data_utilities as du
import connectome_loading as cl
from matplotlib import pyplot as plt
import neural_models as nm


# Load configuration from TOML file
config = du.load_config()

# --- Initialize a current-based model from the standardised connectome loader ---
ModelClass = nm.compose_model_class('CurrentDynamics')
model = ModelClass(
    connectome=cl.load_connectome(
        areas=config['model']['areas'],
        side=config['model']['side'],
        connectome_folder=config['data']['connectome_folder'],
    ),
    device='cpu',
    init_new_model=False,  # Skip steady state finding for analysis
)

# Extract data from the model (uses filtered neurons/synapses)
cell_types = model.cell_types
column_indices = model.column_indices
column_phi_theta = model.column_phi_theta  # (num_columns, 2) with (phi, theta)
num_neurons = model.num_neurons
num_columns = model.num_columns

# Extract COO data from model.synapse_counts sparse tensor (post all model manipulation)
synapse_counts_sparse = model.synapse_counts.coalesce()
coo_indices = synapse_counts_sparse.indices().cpu().numpy()  # shape (2, num_synapses): [row=post, col=pre]
coo_values = synapse_counts_sparse.values().cpu().numpy()
coo_post = coo_indices[0]  # postsynaptic neuron indices
coo_pre = coo_indices[1]   # presynaptic neuron indices


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

    # Find presynaptic connections to this neuron from the COO arrays
    # coo_post has postsynaptic indices, coo_pre has presynaptic indices
    post_mask = coo_post == cti
    pre_indices = coo_pre[post_mask]
    
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

pre_syn = {i: np.concatenate(pre_syn[i], axis=0) for i in chosen_presyn}

# Create output directory
output_dir = Path('figures/connectome_analysis')
output_dir.mkdir(parents=True, exist_ok=True)

plt.figure()
colors = ['orange', 'blue', 'red']
for pi, p in enumerate(pre_syn):
    rand_scatter = 0*np.random.randn(pre_syn[p].shape[0], pre_syn[p].shape[1]) / 10
    plt.scatter(pre_syn[p][:, 0] + rand_scatter[:, 0], pre_syn[p][:, 1]+ rand_scatter[:, 1], color=colors[pi])
plt.legend(pre_syn.keys())
plt.xlabel('delta phi (azimuth)')
plt.ylabel('delta theta (elevation)')
plt.title(f'Presynaptic inputs to {chosen_type}')
plt.savefig(output_dir / f'presynaptic_inputs_to_{chosen_type}.png', dpi=150, bbox_inches='tight')
plt.close()