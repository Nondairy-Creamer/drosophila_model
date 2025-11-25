import numpy as np
import pandas as pd
from pathlib import Path
import data_utilities as du
from matplotlib import pyplot as plt


# Load configuration from TOML file
config = du.load_config()

# --- load in connectivity and cell info data ---
connectome_folder = Path(config['data']['connectome_folder'])
cell_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_cell_info.parquet')
column_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_column_info.parquet')
synapse_counts_df = pd.read_parquet(connectome_folder / 'optic_lobes_synapse_counts.parquet')
num_neurons = len(cell_info_df)
num_columns = len(column_info_df)


# get all neurons that synpase onto a specific cell and where they are spatially

chosen_type = 'T4a'
chosen_type_indexes = np.where(cell_info_df['type'] == chosen_type)[0]
chosen_presyn = ['Mi9', 'Mi1', 'Mi4']
chosen_presyn_index = [np.where(cell_info_df.type == i)[0] for i in chosen_presyn]
all_chosen = np.unique(np.concatenate(chosen_presyn_index))


pre_syn = {i: [] for i in chosen_presyn}
# loop through each T4a cell and plot the location of its connections
for cti in chosen_type_indexes:
    # get the neurons presynaptic to this neuron
    this_column_id = cell_info_df.iloc[cti].column_id

    if np.isnan(this_column_id):
        continue

    this_column_id = int(this_column_id)

    column_x = column_info_df[column_info_df.column_id == this_column_id].x_flat.to_numpy()
    column_y = column_info_df[column_info_df.column_id == this_column_id].y_flat.to_numpy()
    chosen_type_xy = np.array([column_x, column_y]).T

    these_connections = synapse_counts_df[synapse_counts_df.post_matrix_ind == cti]
    these_connections = these_connections[these_connections.pre_matrix_ind.isin(all_chosen)]
    these_connections_indexes = these_connections.pre_matrix_ind.to_numpy()

    for tc in these_connections_indexes:
        xy = np.zeros(2)
        tc_column_id = cell_info_df.iloc[tc].column_id

        if np.isnan(tc_column_id):
            continue

        tc_column_id = int(tc_column_id)

        xy[0] = column_info_df.loc[column_info_df.column_id == tc_column_id, 'x_flat'].iloc[0]
        xy[1] = column_info_df.loc[column_info_df.column_id == tc_column_id, 'y_flat'].iloc[0]
        tc_type = cell_info_df.iloc[tc].type
        pre_syn[tc_type].append(xy[None, :] - chosen_type_xy)

pre_syn = {i: np.concatenate(pre_syn[i], axis=0) for i in chosen_presyn}

plt.figure()
colors = ['orange', 'blue', 'red']
for pi, p in enumerate(pre_syn):
    rand_scatter = np.random.randn(pre_syn[p].shape[0], pre_syn[p].shape[1]) / 10
    plt.scatter(pre_syn[p][:, 0] + rand_scatter[:, 0], pre_syn[p][:, 1]+ rand_scatter[:, 1], color=colors[pi])
plt.legend(pre_syn.keys())
plt.xlabel('<- anterior, posterior ->')
plt.ylabel('<- down, up ->')
plt.show()
a=1