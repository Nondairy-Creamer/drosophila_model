import numpy as np
from pathlib import Path
import pickle
import pandas as pd


connectome_folder = Path('/Users/mcreamer/Documents/drosophila_connectome')
visual_neuron_file = connectome_folder / 'visual_neuron_types.csv'
synapse_counts_file = connectome_folder / 'connections_princeton.csv'

# each neuron has a unique "root id" which we can load from this file
# this file only includes neurons that make a synapse in the visual system
visual_neurons_df = pd.read_csv(visual_neuron_file, dtype={"root_id": np.int64})
neuron_ids = visual_neurons_df.root_id.values
neuron_cell_type = visual_neurons_df.type.values
# category specifies whether the neuron is intrinsic (all synapses in the optic lobes)
# or has some synapses outside the optic lobes
category = visual_neurons_df.category.values


# keep only the intrinsic visual neurons
is_intrinsic = category == 'intrinsic'
neuron_ids = neuron_ids[is_intrinsic]
neuron_cell_type = neuron_cell_type[is_intrinsic]

# load synaptic connections and only include those that intrinsic to the optic lobes
# synapse counts specifies the number of synaptic contacts between one individual neuron and another
# we change the order of the columns so that they are [presynaptic neuron, postsynaptic neuron, synapse counts]
synaptic_connections_df = pd.read_csv(synapse_counts_file, dtype={"pre_root_id": np.int64, "post_root_id": np.int64, "syn_count": np.int16})
synaptic_connections_df = synaptic_connections_df[synaptic_connections_df['pre_root_id'].isin(neuron_ids)]
synaptic_connections_df = synaptic_connections_df[synaptic_connections_df['post_root_id'].isin(neuron_ids)]

presynaptic_ids = synaptic_connections_df.pre_root_id.values
postsynaptic_ids = synaptic_connections_df.post_root_id.values
synapse_counts = synaptic_connections_df.syn_count.values
neurotransmitters = synaptic_connections_df.nt_type.values

synapse_sign = np.zeros_like(synapse_counts)
synapse_sign[neurotransmitters == 'ACH'] = 1
synapse_sign[neurotransmitters == 'GABA'] = -1
synapse_sign[neurotransmitters == 'GLUT'] = -1
synapse_counts *= synapse_sign

# we will have a separate cell ID vector that links index to root id so lets make that now
# first, get the list of all root IDs we actually use
used_root_ids = np.unique((presynaptic_ids, postsynaptic_ids))
num_neurons = len(used_root_ids)

# from this get the list of cell types for each root_id
neuron_ids_that_are_used = np.isin(neuron_ids, used_root_ids)
used_ids_cell_types = neuron_cell_type[neuron_ids_that_are_used]
# this is our vector of used root_ids and the cell types they belong to
cell_types_df = pd.DataFrame({'matrix_index': np.arange(num_neurons), 'root_id': used_root_ids, 'cell_type': used_ids_cell_types})

# currently, our synapse counts uses the pre / post root ID. but this isn't useful for making a sparse matrix. Instead,
# we will make our synapse counts matrix use the postsynaptic / presynaptic matrix index. this will be used to make
# a sparse matrix of the synaptic connectivity
# for speed, make a hash map that takes you from root id to matrix index
id_mapping = {}
pre_matrix_ind = np.zeros_like(presynaptic_ids)
post_matrix_ind = np.zeros_like(postsynaptic_ids)
for i in range(num_neurons):
    id_mapping[used_root_ids[i]] = i

# loop through all the pre / post cells and map them from root id to matrix index
for i in range(synapse_counts.shape[0]):
    pre_matrix_ind[i] = id_mapping[presynaptic_ids[i]]
    post_matrix_ind[i] = id_mapping[postsynaptic_ids[i]]

synapse_counts_df = pd.DataFrame({'pre_matrix_ind': pre_matrix_ind, 'post_matrix_ind': post_matrix_ind, 'synapse_counts': synapse_counts})

# Save the synapse counts and cell types to a dict to use in modeling
cell_types_df.to_parquet(connectome_folder / 'optic_lobes_cell_types.parquet', index=False)
synapse_counts_df.to_parquet(connectome_folder / 'optic_lobes_synapse_counts.parquet', index=False)

