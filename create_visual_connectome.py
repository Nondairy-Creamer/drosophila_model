import numpy as np
from pathlib import Path
import model_utilities as mu
import pandas as pd
from data_utilities import load_config
import plotting_utilities as pu

# Load configuration
config = load_config()

# Extract parameters from config
connectome_folder = Path(config["data"]["connectome_folder"])
synapse_count_dtype = getattr(np, config["data"]["synapse_count_dtype"])
root_id_dtype = getattr(np, config["data"]["root_id_dtype"])
fly_eye = config["data"]["fly_eye"]
use_intrinsic = config["data"]["use_intrinsic"]
hemisphere_coverage = config["data"]["hemisphere_coverage"]
neurons_to_exclude = config['data']['neurons_to_exclude']

# --- data sets ---
# each neuron in the drosophila connectome has a unique "root id"
# This data set that includes the root ID for each visual neuron, whether that neuron is intrinsic to the optic lobes
# and whether that neuron is on the right or left side
visual_neurons_df = pd.read_csv(connectome_folder / 'visual_neuron_types_no_threshold.csv', dtype={"root_id": root_id_dtype})

# data set that specifies the number of synaptic contacts between one individual neuron and another
# this also specifies the neurotransmitter used as the synapse which we'll use to estimate synapse sign
synaptic_connections_df = pd.read_csv(connectome_folder / 'connections_no_threshold.csv',
                                      dtype={"pre_root_id": root_id_dtype,
                                             "post_root_id": root_id_dtype,
                                             "syn_count": synapse_count_dtype}
                                      )

# The optic lobes are organized into retinotopic visual columns and this data set specifies the column id and location
# of each cell on a hexagonal grid. not all visual neurons have this column information
columns_df = pd.read_csv(connectome_folder / 'column_assignments_no_threshold.csv', dtype={"root_id": root_id_dtype})

# --- process the incoming data ---
# restrict neurons to intrinsic visual neurons in the left eye
restriction = visual_neurons_df['side'] == fly_eye
if use_intrinsic:
    restriction &= visual_neurons_df['category'] == 'intrinsic'
visual_neurons_df = visual_neurons_df[restriction]

# --- limit the connections to those found in the optic lobes ---
# take the list of synaptic connections and limit it to the neuron ids from the visual neurons
pre_and_post_are_visual = (synaptic_connections_df.pre_root_id.isin(visual_neurons_df.root_id) &
                           synaptic_connections_df.post_root_id.isin(visual_neurons_df.root_id))
synaptic_connections_df = synaptic_connections_df[pre_and_post_are_visual]

# now that we have limited our neurons to the visual neurons, pull out the ids and synapse counts
presynaptic_ids = synaptic_connections_df.pre_root_id.values
postsynaptic_ids = synaptic_connections_df.post_root_id.values
synapse_counts = synaptic_connections_df.syn_count.values
neurotransmitters = synaptic_connections_df.nt_type.values

# --- Derive neurotransmitter per cell type ---
# For each presynaptic neuron, count the neurotransmitter usage across all its synapses
# Then aggregate by cell type to get the dominant neurotransmitter for each type
# Add presynaptic cell type to the connections
pre_root_to_type = visual_neurons_df.set_index('root_id')['type'].to_dict()
pre_types = np.array([pre_root_to_type.get(rid, None) for rid in presynaptic_ids])

# Create a dataframe to aggregate neurotransmitter by cell type
nt_df = pd.DataFrame({'cell_type': pre_types, 'nt_type': neurotransmitters, 'syn_count': synapse_counts})
# Weight by synapse count and get the most common neurotransmitter per cell type
nt_per_type = nt_df.groupby(['cell_type', 'nt_type'])['syn_count'].sum().reset_index()
idx = nt_per_type.groupby('cell_type')['syn_count'].idxmax()
cell_type_nt_df = nt_per_type.loc[idx, ['cell_type', 'nt_type']].reset_index(drop=True)
cell_type_nt_df.columns = ['cell_type', 'neurotransmitter']

# exclude photoreceptors and other neurons that interfere with learning
# Note: neurons_to_exclude uses cell TYPE names, not root IDs
# We need to find the root IDs of neurons with these types
excluded_root_ids = visual_neurons_df[visual_neurons_df['type'].isin(neurons_to_exclude)]['root_id'].values
# Exclude connections where these neurons are pre-synaptic OR post-synaptic
exclude_bool = np.isin(presynaptic_ids, excluded_root_ids) | np.isin(postsynaptic_ids, excluded_root_ids)
presynaptic_ids = presynaptic_ids[~exclude_bool]
postsynaptic_ids = postsynaptic_ids[~exclude_bool]
synapse_counts = synapse_counts[~exclude_bool]

# -- Get the list of all root IDs we actually use --
model_root_ids = np.unique((presynaptic_ids, postsynaptic_ids))
num_neurons = len(model_root_ids)

# limit cell info to those in the synaptic connectivity matrix. Make sure we keep the same ordering of model_root_ids
visual_neurons_indices = pd.Index(visual_neurons_df.root_id).get_indexer(model_root_ids)
# Verify all root IDs were found (no -1 indices) - this should always pass if data is consistent
assert np.all(visual_neurons_indices >= 0), "Some model_root_ids not found in visual_neurons_df - data inconsistency"
visual_neurons_df = visual_neurons_df.iloc[visual_neurons_indices]

# save the root IDs and synapse counts to a data frame
# the conversion from root IDs to matrix indices is done in NeuralModel.__init__
synapse_counts_df = pd.DataFrame({'pre_root_id': presynaptic_ids, 'post_root_id': postsynaptic_ids, 'synapse_counts': synapse_counts})
# Save the synapse counts and cell types to a dict to use in modeling
synapse_counts_df.to_parquet(connectome_folder / 'optic_lobes_synapse_counts.parquet', index=False)

# --- makea  data frame of column information ---
# make a dataframe that has all the information for the neurons in the model that is:
# root_id, hemisphere, type, column_id, theta, phi
columns_df = columns_df[columns_df.root_id.isin(model_root_ids)]

# Calculate theta and phi from the p and q hexagonal lattice coordinates
# first, project from p and q coordinates to cartesian space
# these coordinates were chosen by the connectome paper
pq = np.vstack((columns_df.p.values, columns_df.q.values))
angle_1 = np.pi / 6  # angle of the first basis
angle_2 = 2 * np.pi / 3 + angle_1  # the second basis is 120 degrees off
basis = np.array([[np.cos(angle_1), np.cos(angle_2)], [np.sin(angle_1), np.sin(angle_2)]])
# transform p and q to x and y
xy = basis @ pq
xy -= np.mean(xy, axis=1)[:, None]

# project x and y to angular spherical coordinates using Lambert azimuthal equal-area (LAEA) projection
# assume that the north pole of the sphere is at the center of the eye in our new x y coordinates
# in this mapping, a hemisphere lies within a radius of sqrt(2)
# the fly eye takes up roughly 3/4rds of the angle of a hemisphere
max_angle = np.pi / 2 * hemisphere_coverage
max_radius = np.sqrt(2) * np.sin(max_angle / 2)

# radius is calculated as sqrt(x^2+y^2)
# calculate a scalar for x and y such that sqrt(x^2+y^2) is less than max_radius
xy_scalar = np.min(np.sqrt(max_radius ** 2 / np.sum(xy ** 2, axis=0)))
xy *= xy_scalar

# convert to spherical angles
rho = np.sqrt(np.sum(xy ** 2, axis=0))
theta = 2 * np.arcsin(rho / np.sqrt(2))
phi = np.arctan2(xy[1, :], xy[0, :])
r = np.ones_like(theta)
rtp = np.vstack((r, theta, phi))
xyz = mu.rtp_to_xyz(rtp)

# diagram at
# https://www.evernote.com/client/web#/notes/0e9caaaf-0a32-585c-55a5-f1687890c2a7
# goal is to align the fly looking forward down the x axis. +z is up. +y is left
# this means that in this frame, theta = 90 is directly in front of the fly
# the right eye is phi < 0 (negative Y) and left eye is phi > 0 (positive Y)
# positive phi rotation corresponds to leftward motion (toward +Y)
# negative phi rotation corresponds to rightward motion (toward -Y)
# rightward / leftward optic flow => positive / negative optic_flow[..., 0]
# upward / downward optic flow => positive / negative optic_flow[..., 1]

# procedure
# flip x so that x 0 will align with the front of the font of the fly
# rot 90 degrees around the x axis
# rotate around z until one ommatidia faces front
xyz[0, :] = -xyz[0, :]
xyz = mu.rotate_3d(xyz, np.pi / 2, 'x')

# get the largest phi so we can rotate it to 0
rtp = mu.xyz_to_rtp(xyz)
# don't need to abs here. all points are on a hemisphere so after rotation all phi are positive
max_phi = np.max(rtp[2, :])
xyz = mu.rotate_3d(xyz, -max_phi, 'z')
rtp = mu.xyz_to_rtp(xyz)

# --- plot the ommatidia locations
pu.scatter_3d_with_origin_axes(xyz)

# --- save the column data ---
# rename the x and y values to "x_flat and y_flat"
column_info_df = columns_df.rename(columns={'x': 'x_flat', 'y': 'y_flat'})
column_info_df['theta'] = rtp[1, :]
column_info_df['phi'] = rtp[2, :]
column_info_df['x'] = xyz[0, :]
column_info_df['y'] = xyz[1, :]
column_info_df['z'] = xyz[2, :]

# save the column id to visual neurons for those neurons that have a column
visual_neurons_df['column_id'] = np.nan
row_pos = pd.Index(visual_neurons_df.root_id).get_indexer(column_info_df.root_id)
row_pos = row_pos[row_pos >= 0] # don't index into neurons that don't have a column
col_idx = visual_neurons_df.columns.get_loc('column_id')
visual_neurons_df.iloc[row_pos, col_idx] = column_info_df['column_id'].to_numpy().astype(int)
visual_neurons_df['column_index'] = visual_neurons_df['column_id'] - 1

# Add neurotransmitter to each neuron based on its cell type
type_to_nt = cell_type_nt_df.set_index('cell_type')['neurotransmitter'].to_dict()
visual_neurons_df['neurotransmitter'] = visual_neurons_df['type'].map(type_to_nt)

# get a df that just has all the unique columns and where they are in space. can map to the visual_neurons_df via column_id
cols = ['column_id', 'theta', 'phi', 'x', 'y', 'z', 'x_flat', 'y_flat', 'p', 'q']
unique_columns_df = (
    column_info_df
    .drop_duplicates(subset=['column_id'], keep='first')[cols]
    .sort_values('column_id')
    .reset_index(drop=True)
)
unique_columns_df['column_index'] = unique_columns_df['column_id'] - 1

# save the cell info (includes neurotransmitter per neuron) and column info
visual_neurons_df.to_parquet(connectome_folder / 'optic_lobes_cell_info.parquet', index=False)
unique_columns_df.to_parquet(connectome_folder / 'optic_lobes_column_info.parquet', index=False)

