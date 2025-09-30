import numpy as np
from pathlib import Path
import model_utilities as mu
import pandas as pd


connectome_folder = Path('/Users/mcreamer/Documents/drosophila_connectome')
synapse_count_dtype = np.float64
root_id_dtype = np.int64
excitatory_nt = ['ACH']
inhibitory_nt = ['GABA', 'GLUT']
fly_eye = 'right'
use_intrinsic = False
hemisphere_coverage = 0.85  # fraction of the hemisphere the fly's vision covers

# --- data sets ---
# each neuron in the drosophila connectome has a unique "root id"
# This data set that includes the root ID for each visual neuron, whether that neuron is intrinsic to the optic lobes
# and whether that neuron is on the right or left side
visual_neurons_df = pd.read_csv(connectome_folder / 'visual_neuron_types.csv', dtype={"root_id": root_id_dtype})

# data set that specifies the number of synaptic contacts between one individual neuron and another
# this also specifies the neurotransmitter used as the synapse which we'll use to estimate synapse sign
synaptic_connections_df = pd.read_csv(connectome_folder / 'connections_princeton.csv',
                                      dtype={"pre_root_id": root_id_dtype,
                                             "post_root_id": root_id_dtype,
                                             "syn_count": synapse_count_dtype}
                                      )

# The optic lobes are organized into retinotopic visual columns and this data set specifies the column id and location
# of each cell on a hexagonal grid
columns_df = pd.read_csv(connectome_folder / 'column_assignment.csv', dtype={"root_id": root_id_dtype})

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

# change synapse sign to reflect the neurotransmitter being used. If the neurotransmitter is not in our list, set
# the connection to 0
synapse_sign = np.zeros_like(synapse_counts)
excitatory_connections = synaptic_connections_df.nt_type.isin(excitatory_nt)
inhibitory = synaptic_connections_df.nt_type.isin(inhibitory_nt)
synapse_sign[excitatory_connections] = 1
synapse_sign[inhibitory] = -1

# make synapse counts signed
synapse_counts *= synapse_sign

# -- make vector with (matrix row, matrix col, value). Can be used to construct a sparse or dense matrix --
# Get the list of all root IDs we actually use
model_root_ids = np.unique((presynaptic_ids, postsynaptic_ids))
num_neurons = len(model_root_ids)

# limit cell info to those in the synaptic connectivity matrix and save
visual_neurons_df = visual_neurons_df[visual_neurons_df.root_id.isin(model_root_ids)]
# save the cell info
visual_neurons_df.to_parquet(connectome_folder / 'optic_lobes_cell_info.parquet', index=False)

# for speed, make a hash map that takes you from root id to matrix index
id_to_index_mapping = {}
pre_matrix_ind = np.zeros_like(presynaptic_ids)
post_matrix_ind = np.zeros_like(postsynaptic_ids)
for i in range(num_neurons):
    id_to_index_mapping[model_root_ids[i]] = i

# loop through all the pre / post cells and map them from root id to matrix index
for i in range(synapse_counts.shape[0]):
    pre_matrix_ind[i] = id_to_index_mapping[presynaptic_ids[i]]
    post_matrix_ind[i] = id_to_index_mapping[postsynaptic_ids[i]]

# save the matrix indices and values to a data frame
synapse_counts_df = pd.DataFrame({'pre_matrix_ind': pre_matrix_ind, 'post_matrix_ind': post_matrix_ind, 'synapse_counts': synapse_counts})
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
theta = 2 * np.asin(rho / np.sqrt(2))
phi = np.atan2(xy[1, :], xy[0, :])
r = np.ones_like(theta)
rtp = np.vstack((r, theta, phi))
xyz = mu.rtp_to_xyz(rtp)

# diagram at
# https://www.evernote.com/client/web#/notes/0e9caaaf-0a32-585c-55a5-f1687890c2a7
# goal is to align the fly looking forward down the x axis. +z is up. +y is left
# this means that in this frame, theta = 90 is directly in front of the fly
# the right eye is phi [0, 180] and left eye is phi [0, 180)
# positive phi rotation corresponds to rightward motion
# progressive / regressive => +/- phi
# dorsal / ventral => +/- theta

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
mu.scatter_3d_with_origin_axes(xyz)

# --- save the column data ---
column_info_df = columns_df.drop(columns = ['p', 'q', 'x', 'y'])
column_info_df['theta'] = rtp[1, :]
column_info_df['phi'] = rtp[2, :]
column_info_df['x'] = xyz[0, :]
column_info_df['y'] = xyz[1, :]
column_info_df['z'] = xyz[2, :]
column_info_df.to_parquet(connectome_folder / 'optic_lobes_column_info.parquet', index=False)

