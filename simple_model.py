import model_utilities as mu
import torch
import numpy as np
import pandas as pd
from pathlib import Path

device = 'cpu'
dtype = torch.float32
dt = 0.01  # in seconds
max_time = 1 # in seconds
time_points = torch.arange(0, max_time, step=dt)
num_time = len(time_points)
synapse_strength_init_log = np.log(0.01)
temporal_frequencies = list(np.arange(2, 12, step=2))
nonlinearity = torch.relu

# --- load in connectivity and cell info data ---
connectome_folder = Path('/Users/mcreamer/Documents/drosophila_connectome')
column_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_column_info.parquet')
cell_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_cell_info.parquet')
synapse_counts_df = pd.read_parquet(connectome_folder / 'optic_lobes_synapse_counts.parquet')
num_neurons = len(cell_info_df.index)

# --- define the parameters and get the sparse connectivity matrix ---
synapse_counts = mu.create_sparse_matrix(synapse_counts_df, device=device, dtype=dtype)
diagonal = torch.full((synapse_counts.shape[0],), 0.9, requires_grad=True, dtype=dtype, device=device)
synapse_strength = torch.tensor(synapse_strength_init_log, requires_grad=True, dtype=dtype, device=device)
dynamics_matrix = torch.exp(synapse_strength) * synapse_counts + torch.diag(diagonal)

# --- get a sign wave stimulus ---
# generate a sine wave on a cylinder around the fly
sample_phi = column_info_df[column_info_df['type'] == 'L1']['phi'].values
sample_phi = torch.tensor(sample_phi, dtype=dtype, device=device)
sample_theta = column_info_df[column_info_df['type'] == 'L1']['theta'].values
sample_theta = torch.tensor(sample_theta, dtype=dtype, device=device)
sine_waves = mu.create_sign_wave(sample_phi, sample_theta, temporal_frequency=temporal_frequencies, dt=dt)

# --- define the external inputs to cells ---
# in this case we will directly stimulate L1, L2, and L3 with negative contrast
# we aren't stimulating receptors because they have a complicated relationship between location and viewing angle
# we are using negative contrast because L1-3 are inhibited by light

cell_inputs = []
for sw in sine_waves:
    cell_inputs.append(torch.zeros((num_neurons, num_time), dtype=dtype, device=device))
    cell_inputs[-1][column_info_df['type'] == 'L1', :] = -sw

# --- run the model forward ---
x = torch.zeros((num_neurons, num_time), dtype=dtype, device=device) + 0.1
for ti in range(num_time):
    x[:, ti] = dynamics_matrix @ nonlinearity(x[:, ti-1])
    print(ti, '/', num_time)

a=1
