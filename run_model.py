import model_utilities as mu
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
import neural_models as nm
import time


device = torch.device('cpu')
dtype = torch.float32
dt = 0.1  # in seconds
max_time = 1 # in seconds
num_steps = 2
alpha = 1e-3
time_points = torch.arange(0, max_time, step=dt)
num_time = len(time_points)
temporal_frequencies = list(np.arange(2, 12, step=2))
temporal_frequencies += [-i for i in temporal_frequencies]
nonlinearity = torch.relu
lam_neurons = ('L1', 'L2', 'L3')
input_neurons = lam_neurons
output_neuron = 'HSE'
use_sparse = False

# --- load in connectivity and cell info data ---
connectome_folder = Path('../connectome_data')
column_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_column_info.parquet')
cell_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_cell_info.parquet')
synapse_counts_df = pd.read_parquet(connectome_folder / 'optic_lobes_synapse_counts.parquet')
num_neurons = len(cell_info_df.index)

# --- define the parameters and get the sparse connectivity matrix ---
if use_sparse:
    synapse_counts = mu.create_sparse_matrix(synapse_counts_df, device=device, dtype=dtype)
else:
    synapse_counts = torch.zeros((num_neurons, num_neurons), device=device, dtype=dtype)
    synapse_counts_values = torch.tensor(synapse_counts_df['synapse_counts'].values, device=device, dtype=dtype)
    synapse_counts[synapse_counts_df['post_matrix_ind'], synapse_counts_df['pre_matrix_ind']] = synapse_counts_values
readout_weight = torch.nn.Parameter(torch.tensor(1e0, dtype=dtype, device=device))

# create the model
model = nm.SimpleRnn(synapse_counts=synapse_counts, device=device, dtype=dtype, use_sparse=use_sparse)

# --- get a sign wave stimulus ---
# generate a sine wave on a cylinder around the fly
l1_df = column_info_df[column_info_df['type'] == 'L1']
sine_waves = mu.create_sine_wave(l1_df.phi.to_numpy(), l1_df.theta.to_numpy(), temporal_frequency=temporal_frequencies, dt=dt)
sine_waves = [torch.tensor(i, dtype=dtype, device=device) for i in sine_waves]
temporal_frequencies = torch.tensor(temporal_frequencies, dtype=dtype, device=device)
num_stim = len(sine_waves)

# --- define the external inputs to cells ---
# in this case we will directly stimulate L1, L2, and L3 with negative contrast
# we aren't stimulating receptors because they have a complicated relationship between location and viewing angle
# we are using negative contrast because L1-3 are inhibited by light

# find a mapping from column ID to neuron index
# column_info_df has the column ID for the neurons. However, not all neurons are included in column_info_df
# so we need to get the root_id for each L1-3 neuron for each column. The, we need to convert those root_ids
# to the neuron_index in the larger cell_info_df.

cell_inputs = []
lam_column_bool = column_info_df['type'].isin(lam_neurons)
root_id_to_column_id = column_info_df.loc[lam_column_bool, ('column_id', 'root_id')]
root_id_to_column_id['root_ind'] = pd.Index(cell_info_df.root_id).get_indexer(root_id_to_column_id.root_id)

assert np.all(root_id_to_column_id['root_ind'] != -1), 'column_info has cells not found in cell_info'

# create a dictionary where the key is the column number and the output is the indices of the L1-3 cells in that column
col_to_cell_ind = (
    root_id_to_column_id.groupby('column_id', sort=False).root_ind
      .agg(list)            # Python lists per group
      .map(np.asarray)      # convert each to a NumPy array
      .to_dict()
)

cell_inputs = torch.zeros((num_stim, num_time, num_neurons), dtype=dtype, device=device)
for swi, sw in enumerate(sine_waves):
    for col_ind, col in enumerate(l1_df.column_id):
        # each value in the sine wave is assigned to a specific column in the eye
        # for each column, stimulate the L1, L2, and L3 cells
        cell_inputs[swi, :, col_to_cell_ind[col]] = -sw[col_ind, :, None]

# --- train the model ---
opt = torch.optim.Adam(list(model.parameters()) + [readout_weight], lr=alpha)
mask = torch.as_tensor(cell_info_df.type == output_neuron, device=device)
target = torch.as_tensor(1e-3 * temporal_frequencies, device=device, dtype=dtype)

for step in range(num_steps):
    start = time.time()
    opt.zero_grad()

    # forward
    model.update_dynamics_matrix()
    outputs = mu.run_rnn(model, cell_inputs)            # (batch, T, n)
    readout_val = outputs[:, :, mask].mean(dim=(1, 2))  # (batch,)
    pred = readout_weight * readout_val

    # loss
    loss = torch.nn.functional.mse_loss(pred, target)

    # backward + update
    loss.backward()
    opt.step()

    if step % 1 == 0:
        print(f"step {step + 1} / {num_steps}: loss = {loss.item():.4f}")

    print('epoch took', time.time() - start, 's')


plt.figure()
plt.plot(pred.detach().cpu().numpy())
plt.plot(target.detach().cpu().numpy())
plt.show()

a=1