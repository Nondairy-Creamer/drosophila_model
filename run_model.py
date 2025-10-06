import model_utilities as mu
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import neural_models as nm
import time
import data_utilities as du

# Load configuration from TOML file
config = du.load_config()

# Extract parameters from config
device = torch.device(config["model"]["device"])
dtype = getattr(torch, config["model"]["dtype"])
dt = config["simulation"]["dt"]
max_time = config["simulation"]["max_time"]
num_steps = config["simulation"]["num_steps"]
alpha = config["simulation"]["alpha"]
normalize_rows = config["data"]["normalize_rows"]
time_points = torch.arange(0, max_time, step=dt)
num_time = len(time_points)

# Generate temporal frequencies
temporal_frequencies = list(np.arange(
    config["stimulus"]["temporal_freq_start"],
    config["stimulus"]["temporal_freq_end"],
    config["stimulus"]["temporal_freq_step"],
))
nonlinearity = getattr(torch, config["model"]["nonlinearity"])

input_neurons = config["neurons"]["input_neurons"]  # Convert list to tuple for compatibility
output_neuron = config["neurons"]["output_neuron"]
use_sparse = config["model"]["use_sparse"]

# --- load in connectivity and cell info data ---
connectome_folder = Path(config["data"]["connectome_folder"])
column_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_column_info.parquet')
cell_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_cell_info.parquet')
synapse_counts_df = pd.read_parquet(connectome_folder / 'optic_lobes_synapse_counts.parquet')
num_neurons = len(cell_info_df.index)

# --- define the parameters and get the sparse connectivity matrix ---
if use_sparse:
    synapse_counts = mu.create_sparse_matrix(synapse_counts_df, device=device, dtype=dtype, normalize_rows=normalize_rows)
else:
    synapse_counts = torch.zeros((num_neurons, num_neurons), device=device, dtype=dtype)
    synapse_counts_values = torch.tensor(synapse_counts_df['synapse_counts'].values, device=device, dtype=dtype)
    synapse_counts[synapse_counts_df['post_matrix_ind'], synapse_counts_df['pre_matrix_ind']] = synapse_counts_values
readout_weight = torch.nn.Parameter(torch.tensor(config["simulation"]["readout_weight_init"], dtype=dtype, device=device))
readout_bias = torch.nn.Parameter(torch.tensor(config["simulation"]["readout_bias_init"], dtype=dtype, device=device))

# create the model
model = nm.SimpleRnn(synapse_counts=synapse_counts, device=device, dtype=dtype, use_sparse=use_sparse)

# --- get a sign wave stimulus ---
# generate a sine wave on a cylinder around the fly
l1_df = column_info_df[column_info_df['type'] == 'L1']

sine_waves = mu.create_sine_wave(l1_df.phi.to_numpy(), l1_df.theta.to_numpy(), 
    temporal_frequency=temporal_frequencies, dt=dt, max_time=max_time)

# --- define the external inputs to cells ---
# in this case we will directly stimulate L1, L2, and L3 with negative contrast
# we aren't stimulating receptors because they have a complicated relationship between location and viewing angle
# we are using negative contrast because L1-3 are inhibited by light

# find a mapping from column ID to neuron index
# column_info_df has the column ID for the neurons. However, not all neurons are included in column_info_df
# so we need to get the root_id for each L1-3 neuron for each column. The, we need to convert those root_ids
# to the neuron_index in the larger cell_info_df.

cell_inputs = []
lam_column_bool = column_info_df['type'].isin(input_neurons)
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

cell_inputs = []
for direction in ['right', 'left', 'up', 'down']:
    num_waves = len(sine_waves[direction])
    cell_inputs.append(torch.zeros((num_waves, num_time, num_neurons), dtype=dtype, device=device))

    for swi, sw in enumerate(sine_waves[direction]):
        for col_ind, col in enumerate(l1_df.column_id):
            # each value in the sine wave is assigned to a specific column in the eye
            # for each column, stimulate the L1, L2, and L3 cells
            this_column_sine_wave = torch.tensor(-sw[col_ind, :, None], dtype=dtype, device=device)
            cell_inputs[-1][swi, :, col_to_cell_ind[col]] = this_column_sine_wave

cell_inputs = torch.concat(cell_inputs, dim=0)
num_stim = cell_inputs.shape[0]

# --- train the model ---
opt = torch.optim.SGD(list(model.parameters()) + [readout_weight, readout_bias], lr=alpha)
# mask = torch.as_tensor(cell_info_df.type == output_neuron, device=device)
mask1 = torch.as_tensor(cell_info_df.type == 'T4a', device=device)
mask2 = torch.as_tensor(cell_info_df.type == 'T4b', device=device)
target = torch.zeros(num_stim, device=device, dtype=dtype)
target[0:len(temporal_frequencies)] = torch.tensor([i * 1e-5 for i in temporal_frequencies], device=device, dtype=dtype)
target[len(temporal_frequencies):2*len(temporal_frequencies)] = torch.tensor([-i * 1e-5 for i in temporal_frequencies], device=device, dtype=dtype)

for step in range(num_steps):
    start = time.time()
    opt.zero_grad()

    # forward
    model.update_dynamics_matrix()
    outputs = mu.run_rnn(model, cell_inputs)            # (batch, T, n)
    readout_val1 = outputs[:, :, mask1].mean(dim=(1, 2))  # (batch,)
    readout_val2 = outputs[:, :, mask2].mean(dim=(1, 2))  # (batch,)
    pred = readout_weight * (readout_val1 - readout_val2) + readout_bias

    # loss
    loss = torch.nn.functional.mse_loss(pred, target)

    # backward + update
    loss.backward()
    opt.step()

    if step % 1 == 0:
        print(f"step {step + 1} / {num_steps}: loss = {1e6 * loss.item():.4f}")

    print('epoch took', time.time() - start, 's')

    plt.figure()
    plt.plot(readout_val1.detach().cpu().numpy(), label='readout_val1')
    plt.plot(readout_val2.detach().cpu().numpy(), label='readout_val2')
    plt.legend()
    plt.savefig('figures/readout_val1_vs_readout_val2_' + str(step) + '.png')
    plt.close()

    plt.figure()
    plt.plot(pred.detach().cpu().numpy(), label='prediction')
    plt.plot(target.detach().cpu().numpy(), label='target')
    plt.legend()
    plt.savefig('figures/pred_vs_target_' + str(step) + '.png')
    plt.close()

a=1

