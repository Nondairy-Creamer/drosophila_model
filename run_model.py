import numpy as np
import torch
import model_utilities as mu
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

use_bfgs = False
# Clear previous figures (delete non-hidden files)
fig_dir = Path('figures')
fig_dir.mkdir(parents=True, exist_ok=True)
for p in fig_dir.iterdir():
    if p.is_file() and not p.name.startswith('.'):
        p.unlink()

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

# create the model
model = nm.SimpleRnn(synapse_counts_df=synapse_counts_df, cell_types=cell_info_df.type.values, nonlinearity=nonlinearity,
 device=device, dtype=dtype, use_sparse=use_sparse)

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
root_id_to_column_id = column_info_df.loc[lam_column_bool, ['column_id', 'root_id']]
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
            # Expand the sine wave across all L1-3 cells in this column for deterministic CUDA assignment
            idx = torch.as_tensor(col_to_cell_ind[col], dtype=torch.long)
            k = idx.numel()
            this_column_sine_wave = torch.tensor(-sw[col_ind, :], dtype=dtype, device=device)  # (T,)
            this_column_sine_wave_expanded = this_column_sine_wave[:, None].expand(num_time, k)  # (T, K)
            cell_inputs[-1][swi, :, idx] = this_column_sine_wave_expanded

cell_inputs = torch.concat(cell_inputs, dim=0)
num_stim = cell_inputs.shape[0]

# --- train the model ---
# opt = torch.optim.Adam(model.parameters(), lr=alpha)
opt = torch.optim.LBFGS(model.parameters(), line_search_fn='strong_wolfe')
    
# Halve the learning rate every 400 iterations
target = torch.zeros(num_stim, device=device, dtype=dtype)
target[0 : len(temporal_frequencies)] = torch.tensor([i for i in temporal_frequencies], device=device, dtype=dtype)
target[len(temporal_frequencies) : 2 * len(temporal_frequencies)] = torch.tensor([-i for i in temporal_frequencies], device=device, dtype=dtype)

# find a good estimate of the readout weights
model.estimate_readout_weights(cell_inputs, target)
loss_list = []

for step in range(num_steps):
    def closure():
        opt.zero_grad(set_to_none=True)
        pred = model.predict_and_readout(cell_inputs)
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        return loss

    loss = opt.step(closure)

    # log once per outer step
    loss_list.append(loss.item())

    if step % config['simulation']['print_every'] == 0:
        print(f"step {step + 1} / {num_steps} | loss = {loss.item()}")

        with torch.no_grad():
            pred = model.predict_and_readout(cell_inputs)

            plt.figure()
            plt.plot(pred.detach().cpu().numpy(), label='prediction')
            plt.plot(target.detach().cpu().numpy(), label='target')
            plt.legend()
            plt.savefig('figures/pred_vs_target_' + str(step) + '.png')
            plt.close()

            plt.figure()
            plt.plot(loss_list, label='loss')
            plt.legend()
            plt.savefig('figures/loss.png')
            plt.close()

a=1

