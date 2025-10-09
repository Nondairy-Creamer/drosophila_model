import numpy as np
import torch
import model_utilities as mu
import pandas as pd
from pathlib import Path
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
import neural_models as nm
import time
import data_utilities as du

# Load configuration from TOML file
config = du.load_config()

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
use_bfgs = config['simulation']['use_bfgs']

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

# we have a larger list of neuron ids for each neuron
# we have a subset of that list that says for each neuron Id what column it belongs to
# need to find the neuron_index in the larger list for each entry in the subset list
# then we need to find the index in the stimulus that belongs to each column.
# with those, we can assign the correct stimulus to each neuron
column_info_df['root_ind'] = pd.Index(cell_info_df.root_id).get_indexer(column_info_df.root_id)
column_info_df['column_ind'] = pd.Index(l1_df.column_id).get_indexer(column_info_df.column_id)

# get rid of any entries where we have column info but not cell info, or we have L2/L3 but not L1
column_info_df = column_info_df.query('root_ind != -1 and column_ind != -1')

# get the lam neurons
lam_column_bool = column_info_df['type'].isin(input_neurons)
root_id_to_column_id = column_info_df[lam_column_bool].copy()

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
    cell_inputs.append(np.zeros((num_waves, num_time, num_neurons)))

    for swi, sw in enumerate(sine_waves[direction]):
        # each value in the sine wave is assigned to a specific column in the eye
        # for each column, stimulate the L1, L2, and L3 cells
        cell_inputs[-1][swi, :, root_id_to_column_id.root_ind.values] = sw[root_id_to_column_id.column_ind.values, :]

cell_inputs = torch.tensor(np.concatenate(cell_inputs, axis=0), device=device, dtype=dtype)
num_stim = cell_inputs.shape[0]

# --- train the model ---
if use_bfgs:
    opt = torch.optim.LBFGS(model.parameters(), line_search_fn='strong_wolfe')
else:
    opt = torch.optim.Adam(model.parameters(), lr=alpha)
    
# Halve the learning rate every 400 iterations
target = torch.zeros(num_stim, device=device, dtype=dtype)
target[0 : len(temporal_frequencies)] = torch.tensor([i for i in temporal_frequencies], device=device, dtype=dtype)
target[len(temporal_frequencies) : 2 * len(temporal_frequencies)] = torch.tensor([-i for i in temporal_frequencies], device=device, dtype=dtype)

# find a good estimate of the readout weights
model.estimate_readout_weights(cell_inputs, target)
loss_list = []

for step in range(num_steps):
    def closure():
        opt.zero_grad()
        readout = model.predict_and_readout(cell_inputs)[1]
        loss = torch.nn.functional.mse_loss(readout, target)
        loss.backward()
        return loss

    if use_bfgs:
        loss = opt.step(closure)
    else:
        loss = closure()
        opt.step()

    # log once per outer step
    loss_list.append(loss.item())

    if step % config['simulation']['print_every'] == 0:
        print(f"step {step + 1} / {num_steps} | loss = {loss.item()}")

        with torch.no_grad():
            pred, readout = model.predict_and_readout(cell_inputs)

            if config['visualization']['show_plots']:
                cell_to_plot = 'L1'
                cell_to_plot_bool = column_info_df.type == cell_to_plot
                cell_x = column_info_df[cell_to_plot_bool].x
                cell_z = column_info_df[cell_to_plot_bool].z

                cell_activity = pred[0, 10, column_info_df[cell_to_plot_bool].root_ind.values]
                plt.figure()
                plt.scatter(cell_x, cell_z, c=cell_activity)
                plt.show()

                cell_activity = pred[6, 10, column_info_df[cell_to_plot_bool].root_ind.values]
                plt.figure()
                plt.scatter(cell_x, cell_z, c=cell_activity)
                plt.show()

            plt.figure()
            plt.plot(readout.detach().cpu().numpy(), label='prediction')
            plt.plot(target.detach().cpu().numpy(), label='target')
            plt.legend()
            plt.savefig('figures/pred_vs_target_' + str(step) + '.png')
            plt.close()

            plt.figure()
            plt.plot(loss_list, label='loss')
            plt.legend()
            plt.savefig('figures/loss.png')
            plt.close()

