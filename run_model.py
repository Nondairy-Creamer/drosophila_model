import numpy as np
import torch
import model_utilities as mu
import pandas as pd
from pathlib import Path
import plotting_utilities as pu
import neural_models as nm
import time
import data_utilities as du
from datetime import datetime
import training_utilities as tu


# Load configuration from TOML file
config = du.load_config()

# Clear previous figures (delete non-hidden files)
fig_dir = Path('figures')
fig_dir.mkdir(parents=True, exist_ok=True)
for p in fig_dir.iterdir():
    if p.is_file() and not p.name.startswith('.'):
        p.unlink()

# Extract parameters from config
device = torch.device(config['model']['device'])
dtype = getattr(torch, config['model']['dtype'])
dt = config['stimulus']['dt']
max_time = config['stimulus']['max_time']
time_points = torch.arange(0, max_time, step=dt)
num_time = len(time_points)
temporal_frequencies = config['stimulus']['temporal_frequencies']

# --- load in connectivity and cell info data ---
connectome_folder = Path(config['data']['connectome_folder'])
cell_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_cell_info.parquet')
synapse_counts_df = pd.read_parquet(connectome_folder / 'optic_lobes_synapse_counts.parquet')
num_neurons = len(cell_info_df.index)

# first generate the information about the columns - their location and ID
# we'll use the columns that have all our input and output neurons
input_neurons = ['L1', 'L2', 'L3']  # Convert list to tuple for compatibility
output_neurons = ['T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d']

# get the column id for each column that has all of the neurons in input_neurons
column_info_input_df, column_ids_input = du.get_column_df(cell_info_df, input_neurons)
column_info_output_df, column_ids_output = du.get_column_df(cell_info_df, output_neurons)

# get the index in the model for the output neurons and in the chosen columns
readout_indexes = {}
correct_columns_bool = cell_info_df.column_id.isin(column_ids_output)
for i in output_neurons:
    this_cell_and_column = (cell_info_df.type == i) & correct_columns_bool
    readout_indexes[i] = np.where(this_cell_and_column)[0]

# create the model
model = nm.NeuralRnn(synapse_counts_coo=synapse_counts_df[['pre_matrix_ind', 'post_matrix_ind', 'synapse_counts']].to_numpy(), 
                     cell_types=cell_info_df.type.to_numpy(), nonlinearity=config['model']['nonlinearity'],
                     device=config['model']['device'], dtype=config['model']['dtype'],
                     use_sparse=config['model']['use_sparse'], scale_in_out=config['model']['scale_in_out'],
                     readout_indexes=readout_indexes)

# --- get a sign wave stimulus ---
# generate a sine wave on a cylinder around the fly
sine_waves, sine_wave_tfs, contrast_scaling_function = mu.create_sine_wave(column_info_input_df.phi.to_numpy(), column_info_input_df.theta.to_numpy(), 
    temporal_frequency=temporal_frequencies, dt=dt, max_time=max_time)
sine_waves = torch.tensor(sine_waves, device=device, dtype=dtype)
sine_wave_tfs = torch.tensor(sine_wave_tfs, device=device, dtype=dtype)

# map the sine waves to the cells
cell_inputs = mu.map_stimulus_to_cells(sine_waves, cell_info_df, column_info_input_df, input_neurons=input_neurons)

# --- plot the stimuli ---
pu.plot_response_over_space(sine_waves.detach().cpu().numpy()[:, 0, :], column_info_input_df.x, column_info_input_df.z, save_folder='figures/stimulus')

# specify the target responses you expect for each cell type. this is what we will train on
num_readouts = len(readout_indexes)
num_stimuli = cell_inputs.shape[0]
readout_size = column_ids_output.shape[0]
target = torch.zeros((num_readouts, num_stimuli, readout_size), device=device, dtype=dtype)

# stimuli are right left up down
# cell preferences are currently right left up down right left up down
num_stimulus_blocks = 4
stimulus_block_size = num_stimuli // num_stimulus_blocks  # limits this cell prefernce to one of the right left up down blocks
for i in range(num_stimulus_blocks):
    quart = torch.arange(i * stimulus_block_size, (i + 1) * stimulus_block_size)
    # both the i and i+4 cells respond to the same stimulus
    # T4a is rightward selective and T5a is as well
    # this code will change if you change the cell readouts
    target[i, quart, :] = torch.abs(sine_wave_tfs[:stimulus_block_size, None])  # just take block of TFs and only use them as positive
    target[i + 4, quart, :] = torch.abs(sine_wave_tfs[:stimulus_block_size, None])

# save the config file and inputs
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_save_folder = Path('models/' + timestamp)
model_save_folder.mkdir(parents=True, exist_ok=True)
du.save_config(config, model_save_folder / 'config.toml')
torch.save(cell_inputs.detach().cpu(), model_save_folder / "cell_inputs.pt")
torch.save(target.detach().cpu(), model_save_folder / "target.pt")

contrast_scaling_factor = contrast_scaling_function(column_info_output_df['theta'].to_numpy())
contrast_scaling_factor = torch.tensor(contrast_scaling_factor, dtype=dtype, device=device)
torch.save(contrast_scaling_factor.detach().cpu(), model_save_folder / "contrast_scaling_factor.pt")

def loss_scaled(y_hat, y):
    return tu.combined_loss(y_hat, y, neuron_scaling=contrast_scaling_factor, eps=1e-6)
    # return tu.cos_loss(y_hat, y, neuron_scaling=contrast_scaling_factor, eps=1e-6)

model.save(model_save_folder / 'model_init.pt')

# --- train the model ---
# find a good estimate of the readout weights
# model.estimate_readout_weights(cell_inputs, target)
tu.train_model(model, cell_inputs, target, model_save_folder=model_save_folder,  num_steps=config['training']['num_steps'], 
               save_every=config['training']['save_every'], alpha=config['training']['alpha'], clip_grad_norm=config['training']['clip_norm'],
               readout_xy=column_info_output_df[['x', 'z']].to_numpy(), loss_fn=loss_scaled)

