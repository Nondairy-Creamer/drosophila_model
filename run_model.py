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
import stimulus_utilities as su


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
dt = config['model']['dt']
stim_time = config['stimulus']['stim_time']  # seconds the sine is moving
pre_stim_time = config['stimulus']['pre_stim_time']  # seconds sine is stationary before the sine wave starts moving
post_stim_time = config['stimulus']['post_stim_time']  # seconds sine is stationary after the sine wave finishes moving
time_points = torch.arange(0, pre_stim_time + stim_time + post_stim_time, step=dt)
num_time = len(time_points)
temporal_frequencies = config['stimulus']['temporal_frequencies']

# --- load in connectivity and cell info data ---
connectome_folder = Path(config['data']['connectome_folder'])
cell_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_cell_info.parquet')
column_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_column_info.parquet')
synapse_counts_df = pd.read_parquet(connectome_folder / 'optic_lobes_synapse_counts.parquet')
num_neurons = len(cell_info_df)
num_columns = len(column_info_df)

# first generate the information about the columns - their location and ID
# we'll use the columns that have all our input and output neurons
input_neurons = ['L1', 'L2', 'L3']  # Convert list to tuple for compatibility
output_neurons = ['T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d']
# output_neurons_direction = ['right', 'left', 'up', 'down', 'right', 'left', 'up', 'down']
output_neurons_direction = ['right_left', 'right_left', 'up_down', 'up_down', 'right_left', 'right_left', 'up_down', 'up_down']

# create the model
model = nm.NeuralRnn(synapse_counts_coo=synapse_counts_df[['pre_matrix_ind', 'post_matrix_ind', 'synapse_counts']].to_numpy(), 
                     column_indices=cell_info_df['column_index'].to_numpy(), cell_types=cell_info_df['type'].to_numpy(), 
                     nonlinearity=config['model']['nonlinearity'],
                     device=config['model']['device'], dtype=config['model']['dtype'],
                     use_layer_norm=config['model']['use_layer_norm'], dt=dt)

# # --- get a sign wave stimulus ---
# # generate a sine wave on a cylinder around the fly
# sine_waves, sine_wave_tfs, contrast_scaling_factor, blocks = su.create_sine_wave(column_info_df.phi.to_numpy(), column_info_df.theta.to_numpy(), 
#     temporal_frequency=temporal_frequencies, dt=dt, stim_time=stim_time, pre_stim_time=pre_stim_time, post_stim_time=post_stim_time)
# sine_waves = torch.tensor(sine_waves, device=device, dtype=dtype)
# sine_wave_tfs = torch.tensor(sine_wave_tfs, device=device, dtype=dtype)

# --- generate a sine wave with random tf over time
sine_waves, sine_wave_tfs, contrast_scaling_factor, blocks = su.create_random_tf_sine_wave(
    column_info_df.phi.to_numpy(), column_info_df.theta.to_numpy(),
    tf_amp=config['stimulus']['tf_amp'], tf_time_scale=config['stimulus']['tf_time_scale'], 
    num_repeats=config['stimulus']['num_repeats'], dt=dt, stim_time=stim_time)
sine_waves = torch.tensor(sine_waves, device=device, dtype=dtype)
sine_wave_tfs = torch.tensor(sine_wave_tfs, device=device, dtype=dtype)

# --- plot the stimuli ---
pu.xt_plot(sine_waves.detach().cpu().numpy(), column_info_df['phi'].to_numpy(), target=sine_wave_tfs.detach().cpu().numpy(), save_folder='figures/stimulus')

# specify the target responses you expect for each cell type. this is what we will train on
num_stimuli = sine_waves.shape[0]
target = {}

# for i in range(len(output_neurons)):
#     this_neuron = output_neurons[i]
#     this_block = output_neurons_direction[i]
#     this_block_index = blocks.index(this_block)
#     num_blocks = len(blocks)
#     stimulus_block_size = num_stimuli // num_blocks  # limits this cell preference to one of the right left up down blocks

#     # get the index of the stimuli that this neruon will respond to, based on the direction
#     stim_resp_inds = torch.arange(this_block_index * stimulus_block_size, (this_block_index + 1) * stimulus_block_size)

#     target[this_neuron] = torch.zeros((num_stimuli, num_time, num_columns), device=device, dtype=dtype)
#     start_index = int(pre_stim_time / dt)
#     stop_index = int((pre_stim_time + stim_time) / dt)
#     target[this_neuron][stim_resp_inds, start_index:stop_index, :] = sine_wave_tfs[stim_resp_inds, None, None]

# --- make stimulus for random velocity sine wave ---
for i in range(len(output_neurons)):
    this_neuron = output_neurons[i]
    this_block = output_neurons_direction[i]
    this_block_index = blocks.index(this_block)
    num_blocks = len(blocks)
    stimulus_block_size = num_stimuli // num_blocks  # limits this cell preference to one of the right_left or up_down blocks

    # get the index of the stimuli that this neruon will respond to, based on the direction
    stim_resp_inds = torch.arange(this_block_index * stimulus_block_size, (this_block_index + 1) * stimulus_block_size)

    # limit the left-right neurons to only respond to the left-right stimulus
    # same for up-down to up-down stimulus. respond 0 otherwise
    target[this_neuron] = torch.zeros_like(sine_wave_tfs)
    target[this_neuron][stim_resp_inds, :, :] = sine_wave_tfs[stim_resp_inds, :, :]
    
    if this_neuron in ['T4a', 'T5a', 'T4c', 'T5c']:
        # front-to-back and up responding neurons respond positively to positive TFs in their preferred direction
        target[this_neuron][target[this_neuron] < 0] = 0
    elif this_neuron in ['T4b', 'T5b', 'T4d', 'T5d']:
        # back-to-front and down respond neurons respond positively to negative TFs in their preferred direction
        target[this_neuron] = -target[this_neuron]
        target[this_neuron][target[this_neuron] < 0] = 0

# save the config file and inputs
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_save_folder = Path('models/' + timestamp)
model_save_folder.mkdir(parents=True, exist_ok=True)
du.save_config(config, model_save_folder / 'config.toml')
torch.save(sine_waves.detach().cpu(), model_save_folder / "cell_inputs.pt")
torch.save({k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in target.items()}, model_save_folder / "target.pt")

contrast_scaling_factor = torch.tensor(contrast_scaling_factor, dtype=dtype, device=device)
torch.save(contrast_scaling_factor.detach().cpu(), model_save_folder / "contrast_scaling_factor.pt")

# get the loss function
loss_fn = getattr(tu, config['training']['loss_fn'])
loss_scaled = tu.get_scaled_loss_fn(loss_fn, contrast_scaling_factor)

model.save(model_save_folder / 'model_init.pt')
model.save(model_save_folder / 'model.pt')

# --- train the model ---
tu.train_model(model, sine_waves, target, output_neurons, input_neurons,
    model_save_folder=model_save_folder,
    num_steps=config['training']['num_steps'],
    save_every=config['training']['save_every'], 
    alpha=config['training']['alpha'], 
    clip_grad_norm=config['training']['clip_norm'],
    reset_init_every=config['training']['reset_init_every'],
    sigma_lambda=config['training']['sigma_lambda'],
    loss_fn=loss_scaled)

