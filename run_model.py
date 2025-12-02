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
output_neurons_direction = ['right', 'left', 'up', 'down', 'right', 'left', 'up', 'down']

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
# stimulus, optic_flow = su.create_random_tf_sine_wave(
#     column_info_df.phi.to_numpy(), column_info_df.theta.to_numpy(),
#     tf_amp=config['stimulus']['tf_amp'], tf_time_scale=config['stimulus']['tf_time_scale'], 
#     num_repeats=config['stimulus']['num_repeats'], dt=dt, stim_time=stim_time)
# stimulus = torch.tensor(stimulus, device=device, dtype=dtype)
# optic_flow = torch.tensor(optic_flow, device=device, dtype=dtype)

# --- grab the natural movie stimulus ---
stimulus, optic_flow = su.load_natural_movie()
stimulus = torch.tensor(stimulus, device=device, dtype=dtype)
optic_flow = torch.tensor(optic_flow, device=device, dtype=dtype)

# --- plot the stimuli ---
pu.xt_plot(stimulus.detach().cpu().numpy(), column_info_df['phi'].to_numpy(), target=optic_flow.detach().cpu().numpy(), save_folder='figures/stimulus')

# specify the target responses you expect for each cell type. this is what we will train on
num_stimuli = stimulus.shape[0]
target = {}

# --- make stimulus for random velocity sine wave ---
for n, d in zip(output_neurons, output_neurons_direction):
    if d == 'right':
        target[n] = optic_flow[:, :, :, 0]
        target[n][target[n] < 0] = 0

    elif d == 'left':
        target[n] = -optic_flow[:, :, :, 0]
        target[n][target[n] < 0] = 0

    elif d == 'up':
        target[n] = optic_flow[:, :, :, 1]
        target[n][target[n] < 0] = 0

    elif d == 'down':
        target[n] = -optic_flow[:, :, :, 1]
        target[n][target[n] < 0] = 0

# save the config file and inputs
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_save_folder = Path('models/' + timestamp)
model_save_folder.mkdir(parents=True, exist_ok=True)
du.save_config(config, model_save_folder / 'config.toml')
torch.save(stimulus.detach().cpu(), model_save_folder / "cell_inputs.pt")
torch.save({k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in target.items()}, model_save_folder / "target.pt")

# get the loss function
loss_fn = getattr(tu, config['training']['loss_fn'])
model.save(model_save_folder / 'model_init.pt')
model.save(model_save_folder / 'model.pt')

# --- train the model ---
tu.train_model(model, stimulus, target, output_neurons, input_neurons,
    model_save_folder=model_save_folder,
    num_steps=config['training']['num_steps'],
    save_every=config['training']['save_every'], 
    alpha=config['training']['alpha'], 
    clip_grad_norm=config['training']['clip_norm'],
    reset_init_every=config['training']['reset_init_every'],
    sigma_lambda=config['training']['sigma_lambda'],
    peak_lambda=config['training']['peak_lambda'],
    loss_fn=loss_fn)

