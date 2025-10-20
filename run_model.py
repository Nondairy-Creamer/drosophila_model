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
input_neurons = ['L1', 'L2', 'L3']  # Convert list to tuple for compatibility

# Generate temporal frequencies
temporal_frequencies = list(np.arange(
    config['stimulus']['temporal_freq_start'],
    config['stimulus']['temporal_freq_end'],
    config['stimulus']['temporal_freq_step'],
))

# --- load in connectivity and cell info data ---
connectome_folder = Path(config['data']['connectome_folder'])
column_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_column_info.parquet')
cell_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_cell_info.parquet')
synapse_counts_df = pd.read_parquet(connectome_folder / 'optic_lobes_synapse_counts.parquet')
num_neurons = len(cell_info_df.index)

# create the model
model = nm.NeuralRnn(synapse_counts_coo=synapse_counts_df[['pre_matrix_ind', 'post_matrix_ind', 'synapse_counts']].to_numpy(), 
                     cell_types=cell_info_df.type.to_numpy(), nonlinearity=config['model']['nonlinearity'],
                     device=config['model']['device'], dtype=config['model']['dtype'], readout_neurons=config['model']['readout_neurons'],
                     use_sparse=config['model']['use_sparse'], scale_in_out=config['model']['scale_in_out'])

# --- get a sign wave stimulus ---
# generate a sine wave on a cylinder around the fly
# get the column info for all L1s and use that to know the location and column ID for each ommatidia
l1_df = column_info_df[column_info_df['type'] == 'L1']

sine_waves, sine_wave_tfs = mu.create_sine_wave(l1_df.phi.to_numpy(), l1_df.theta.to_numpy(), 
    temporal_frequency=temporal_frequencies, dt=dt, max_time=max_time)
# we're going to put the sinewaves into L1, L2, L3 so invert contast
sine_waves = torch.tensor(sine_waves, device=device, dtype=dtype)
cell_inputs = mu.map_stimulus_to_cells(sine_waves, column_info_df, cell_info_df.root_id, l1_df.column_id, input_neurons=input_neurons)
# column_info_df got updated with the mapping to cell_info_df. save here
column_info_df.to_parquet(connectome_folder / 'optic_lobes_column_info.parquet', index=False)

# --- plot the stimuli ---
cell_x = l1_df.x
cell_z = l1_df.z
stim_fig_folder = Path('figures/stimulus')
stim_fig_folder.mkdir(parents=True, exist_ok=True)

for si, s in enumerate(cell_inputs):
    # just plot the first timepoint
    c = s[0, l1_df.root_ind.to_numpy()].detach().cpu().numpy()
    plt.figure()
    plt.scatter(cell_x, cell_z, c=c)
    plt.savefig(stim_fig_folder / ('stim_' + str(si) + '.png'))
    plt.close()


# Define the target as the temporal freuqencies. but set up / down to zero
target = torch.tensor(sine_wave_tfs, device=device, dtype=dtype)
target[target.shape[0] // 2:] = 0

# save the config file and inputs
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_save_folder = Path('models/' + timestamp)
du.save_config(config, model_save_folder / 'config.toml')
torch.save(cell_inputs.detach().cpu(), model_save_folder / "cell_inputs.pt")
torch.save(target.detach().cpu(), model_save_folder / "target.pt")

# --- train the model ---
# find a good estimate of the readout weights
model.estimate_readout_weights(cell_inputs, target)
tu.train_model(model, cell_inputs, target, model_save_folder=model_save_folder, 
               num_steps=config['training']['num_steps'], save_every=config['training']['save_every'], 
               alpha=config['training']['alpha'], use_bfgs=config['training']['use_bfgs'])