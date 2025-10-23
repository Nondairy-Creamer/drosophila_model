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
from functools import reduce

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

# our loss function will compare cells at every point in space. To do this, we need a paired set of vectors
# that gives us the index in cell_info_df belonging to each neuron
readout_neurons = ['T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d']

column_ids = [column_info_df[column_info_df.type == i].column_id.to_numpy() for i in readout_neurons]
column_ids = reduce(np.intersect1d, column_ids)
readout_inds = np.zeros((len(column_ids), len(readout_neurons))) - 1

# get the column locations in space for plotting
_t4a_rows = column_info_df[column_info_df.type == readout_neurons[0]]
_t4a_order = pd.Index(_t4a_rows.column_id).get_indexer(column_ids)
readout_column_xy = _t4a_rows.iloc[_t4a_order][['x', 'z']].to_numpy()

for ccti, cct in enumerate(readout_neurons):
    _subset = column_info_df[column_info_df.type == cct]
    _order = pd.Index(_subset.column_id).get_indexer(column_ids)
    assert not np.any(_order == -1), f'missing columns for {cct}'
    _sorted_root_ids = _subset.iloc[_order].root_id
    readout_inds[:, ccti] = pd.Index(cell_info_df.root_id).get_indexer(_sorted_root_ids)

assert ~np.any(readout_inds == -1), 'some readout inds not assigned'
readout_inds = torch.tensor(readout_inds, device=device, dtype=torch.int64)

readout_neurons_dict = {'right_left_light': readout_neurons[0:2],
                        'up_down_light': readout_neurons[2:4],
                        'right_left_dark': readout_neurons[4:6],
                        'up_down_dark': readout_neurons[6:8],
                        }

readout_inds_dict = {'right_left_light': readout_inds[:, 0:2],
                     'up_down_light': readout_inds[:, 2:4],
                     'right_left_dark': readout_inds[:, 4:6],
                     'up_down_dark': readout_inds[:, 6:8],
                     }

# create the model
model = nm.NeuralRnn(synapse_counts_coo=synapse_counts_df[['pre_matrix_ind', 'post_matrix_ind', 'synapse_counts']].to_numpy(), 
                     cell_types=cell_info_df.type.to_numpy(), nonlinearity=config['model']['nonlinearity'],
                     device=config['model']['device'], dtype=config['model']['dtype'],
                     use_sparse=config['model']['use_sparse'], scale_in_out=config['model']['scale_in_out'],
                     readout_neurons=readout_neurons_dict, readout_inds=readout_inds_dict)

# --- get a sign wave stimulus ---
# get the column's we're going to use based on which columns have L1 cells
l1_bool = column_info_df['type'] == 'L1'
l1_df = column_info_df[l1_bool]

# generate a sine wave on a cylinder around the fly
sine_waves, sine_wave_tfs, contrast_scaling_func = mu.create_sine_wave(l1_df.phi.to_numpy(), l1_df.theta.to_numpy(), 
    temporal_frequency=temporal_frequencies, dt=dt, max_time=max_time)
# we're going to put the sinewaves into L1, L2, L3 so invert contast
sine_waves = torch.tensor(sine_waves, device=device, dtype=dtype)
cell_inputs = mu.map_stimulus_to_cells(sine_waves, column_info_df, cell_info_df.root_id, l1_df.column_id, input_neurons=input_neurons)
# column_info_df got updated with the mapping to cell_info_df. save here
column_info_df.to_parquet(connectome_folder / 'optic_lobes_column_info.parquet', index=False)

# --- plot the stimuli ---
l1_df = column_info_df[l1_bool]  # this needs to be defined again since map_stimulus_to_cells added the root_ind column
stimuli_plot = [i[0, l1_df.root_ind.to_numpy()].detach().cpu().numpy() for i in cell_inputs]
pu.plot_response_over_space(stimuli_plot, l1_df.x, l1_df.z, save_folder='figures/stimulus')

# Define the target as the temporal frequencies. but set up / down to zero for left/right cells
# do set right / left to zero for up/down cells
# target shape should (be stimuli / batch size, readout types, neurons)
target = torch.zeros((len(sine_wave_tfs), readout_inds.shape[0], 4), device=device, dtype=dtype)
target[:] = torch.tensor(sine_wave_tfs, device=device, dtype=dtype)[:, None, None]
target[target.shape[0] // 2:, :, 0] = 0
target[:target.shape[0] // 2, :, 1] = 0
target[target.shape[0] // 2:, :, 2] = 0
target[:target.shape[0] // 2, :, 3] = 0

# save the config file and inputs
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_save_folder = Path('models/' + timestamp)
du.save_config(config, model_save_folder / 'config.toml')
torch.save(cell_inputs.detach().cpu(), model_save_folder / "cell_inputs.pt")
torch.save(target.detach().cpu(), model_save_folder / "target.pt")

# --- train the model ---
# find a good estimate of the readout weights
# model.estimate_readout_weights(cell_inputs, target)
tu.train_model(model, cell_inputs, target, model_save_folder=model_save_folder, 
               num_steps=config['training']['num_steps'], save_every=config['training']['save_every'], 
               alpha=config['training']['alpha'], use_bfgs=config['training']['use_bfgs'],
               readout_xy=readout_column_xy)

