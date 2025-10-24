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

# Generate temporal frequencies
temporal_frequencies = list(np.arange(
    config['stimulus']['temporal_freq_start'],
    config['stimulus']['temporal_freq_end'],
    config['stimulus']['temporal_freq_step'],
))

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
column_ids_input = [cell_info_df[cell_info_df.type == i].column_id.to_numpy() for i in input_neurons]
column_ids_input = [i[~np.isnan(i)] for i in column_ids_input]
column_ids_input = reduce(np.intersect1d, column_ids_input)

# get the column id for each column that has all of the neurons in output_neurons AND input neurons so we dont read outputs from a bad column as often
column_ids_output = [cell_info_df[cell_info_df.type == i].column_id.to_numpy() for i in (output_neurons + input_neurons)]
column_ids_output = [i[~np.isnan(i)] for i in column_ids_output]
column_ids_output = reduce(np.intersect1d, column_ids_output)

# create a dataframe that has our column info for the input neurons
# we just need to get the info from cell_info_df but restrict it to the column ids above and only 1 cell in that column since all columns with the same ID have
# the same x, y, z position
column_bool = (cell_info_df.type == input_neurons[0]) & cell_info_df.column_id.isin(column_ids_input)
column_info_input_df = cell_info_df[column_bool][['column_id', 'phi', 'theta', 'x', 'y', 'z']]
column_info_input_df = column_info_input_df.sort_values(by='column_id').reset_index(drop=True)  # for clarity, make sure its always with increasing column_ids

# and output neurons
column_bool = (cell_info_df.type == output_neurons[0]) & cell_info_df.column_id.isin(column_ids_output)
column_info_output_df = cell_info_df[column_bool][['column_id', 'phi', 'theta', 'x', 'y', 'z']]
column_info_output_df = column_info_output_df.sort_values(by='column_id').reset_index(drop=True)  # for clarity, make sure its always with increasing column_ids

# get the index in the model for the output neurons and in the chosen columns
output_indexes = []
correct_columns_bool = cell_info_df.column_id.isin(column_ids_output)
for i in output_neurons:
    this_cell_and_column = (cell_info_df.type == i) & correct_columns_bool
    output_indexes.append(np.where(this_cell_and_column)[0])

output_neurons_dict = {'right_left_light': output_neurons[0:2],
                       'up_down_light': output_neurons[2:4],
                       'right_left_dark': output_neurons[4:6],
                       'up_down_dark': output_neurons[6:8],
                       }

output_indexes_dict = {'right_left_light': output_indexes[0:2],
                       'up_down_light': output_indexes[2:4],
                       'right_left_dark': output_indexes[4:6],
                       'up_down_dark': output_indexes[6:8],
                       }

# create the model
model = nm.NeuralRnn(synapse_counts_coo=synapse_counts_df[['pre_matrix_ind', 'post_matrix_ind', 'synapse_counts']].to_numpy(), 
                     cell_types=cell_info_df.type.to_numpy(), nonlinearity=config['model']['nonlinearity'],
                     device=config['model']['device'], dtype=config['model']['dtype'],
                     use_sparse=config['model']['use_sparse'], scale_in_out=config['model']['scale_in_out'],
                     readout_neurons=output_neurons_dict, readout_indexes=output_indexes_dict)

# --- get a sign wave stimulus ---
# generate a sine wave on a cylinder around the fly
sine_waves, sine_wave_tfs, contrast_scaling_func = mu.create_sine_wave(column_info_input_df.phi.to_numpy(), column_info_input_df.theta.to_numpy(), 
    temporal_frequency=temporal_frequencies, dt=dt, max_time=max_time)
sine_waves = torch.tensor(sine_waves, device=device, dtype=dtype)

# map the sine waves to the cells
cell_inputs = mu.map_stimulus_to_cells(sine_waves, cell_info_df, column_info_input_df, input_neurons=input_neurons)

# --- plot the stimuli ---
pu.plot_response_over_space(sine_waves.detach().cpu().numpy()[:, 0, :], column_info_input_df.x, column_info_input_df.z, save_folder='figures/stimulus')

# Define the target as the temporal frequencies. but set up / down to zero for left/right cells
# do set right / left to zero for up/down cells
# target shape should (be stimuli / batch size, readout types, neurons)
num_readouts = len(output_neurons_dict)
num_stimuli = cell_inputs.shape[0]
readout_size = column_ids_output.shape[0]
target = torch.zeros((num_readouts, num_stimuli, readout_size), device=device, dtype=dtype)
target[:] = torch.tensor(sine_wave_tfs, device=device, dtype=dtype)[None, :, None]
target[0, target.shape[0] // 2:, :] = 0
target[1, :target.shape[0] // 2, :] = 0
target[2, target.shape[0] // 2:, :] = 0
target[3, :target.shape[0] // 2, :] = 0

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
               readout_xy=column_info_output_df[['x', 'z']].to_numpy())

