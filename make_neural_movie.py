import torch
import numpy as np
from pathlib import Path
import pandas as pd
import shutil
import neural_models as nm
import data_utilities as du
import plotting_utilities as pu
import training_utilities as tu
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

from movie_utilities import (
    make_movie,
    make_scatter_plot_factory,
    make_quiver_plot_factory,
    make_scatter_grid_factory,
    make_scatter_diamond_factory,
    make_scatter_diamond_sequence_factory,
    make_quiver_pair_sequence_factory
)
import model_utilities as mu
import sys

if len(sys.argv) == 1:
    model_folder = Path('models/20251125_171505')
else:
    model_folder = Path(sys.argv[1])

model_config = du.load_config(model_folder / 'config.toml')
computer_config = du.load_config()
fig_folder = Path(model_folder / 'movie')
device = torch.device(computer_config['model']['device'])  # what device you use depends on the computer, not how the model was run
dtype = getattr(torch, model_config['model']['dtype'])
connectome_folder = Path(model_config['data']['connectome_folder'])
model = nm.NeuralRnn.load(model_folder / 'model.pt', device=device)
_target_loaded = torch.load(model_folder / 'target.pt')
target = {k: (v.to(device=model.device, dtype=dtype) if torch.is_tensor(v) else v)
            for k, v in _target_loaded.items()}

frame_rate = int(1 / model_config['model']['dt'])
color_percentile = 99
concat_all = False

cell_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_cell_info.parquet')
column_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_column_info.parquet')

# --- load the sine wave stimuli ---
# directions = ['right', 'left', 'up', 'down']
directions = ['right_left', 'up_down']
stimulus_in = torch.load(model_folder / 'cell_inputs.pt').to(model.device)

# restrict the stimulus to one phase (select consecutive repeats, not every Nth)
num_phase = 4
num_dir = len(directions)
num_stim = 4  # number of consecutive repeats to use per direction

# split by direction first
stim_per_dir = stimulus_in.shape[0] // num_dir
stim_split = torch.split(stimulus_in, stim_per_dir, dim=0)

# within each direction, select first num_stim repeats from ONE phase (they are consecutive and continuous)
# structure per direction: [phase0 x num_repeats, phase1 x num_repeats, phase2 x num_repeats, phase3 x num_repeats]
num_repeats_per_phase = stim_per_dir // num_phase
stim_split = [s[:num_stim, :, :] for s in stim_split]  # take first num_stim from phase 0 (consecutive/continuous)
stimulus = torch.cat(stim_split, dim=0)

if concat_all:
    stimulus = torch.cat(torch.split(stimulus, 1, dim=0), dim=1)

# --- silence some neurons ---
neurons_to_silence = ['L1', 'L3']
neurons_to_silence = ['L5']
model.silence_neurons(neurons_to_silence)

# # --- load the stimuli from the camera ---
# directions = ['real_movie']
# stimulus = np.load(connectome_folder / 'fly_view3.npy')
# # encoding_mean = 255 / 2
# stimulus = np.log(stimulus)  # use logarithmic encoding?
# encoding_mean = np.log(255) / 2
# stimulus -= encoding_mean
# stimulus /= encoding_mean

# # Resample stimulus from original 20 Hz (dt=0.05s) to target dt using linear interpolation
# orig_dt = 1.0 / 20.0
# target_dt = float(model_config['model']['dt'])
# num_t0, num_neurons = stimulus.shape
# t0 = np.arange(num_t0, dtype=np.float64) * orig_dt
# t_end = t0[-1]
# num_t1 = int(np.floor(t_end / target_dt)) + 1
# t1 = np.arange(num_t1, dtype=np.float64) * target_dt
# stimulus_resampled = np.empty((num_t1, num_neurons), dtype=stimulus.dtype)
# for j in range(num_neurons):
#     stimulus_resampled[:, j] = np.interp(t1, t0, stimulus[:, j])
# stimulus = stimulus_resampled
# stimulus = stimulus[None, :, :]
# stimulus = torch.tensor(stimulus, dtype=dtype, device=model.device)

# --- feed the stimulus into the model ---
input_neurons = ['L1', 'L2', 'L3']
output_neurons = ['L1', 'Mi1', 'Mi9', 'C3', 'L2', 'Tm1', 'Tm2', 'Tm4', 'Tm9', 'T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d']

print(model.calc_sigma_max())

with torch.no_grad():
    pred = model.predict(stimulus, input_neurons=input_neurons, output_neurons=output_neurons)[0]
    pred = pred['firing_rate']
shutil.rmtree(fig_folder, ignore_errors=True)
fig_folder.mkdir(parents=True, exist_ok=True)
column_info_df.to_parquet(fig_folder / 'column_info_df.parquet', index=False)

cell_x = -column_info_df.x.to_numpy()
cell_z = column_info_df.z.to_numpy()
cell_theta = column_info_df.theta.to_numpy()
cell_phi = column_info_df.phi.to_numpy()
num_stim, num_time, num_columns = stimulus.shape
num_neurons = model.num_neurons

def interp_nan_column(time_by_neurons):
    if time_by_neurons.ndim == 2:
        time_by_neurons = time_by_neurons[None, :, :]
        
    num_neurons = time_by_neurons.shape[2]
    distance = 10 * np.pi / 180
    nan_neurons = np.any(np.isnan(time_by_neurons), axis=(0, 1))

    for i in range(num_neurons):
        if nan_neurons[i]:
            # find nearby neurons
            close_neurons_x = np.abs(cell_phi - cell_phi[i]) < distance
            close_neurons_y = np.abs(cell_theta - cell_theta[i]) < distance
            close_neurons = close_neurons_x & close_neurons_y & ~nan_neurons

            time_by_neurons[:, :, i] = np.mean(time_by_neurons[:, :, close_neurons], axis=-1)

    return time_by_neurons

column_info_df.to_parquet(fig_folder / 'column_info_df.parquet', index=False)

# --- Diamond sequence movie for T4 ---
measured_cell_types = ['T4a', 'T4b', 'T4c', 'T4d']

# Build responses dictionary limited to measured cell types, and add stimulus
seq_data = {}
for mct in measured_cell_types:
    cell_resp = pred[mct].detach().cpu().numpy()
    cell_resp /= np.nanpercentile(cell_resp, color_percentile)
    cell_resp = (np.exp(cell_resp) - 1) / (np.exp(1) - 1)
    cell_resp = interp_nan_column(cell_resp)
    seq_data[mct] = cell_resp  # [num_stimuli, time, columns]

seq_data['stimulus'] = stimulus.detach().cpu().numpy() / 2 + 0.5  # [num_stimuli, time, columns]

init_scatter_diamond = make_scatter_diamond_sequence_factory(seq_data, cell_x, cell_z, scale=1.8, size_scale=0.6)
make_movie(init_scatter_diamond, ['sequence'], fig_folder, frame_rate, save_name="t4")


# --- Diamond sequence movie for T5 ---
measured_cell_types = ['T5a', 'T5b', 'T5c', 'T5d']

seq_data = {}
for mct in measured_cell_types:
    cell_resp = pred[mct].detach().cpu().numpy()
    cell_resp /= np.nanpercentile(cell_resp, color_percentile)
    cell_resp = (np.exp(cell_resp) - 1) / (np.exp(1) - 1)
    cell_resp = interp_nan_column(cell_resp)
    seq_data[mct] = cell_resp  # [num_stimuli, time, columns]

seq_data['stimulus'] = stimulus.detach().cpu().numpy() / 2 + 0.5  # [num_stimuli, time, columns]

init_scatter_diamond = make_scatter_diamond_sequence_factory(seq_data, cell_x, cell_z, scale=1.8, size_scale=0.6)
make_movie(init_scatter_diamond, ['sequence'], fig_folder, frame_rate, save_name="t5")

# init_quiver = make_quiver_pair_sequence_factory(seq_data, cell_x, cell_z, scale=1.8, arrow_scale=0.1)
# make_movie(init_quiver, ['sequence'], fig_folder, frame_rate, save_name="quiver")


# # --- activity for light pathway ---
# # Build responses dictionary limited to measured cell types, and add stimulus
# measured_cell_types = ['L1', 'Mi1', 'Mi9', 'C3']
# seq_data = {}
# for mct in measured_cell_types:
#     cell_resp = pred[mct].detach().cpu().numpy()
#     cell_resp /= np.nanpercentile(cell_resp, color_percentile)
#     cell_resp = (np.exp(cell_resp) - 1) / (np.exp(1) - 1)
#     cell_resp = interp_nan_column(cell_resp)
#     seq_data[mct] = cell_resp  # [num_stimuli, time, columns]

# seq_data['stimulus'] = stimulus.detach().cpu().numpy() / 2 + 0.5  # [num_stimuli, time, columns]

# init_scatter_diamond = make_scatter_diamond_sequence_factory(seq_data, cell_x, cell_z, scale=1.8, size_scale=0.6)
# make_movie(init_scatter_diamond, ['sequence'], fig_folder, frame_rate, save_name="light")

# # --- activity for dark pathway ---
# # Build responses dictionary limited to measured cell types, and add stimulus
# measured_cell_types = ['L2', 'Tm1', 'Tm4', 'Tm9']
# seq_data = {}
# for mct in measured_cell_types:
#     cell_resp = pred[mct].detach().cpu().numpy()
#     cell_resp /= np.nanpercentile(cell_resp, color_percentile)
#     cell_resp = (np.exp(cell_resp) - 1) / (np.exp(1) - 1)
#     cell_resp = interp_nan_column(cell_resp)
#     seq_data[mct] = cell_resp  # [num_stimuli, time, columns]

# seq_data['stimulus'] = stimulus.detach().cpu().numpy() / 2 + 0.5  # [num_stimuli, time, columns]

# init_scatter_diamond = make_scatter_diamond_sequence_factory(seq_data, cell_x, cell_z, scale=1.8, size_scale=0.6)
# make_movie(init_scatter_diamond, ['sequence'], fig_folder, frame_rate, save_name="dark")

