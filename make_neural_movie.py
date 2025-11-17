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
)
import model_utilities as mu

model_folder = Path('models/20251113_213941_largeDrift')
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

frame_rate = int(1 / model_config['model']['dt'] / 2)
color_percentile = 90

cell_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_cell_info.parquet')
column_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_column_info.parquet')

# --- load the sine wave stimuli ---
# directions = ['right', 'left', 'up', 'down']
directions = ['right_left', 'up_down']
stimulus = torch.load(model_folder / 'cell_inputs.pt').to(model.device)

# restrict the stimulus to 8 Hz, one phase
num_phase = 4
num_dir = len(directions)
num_stim = 10
stimulus = stimulus[::num_phase, :, :]
# stimulus = stimulus[(num_stim - 1)::num_stim, :, :]
# stimulus = stimulus[(num_stim - 1)::num_stim, :, :]

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
output_neurons = ['L1', 'T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d']

pred = model.predict(stimulus, input_neurons=input_neurons, output_neurons=output_neurons)
shutil.rmtree(fig_folder, ignore_errors=True)
fig_folder.mkdir(parents=True, exist_ok=True)
column_info_df.to_parquet(fig_folder / 'column_info_df.parquet', index=False)

cell_x = -column_info_df.x.to_numpy()
cell_z = column_info_df.z.to_numpy()
cell_theta = column_info_df.theta.to_numpy()
cell_phi = column_info_df.phi.to_numpy()
num_stim, num_time, num_columns = stimulus.shape
num_neurons = model.num_neurons

data_out = {}
data_out['dt'] = model_config['model']['dt']

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

# --- L1 activity ---
# Create boolean mask for cells in the full cell_info_df (for indexing predictions)
pred_plot_l1 = pred['L1'].detach().cpu().numpy()
pred_plot_l1 /= np.nanpercentile(pred_plot_l1, color_percentile)
pred_plot = interp_nan_column(pred_plot_l1)

data_out['L1'] = pred_plot_l1
data_out['stimulus'] = stimulus.detach().cpu().numpy() / 2 + 0.5

# init_scatter = make_scatter_plot_factory(pred_plot_l1, cell_x, cell_z)
# make_movie(init_scatter, directions, fig_folder, frame_rate, save_name="L1")

# --- Diamond sequence movie for measured cell types + stimulus ---
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

np.savez_compressed(fig_folder / 'fly_model_responses.npz', **data_out)
column_info_df.to_parquet(fig_folder / 'column_info_df.parquet', index=False)


