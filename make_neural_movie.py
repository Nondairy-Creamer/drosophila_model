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
import imageio.v3 as iio  # pip install imageio[ffmpeg]
from movie_utilities import (
    make_movie,
    make_scatter_plot_factory,
    make_quiver_plot_factory,
)
import model_utilities as mu

model_folder = Path('models/20251110_234900')
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

frame_rate = int(1 / model_config['stimulus']['dt'])
color_percentile = 90

cell_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_cell_info.parquet')
column_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_column_info.parquet')
column_info_df.to_parquet(fig_folder / 'column_info_df.parquet', index=False)

# --- load the sine wave stimuli ---
directions = ['right', 'left', 'up', 'down']
stimulus = torch.load(model_folder / 'cell_inputs.pt').to(model.device)

# restrict the stimulus to 8 Hz, one phase
num_phase = 4
num_dir = len(directions)
num_vel = 3
stimulus = stimulus[::num_phase, :, :]
stimulus = stimulus[(num_vel - 1)::num_vel, :, :]

# # --- load the stimuli from the camera ---
# directions = ['real_movie']
# stimulus = np.load(connectome_folder / 'fly_view1.npy')
# stimulus -= stimulus.mean(axis=0)
# stimulus /= 128
# # Resample stimulus from original 20 Hz (dt=0.05s) to target dt using linear interpolation
# orig_dt = 1.0 / 20.0
# target_dt = float(model_config['stimulus']['dt'])
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
# input_neurons = ['L1', 'L2', 'L3']
# column_info_input_df, column_input_ids = du.get_column_df(cell_info_df, input_neurons)
# stim_to_cell_mapping = mu.get_stim_to_cell_mapping(cell_info_df, column_info_input_df, input_neurons=input_neurons)
# stim_to_cell_mapping = torch.tensor(stim_to_cell_mapping, device=model.device)


input_neurons = ['L1', 'L2', 'L3']
output_neurons = ['L1', 'T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d']

pred = model.predict(stimulus, input_neurons=input_neurons, output_neurons=output_neurons)
shutil.rmtree(fig_folder, ignore_errors=True)
fig_folder.mkdir(parents=True, exist_ok=True)

cell_x = column_info_df.x.to_numpy()
cell_z = column_info_df.z.to_numpy()
cell_theta = column_info_df.theta.to_numpy()
cell_phi = column_info_df.phi.to_numpy()
num_stim, num_time, num_columns = stimulus.shape
num_neurons = model.num_neurons

data_out = {}
data_out['dt'] = model_config['stimulus']['dt']

def interp_nan_column(time_by_neurons):
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

init_scatter = make_scatter_plot_factory(pred_plot_l1, cell_x, cell_z)
make_movie(init_scatter, directions, fig_folder, frame_rate, save_name="L1")

# --- T4 and T5 pairings ---
pairs = {'right_cells': ['T4a', 'T5a'], 
         'left_cells': ['T4b', 'T5b'], 
         'up_cells': ['T4c', 'T5c'],
         'down_cells': ['T4d', 'T5d']}

pair_resp = {}

for pkey, pval in pairs.items():
    # normalize
    pred_plot_0 = pred[pval[0]].detach().cpu().numpy()
    pred_plot_0 /= np.nanpercentile(pred_plot_0, color_percentile)
    pred_plot_0 = np.exp(pred_plot_0) / np.exp(1)
    pred_plot_0 = interp_nan_column(pred_plot_0)

    pred_plot_1 = pred[pval[1]].detach().cpu().numpy()
    pred_plot_1 /= np.nanpercentile(pred_plot_1, color_percentile)
    pred_plot_1 = np.exp(pred_plot_1) / np.exp(1)
    pred_plot_1 = interp_nan_column(pred_plot_1)

    pair_resp[pkey] = (pred_plot_0 + pred_plot_1) / 2
    data_out[pkey] = (pred_plot_0 + pred_plot_1) / 2

    # init_scatter = make_scatter_plot_factory(pred_plot, cell_x, cell_z)
    # make_movie(init_scatter, directions, fig_folder, frame_rate, save_name=pval[0])

# --- Predicted direction ---
scaling = 0.2
eps = 1e-3

x_motion = scaling * (pair_resp['right_cells'] - pair_resp['left_cells']) / (pair_resp['right_cells'] + pair_resp['left_cells'] + eps)
z_motion = scaling * (pair_resp['up_cells'] - pair_resp['down_cells']) / (pair_resp['up_cells'] + pair_resp['down_cells'] + eps)

data_out['motion_vector'] = [x_motion, z_motion]

init_quiver = make_quiver_plot_factory(x_motion, z_motion, cell_x, cell_z)
make_movie(init_quiver, directions, fig_folder, frame_rate, save_name="flow")

np.savez_compressed(fig_folder / 'fly_model_responses.npz', **data_out)
column_info_df.to_parquet(fig_folder / 'column_info_df.parquet', index=False)


