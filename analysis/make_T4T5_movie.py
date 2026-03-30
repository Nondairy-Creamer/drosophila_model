import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
import numpy as np
import pandas as pd
import shutil
import neural_models as nm
import data_utilities as du
import plotting_utilities as pu

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
from analysis_utilities import interp_nan_column
import sys

if len(sys.argv) == 1:
    model_folder = Path('models/best_sine')
else:
    model_folder = Path(sys.argv[1])

model_config = du.load_config(model_folder / 'config.toml')
fig_folder = Path(model_folder / 'movie')
dtype = getattr(torch, model_config['model']['dtype'])
connectome_folder = Path(model_config['data']['connectome_folder'])
model = nm.NeuralModel.load(model_folder / 'model.pt')
device = model.device

frame_rate = int(1 / model_config['model']['dt'])
color_percentile = 99
concat_all = False

# --- load the sine wave stimuli ---
# directions = ['right', 'left', 'up', 'down']
directions = ['right_left', 'up_down']
stimulus_in = torch.load(model_folder / 'cell_inputs.pt', map_location=device).to(model.device)

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
# relevant_neurons = mu.path_from_a_to_b(model, ['L1', 'L2', 'L3'], ['T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d'])
# relevant_neurons = model.unique_cell_types
neurons_to_silence = []
model.silence_neurons(neurons_to_silence, invert_silencing=False)

# --- feed the stimulus into the model ---
input_neurons = ['L1', 'L2', 'L3']
output_neurons = ['L1', 'Mi1', 'Mi9', 'C3', 'L2', 'Tm1', 'Tm2', 'Tm4', 'Tm9', 'T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d']

with torch.no_grad():
    pred = model.predict(stimulus, input_neurons=input_neurons, output_neurons=output_neurons)['outputs']
    pred = pred['firing_rate']
shutil.rmtree(fig_folder, ignore_errors=True)
fig_folder.mkdir(parents=True, exist_ok=True)

xyz = model.get_column_xyz()
cell_x = -xyz[:, 0]
cell_z = xyz[:, 1]
cell_phi = model.column_phi_theta[:, 0]
cell_theta = model.column_phi_theta[:, 1]
num_stim, num_time, num_columns = stimulus.shape
num_neurons = model.num_neurons

# --- Diamond sequence movie for T4 ---
measured_cell_types = ['T4a', 'T4b', 'T4c', 'T4d']

# Build responses dictionary limited to measured cell types, and add stimulus
seq_data = {}
for mct in measured_cell_types:
    cell_resp = pred[mct].detach().cpu().numpy()
    cell_resp /= np.nanpercentile(cell_resp, color_percentile)
    cell_resp = interp_nan_column(cell_resp, cell_phi, cell_theta)
    seq_data[mct] = cell_resp  # [num_stimuli, time, columns]

seq_data['stimulus'] = stimulus.detach().cpu().numpy()

init_scatter_diamond = make_scatter_diamond_sequence_factory(seq_data, cell_x, cell_z, scale=1.8, size_scale=0.6)
make_movie(init_scatter_diamond, ['sequence'], fig_folder, frame_rate, save_name="t4")


# --- Diamond sequence movie for T5 ---
measured_cell_types = ['T5a', 'T5b', 'T5c', 'T5d']

seq_data = {}
for mct in measured_cell_types:
    cell_resp = pred[mct].detach().cpu().numpy()
    cell_resp /= np.nanpercentile(cell_resp, color_percentile)
    cell_resp = interp_nan_column(cell_resp, cell_phi, cell_theta)
    seq_data[mct] = cell_resp  # [num_stimuli, time, columns]

seq_data['stimulus'] = stimulus.detach().cpu().numpy()

init_scatter_diamond = make_scatter_diamond_sequence_factory(seq_data, cell_x, cell_z, scale=1.8, size_scale=0.6)
make_movie(init_scatter_diamond, ['sequence'], fig_folder, frame_rate, save_name="t5")
