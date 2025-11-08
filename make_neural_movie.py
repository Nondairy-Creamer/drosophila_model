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

model_folder = Path('models/20251104_203752_21Loss_SqrtInit_SpecNorm')

model_config = du.load_config(model_folder / 'config.toml')
computer_config = du.load_config()
device = torch.device(computer_config['model']['device'])  # what device you use depends on the computer, not how the model was run
dtype = getattr(torch, model_config['model']['dtype'])
model = nm.NeuralRnn.load(model_folder / 'model.pt', device=device)
cell_inputs = torch.load(model_folder / 'cell_inputs.pt')
stimulus = cell_inputs['stimulus'].to(device=model.device)
stim_to_cell_mapping = cell_inputs['mapping'].to(device=model.device)
target = torch.load(model_folder / 'target.pt').to(device=model.device)
frame_rate = int(1 / model_config['stimulus']['dt'])

# Load contrast scaling factor and create loss function
contrast_scaling_factor_path = model_folder / 'contrast_scaling_factor.pt'
contrast_scaling_factor = torch.load(contrast_scaling_factor_path).to(device=model.device)
loss_fn = getattr(tu, model_config['training']['loss_fn'])
loss_scaled = tu.get_scaled_loss(loss_fn, contrast_scaling_factor)

connectome_folder = Path(model_config['data']['connectome_folder'])
cell_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_cell_info.parquet')

# restrict the stimulus to 8 Hz, one phase
num_phase = 4
directions = ['right', 'left', 'up', 'down']
num_dir = len(directions)
num_vel = 3
stimulus = stimulus[::num_phase, :, :]
stimulus = stimulus[(num_vel - 1)::num_vel, :, :]

cells_to_plot = ['L1', 'T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d']
column_info_df, column_ids = du.get_column_df(cell_info_df, cells_to_plot)

pred, readout = model.predict_and_readout(stimulus, stim_to_cell_mapping)
pred = pred.detach().cpu().numpy()
fig_folder = Path(model_folder / 'movie')
shutil.rmtree(fig_folder, ignore_errors=True)
fig_folder.mkdir(parents=True, exist_ok=True)
columns_bool = cell_info_df.column_id.isin(column_ids)

# get the columns we need to interpolate over
column_location_names = ['column_id', 'theta', 'phi', 'x', 'y', 'z', 'x_flat', 'y_flat', 'p', 'q']
mask = cell_info_df['column_id'].notna()
all_columns_df = cell_info_df.loc[mask].drop_duplicates(subset=['column_id'], keep='first')
all_columns_df = all_columns_df[column_location_names]
pred_col_bool = all_columns_df['column_id'].isin(column_ids)
base_interp_col_bool = (~pred_col_bool).to_numpy()
all_columns_df.to_parquet('columns_df.parquet', index=False)

# find L1 cells that weren't stimulated and mark their columns for interpolation
with torch.no_grad():
    # active stimulus features across directions and time
    stim_active_per_feature = (stimulus != 0).any(dim=0).any(dim=0).detach().cpu().numpy()
    mapping_cell_idx = stim_to_cell_mapping[1].detach().cpu().numpy()
    mapping_stim_idx = stim_to_cell_mapping[0].detach().cpu().numpy()
    # OR reduce mapped stimulus activity per neuron
    has_nonzero_stim_per_neuron = np.zeros(cell_info_df.shape[0], dtype=bool)
    np.logical_or.at(has_nonzero_stim_per_neuron, mapping_cell_idx, stim_active_per_feature[mapping_stim_idx])
    # restrict to L1 neurons
    l1_mask_np = (cell_info_df.type == 'L1').to_numpy()
    l1_no_stim_neuron = l1_mask_np & (~has_nonzero_stim_per_neuron)
    # map those neurons to columns and set interp_col_bool true for those columns
    l1_zero_cols = cell_info_df.loc[l1_no_stim_neuron, 'column_id'].to_numpy()
    l1_extra_interp_bool = all_columns_df['column_id'].isin(l1_zero_cols).to_numpy()

interp_bool = base_interp_col_bool | l1_extra_interp_bool
cell_x = all_columns_df.x
cell_z = all_columns_df.z
num_stim, num_time, num_neurons = pred.shape
num_columns = len(all_columns_df)


def interpolate_missing_columns(responses):
    max_range = 10 * np.pi / 180 # distance in radians to average over
    responses_out = np.zeros_like(responses)

    for ii, interp_this_col in enumerate(interp_bool):
        if interp_this_col:
            columns_in_x_range = (np.abs(all_columns_df['phi'] - all_columns_df['phi'].iloc[ii]) < max_range).to_numpy()
            columns_in_y_range = (np.abs(all_columns_df['theta'] - all_columns_df['theta'].iloc[ii]) < max_range).to_numpy()
            column_is_not_0 = ~np.all(responses == 0, axis=(0, 1))  # numpy array
            columns_in_range = columns_in_x_range & columns_in_y_range & column_is_not_0
            columns_in_range[ii] = False  # exclude self

            if np.sum(columns_in_range) > 0:
                responses_out[:, :, ii] = np.mean(responses[:, :, columns_in_range], axis=2)
        else:
            responses_out[:, :, ii] = responses[:, :, ii]

    return responses_out

# Helper to place responses for a given cell type into the correct column order
def place_responses_by_column(pred, cell_info_df, all_columns_df, pred_col_bool, cell_mask, num_stim, num_time, num_columns):
    out = np.zeros((num_stim, num_time, num_columns), dtype=pred.dtype)
    mask_array = cell_mask.to_numpy()
    if not np.any(mask_array):
        return out
    subset_col_ids = cell_info_df.loc[mask_array, 'column_id'].to_numpy()
    target_indices = np.where(pred_col_bool)[0]
    target_col_ids = all_columns_df.loc[pred_col_bool, 'column_id'].to_numpy()
    indexer = pd.Index(subset_col_ids).get_indexer(target_col_ids)  # -1 when missing
    available = indexer >= 0
    if np.any(available):
        out[:, :, target_indices[available]] = pred[:, :, mask_array][:, :, indexer[available]]
    return out
# plot a movie of the following
# L1 activity
# T4a + T5a
# T4b + T5b
# T4c + T5c
# T4d + T5d
# vector of local motion (T4a + T5a - T4b - T5b, T4c + T5c - T4d - T5d)

data_out = {}
data_out['dt'] = model_config['stimulus']['dt']

# --- L1 activity ---
# Create boolean mask for cells in the full cell_info_df (for indexing predictions)
cell_to_plot_bool = (cell_info_df.type == 'L1') & columns_bool

# normalize
pred_plot_l1 = place_responses_by_column(
    pred, cell_info_df, all_columns_df, pred_col_bool, cell_to_plot_bool, num_stim, num_time, num_columns
)
# interpolate L1 using both base-missing columns and L1-unstimulated columns
pred_plot_l1 = interpolate_missing_columns(pred_plot_l1)
pred_plot_l1 /= np.percentile(pred_plot_l1, 90)  # handles negatives; use np.nanmax(pred_plot) if strictly ≥0

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
    # Create boolean mask for cells in the full cell_info_df (for indexing predictions)
    cell_to_plot_bool_0 = (cell_info_df.type == pval[0]) & columns_bool
    cell_to_plot_bool_1 = (cell_info_df.type == pval[1]) & columns_bool

    # normalize
    pred_plot_0 = place_responses_by_column(
        pred, cell_info_df, all_columns_df, pred_col_bool, cell_to_plot_bool_0, num_stim, num_time, num_columns
    )
    pred_plot_0 = interpolate_missing_columns(pred_plot_0)
    pred_plot_0 /= np.percentile(pred_plot_0, 90)  # handles negatives; use np.nanmax(pred_plot) if strictly ≥0
    pred_plot_0 = np.exp(pred_plot_0) / np.exp(1)

    pred_plot_1 = place_responses_by_column(
        pred, cell_info_df, all_columns_df, pred_col_bool, cell_to_plot_bool_1, num_stim, num_time, num_columns
    )
    pred_plot_1 = interpolate_missing_columns(pred_plot_1)
    pred_plot_1 /= np.percentile(pred_plot_1, 90)  # handles negatives; use np.nanmax(pred_plot) if strictly ≥0
    pred_plot_1 = np.exp(pred_plot_1) / np.exp(1)

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

np.savez_compressed("fly_model_responses.npz", **data_out)


