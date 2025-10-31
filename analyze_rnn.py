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

model_folder = Path('models/20251031_153410')

config = du.load_config(model_folder / 'config.toml')
model = nm.NeuralRnn.load(model_folder / 'model_init.pt')
cell_inputs = torch.load(model_folder / 'cell_inputs.pt').to(device=model.device)
target = torch.load(model_folder / 'target.pt').to(device=model.device)

# Load contrast scaling factor and create loss function
contrast_scaling_factor_path = model_folder / 'contrast_scaling_factor.pt'
contrast_scaling_factor = torch.load(contrast_scaling_factor_path).to(device=model.device)
def loss_scaled(y_hat, y):
    return tu.combined_loss(y_hat, y, neuron_scaling=contrast_scaling_factor, eps=1e-6)

connectome_folder = Path(config['data']['connectome_folder'])
cell_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_cell_info.parquet')

cells_to_plot = ['L1', 'Mi1', 'Tm3', 'T4a', 'T4b', 'T4c', 'T4d']
column_info_df, column_ids = du.get_column_df(cell_info_df, cells_to_plot)

# restrict the columns to a line across the eye
vert_lim = 2.5 * np.pi / 180
within_vert_lim = np.abs(column_info_df['theta'] - np.pi / 2) < vert_lim
column_info_df = column_info_df[within_vert_lim]
column_ids = column_ids[within_vert_lim]
sort_inds = np.argsort(column_info_df['phi'])[::-1]

pred, readout = model.predict_and_readout(cell_inputs)
stim_fig_folder = Path('figures/analysis')
shutil.rmtree(stim_fig_folder, ignore_errors=True)
stim_fig_folder.mkdir(parents=True, exist_ok=True)
time_point = 20
these_columns = cell_info_df.column_id.isin(column_ids)

# get the correlation and cosine similarity
correlation_similarity = 1.0 - tu.corr_loss(readout, target).detach().cpu().numpy()
cosine_similarity = 1.0 - tu.cos_loss(readout, target).detach().cpu().numpy()

print('correlation:', correlation_similarity)
print('cosine similarity:', cosine_similarity)

for ctp in cells_to_plot:
    # Create boolean mask for cells in the full cell_info_df (for indexing predictions)
    cell_to_plot_bool = (cell_info_df.type == ctp) & these_columns
    
    # Filter column_info_df directly for this cell type (for spatial information)
    cell_to_plot_df = cell_info_df[cell_to_plot_bool]
    cell_x = cell_to_plot_df.x
    cell_z = cell_to_plot_df.z
    
    # Get sort indices for just this cell type
    sort_inds_for_cell = np.argsort(cell_to_plot_df['phi'].to_numpy())[::-1]

    # plot the response to every fourth stimulus
    # this is to skip the repeats over phase
    num_phase_rep = 4
    pred_plot = pred[::num_phase_rep, :, cell_to_plot_bool].detach().cpu().numpy()
    stim_plot = cell_inputs[::num_phase_rep, :, cell_to_plot_bool].detach().cpu().numpy()
    pu.plot_response_over_space(pred_plot[:, time_point, :], cell_x, cell_z, save_folder=model_folder / 'analysis' / ctp)

    for ri, r in enumerate(pred_plot):
        plt.figure()
        plt.imshow(r[:, sort_inds_for_cell], interpolation='nearest', aspect='auto')
        plt.clim(0, np.max(r))
        plt.colorbar()
        plt.savefig(model_folder / 'analysis' / ctp / ('xt_resp' + str(ri) + '.png'))
        plt.close()

    ave_resp_neurons = np.mean(pred_plot, axis=2)
    plt.figure()
    plt.imshow(ave_resp_neurons, interpolation='nearest', aspect='auto')
    plt.clim(0, np.max(ave_resp_neurons))
    plt.colorbar()
    plt.savefig(model_folder / 'analysis' / ctp / 'ave_resp_neurons.png')
    plt.close()

    ave_resp_neurons_time = np.mean(pred_plot, axis=(1, 2))
    plt.figure()
    plt.plot(ave_resp_neurons_time)
    plt.ylim(0, np.max(ave_resp_neurons_time) * 1.1)
    plt.savefig(model_folder / 'analysis' / ctp / 'ave_resp_neurons_time.png')
    plt.close()

pu.plot_readout_target(readout, target, list(model.readout_indexes.keys()), save_folder=model_folder / 'analysis')

print('analysis completed')