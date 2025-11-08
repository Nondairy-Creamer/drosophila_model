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

cells_to_plot = ['L2', 'Tm1', 'Tm2', 'Tm4', 'Tm9', 'T5a', 'L1', 'Mi1', 'T4a']

# for ctp in cells_to_plot:
#     print('--- Model parameters for', ctp, '---')
#     chosen_neuron_bool = model.unique_cell_types == ctp
#     chosen_diagonal = model.diagonal_logit[chosen_neuron_bool].sigmoid().detach().cpu().numpy()
#     chosen_strength = (model.input_strength_log[chosen_neuron_bool].exp() * model.output_strength_log[chosen_neuron_bool].exp()).detach().cpu().numpy()
#     chosen_offset = model.offset[chosen_neuron_bool].detach().cpu().numpy()
#     chosen_stimulus_strength = model.stimulus_strength_log[[chosen_neuron_bool]].exp().detach().cpu().numpy()

#     print(f'diagonal: {chosen_diagonal[0]:4g}')
#     print(f'synapse strength: {chosen_strength[0]:4g}')
#     print(f'offset: {chosen_offset[0]:4g}')
#     print(f'stimulus strength: {chosen_stimulus_strength[0]:4g}')

# Load contrast scaling factor and create loss function
contrast_scaling_factor_path = model_folder / 'contrast_scaling_factor.pt'
contrast_scaling_factor = torch.load(contrast_scaling_factor_path).to(device=model.device)
loss_fn = getattr(tu, model_config['training']['loss_fn'])
loss_scaled = tu.get_scaled_loss(loss_fn, contrast_scaling_factor)

connectome_folder = Path(model_config['data']['connectome_folder'])
cell_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_cell_info.parquet')
column_info_df, column_ids = du.get_column_df(cell_info_df, cells_to_plot)

# restrict the columns to a line across the eye
vert_lim = 50 * np.pi / 180
within_vert_lim = np.abs(column_info_df['theta'] - np.pi / 2) < vert_lim
column_info_df = column_info_df[within_vert_lim]
column_ids = column_ids[within_vert_lim]
sort_inds = np.argsort(column_info_df['phi'])[::-1]

pred, readout = model.predict_and_readout(stimulus, stim_to_cell_mapping)
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
    pu.plot_response_over_space(pred_plot[:, time_point, :], cell_x, cell_z, save_folder=model_folder / 'analysis' / ctp)

    for ri, r in enumerate(pred_plot):
        plt.figure()
        plt.imshow(r[:, sort_inds_for_cell], interpolation='nearest', aspect='auto', vmin=0)
        plt.xlabel('phi')
        plt.ylabel('time')
        plt.colorbar()
        plt.savefig(model_folder / 'analysis' / ctp / ('xt_resp' + str(ri) + '.png'))
        plt.close()

pu.plot_readout_target(readout, target, list(model.readout_indexes.keys()), save_folder=model_folder / 'analysis')







# # do a full sine wave sweep
# # --- get a sign wave stimulus ---
# # generate a sine wave on a cylinder around the fly
# import model_utilities as mu
# dt = model_config['stimulus']['dt']
# max_time = 8
# input_neurons = ['L1', 'L2', 'L3']  # Convert list to tuple for compatibility
# output_neurons = ['T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d']
# temporal_frequencies = [0.25, 1, 4, 16]

# # get the column id for each column that has all of the neurons in input_neurons
# column_info_input_df, column_ids_input = du.get_column_df(cell_info_df, input_neurons)
# column_info_output_df, column_ids_output = du.get_column_df(cell_info_df, output_neurons)

# sine_waves, sine_wave_tfs, contrast_scaling_function = mu.create_sine_wave(column_info_input_df.phi.to_numpy(), column_info_input_df.theta.to_numpy(), 
#     temporal_frequency=temporal_frequencies, dt=dt, max_time=max_time)
# sine_waves = torch.tensor(sine_waves, device=device, dtype=dtype)
# stim_to_cell_mapping = mu.get_stim_to_cell_mapping(cell_info_df, column_info_input_df, input_neurons=input_neurons)
# stim_to_cell_mapping = torch.tensor(stim_to_cell_mapping, device=model.device)

# pred, readout = model.predict_and_readout(sine_waves, stim_to_cell_mapping=stim_to_cell_mapping)

# ave_sine_resp = readout.detach().cpu().numpy().mean(axis=(2, 3))
# sine_wave_dir = ['right', 'left', 'up', 'down']
# num_phase = 4
# ave_sine_resp = ave_sine_resp[:, ::num_phase] # skip the different phases for now
# num_sine_dir = len(sine_wave_dir)
# ave_sine_resp = ave_sine_resp.reshape((ave_sine_resp.shape[0], ave_sine_resp.shape[1] // num_sine_dir, num_sine_dir))

# for ni, n in enumerate(output_neurons):
#     plt.figure()
#     fig, ax = plt.subplots()
#     ax.plot(np.log(temporal_frequencies), ave_sine_resp[ni])
#     ax.set_xticks(np.log(temporal_frequencies))
#     ax.set_xticklabels(temporal_frequencies)
#     plt.title(n)
#     plt.legend(sine_wave_dir)
#     plt.savefig(model_folder / 'analysis' / (n + '_sine_wave_sweep.png'))
#     plt.close()







print('analysis completed')
