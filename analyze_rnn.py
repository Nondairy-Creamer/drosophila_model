import torch
import numpy as np
from pathlib import Path
import pandas as pd
import shutil
import neural_models as nm
import data_utilities as du
import plotting_utilities as pu
import training_utilities as tu
import stimulus_utilities as su
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import sys

if len(sys.argv) == 1:
    model_folder = Path('models/20251128_224751')
else:
    model_folder = Path(sys.argv[1])

# Clear and recreate the analysis folder so outputs are always fresh
analysis_folder = model_folder / 'analysis'
shutil.rmtree(analysis_folder, ignore_errors=True)
analysis_folder.mkdir(parents=True, exist_ok=True)
do_sine_sweep = False

model_config = du.load_config(model_folder / 'config.toml')
computer_config = du.load_config()
device = torch.device(computer_config['model']['device'])  # what device you use depends on the computer, not how the model was run
dtype = getattr(torch, model_config['model']['dtype'])
model = nm.NeuralRnn.load(model_folder / 'model.pt', device=device)
stimulus = torch.load(model_folder / 'cell_inputs.pt').to(device=model.device)
# target is now a dict of tensors; load and move entries to the right device/dtype
_target_loaded = torch.load(model_folder / 'target.pt')
target = {k: (v.to(device=model.device, dtype=dtype) if torch.is_tensor(v) else v)
            for k, v in _target_loaded.items()}
time_point = 20
input_neurons = ['L1', 'L2', 'L3']

# Load contrast scaling factor and create loss function
contrast_scaling_factor_path = model_folder / 'contrast_scaling_factor.pt'
contrast_scaling_factor = torch.load(contrast_scaling_factor_path).to(device=model.device)
loss_fn = getattr(tu, model_config['training']['loss_fn'])

connectome_folder = Path(model_config['data']['connectome_folder'])
cell_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_cell_info.parquet')
column_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_column_info.parquet')

# # --- get model prediction on the cells it was trained on ---
# trained_cells = ['T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d']
# pred_trained = model.predict(stimulus, input_neurons=input_neurons, output_neurons=trained_cells)[0]
# pred_trained = pred_trained['firing_rate']
# # get the correlation and cosine similarity
# correlation_similarity = 1.0 - tu.calc_loss(tu.corr_loss, pred_trained, target).detach().cpu().numpy()
# cosine_similarity = 1.0 - tu.calc_loss(tu.cos_loss, pred_trained, target).detach().cpu().numpy()
# print('correlation:', correlation_similarity)
# print('cosine similarity:', cosine_similarity)

# # --- make XT plots of the cell responses next to the target ---
# for ctp in trained_cells:
#     ctp_dir = analysis_folder / ctp
#     ctp_dir.mkdir(parents=True, exist_ok=True)
#     # plot the response to every fourth stimulus
#     # this is to skip the repeats over phase
#     num_phase_rep = 4
#     pred_plot = pred_trained[ctp][::num_phase_rep, :, :].detach().cpu().numpy()
#     target_plot = target[ctp][::num_phase_rep, :, :].detach().cpu().numpy()
#     # pu.plot_response_over_space(pred_plot[:, time_point, :], column_info_df.x, column_info_df.z, save_folder=model_folder / 'analysis' / ctp)
#     pu.xt_plot(pred_plot, column_info_df['phi'].to_numpy(), target=target_plot, save_folder=ctp_dir)

#     # --- scatter prediction vs target ---
#     plt.figure()
#     plt.scatter(target_plot.reshape(-1), pred_plot.reshape(-1))
#     plt.xlabel('target')
#     plt.ylabel('prediction')
#     plt.title('ctp')
#     plt.savefig(ctp_dir / 'scatter.png')


# --- select some additional neurons for plotting ---
cells_to_plot = ['L1', 'L2', 'Mi1', 'Tm3', 'Mi9', 'Mi4', 'C3', 'CT1', 'T4a', 'T4b', 'T4c', 'T4d']

# --- silence certain neurons ---
neurons_to_silence = []
model.silence_neurons(neurons_to_silence)

pred = model.predict(stimulus, input_neurons=input_neurons, output_neurons=cells_to_plot)[0]
pred = pred['firing_rate']
stim_fig_folder = Path('figures/analysis')
shutil.rmtree(stim_fig_folder, ignore_errors=True)
stim_fig_folder.mkdir(parents=True, exist_ok=True)
num_connections_to_show = 20

# get the average in / out connections for each cell type (vectorized)
ave_con = np.zeros((model.num_cell_types, model.num_cell_types))
indices = model.synapse_counts.indices().cpu().numpy()  # shape (2, nnz)
values = model.synapse_counts.values().cpu().numpy()    # shape (nnz,)

# Get cell type index for each pre and post neuron in the synapse list
pre_types = model.neuron_to_type_index[indices[0]]
post_types = model.neuron_to_type_index[indices[1]]

# Use pandas groupby to compute mean for each (pre_type, post_type) pair
synapse_df = pd.DataFrame({
    'pre_type': pre_types,
    'post_type': post_types,
    'count': values
})
ave_con_df = synapse_df.groupby(['post_type', 'pre_type'])['count'].mean()

# Fill the ave_con matrix
for (post_type, pre_type), mean_val in ave_con_df.items():
    ave_con[int(post_type), int(pre_type)] = mean_val

for ns in cells_to_plot:
    # --- print out model parameters ---
    print('')
    print('--- Model parameters for', ns, '---')
    cell_index = np.where(model.unique_cell_types == ns)[0][0]
    chosen_persistence = model.persistence_logit[cell_index].sigmoid().detach().cpu().numpy()
    chosen_strength_in = model.input_strength_log[cell_index].exp().detach().cpu().numpy()
    chosen_strength_out = model.output_strength_log[cell_index].exp().detach().cpu().numpy()
    chosen_offset = model.offset[cell_index].detach().cpu().numpy()
    chosen_stimulus_strength = model.stimulus_strength_log[cell_index].exp().detach().cpu().numpy()

    chosen_connections_in = ave_con[:, cell_index]
    chosen_connections_out = ave_con[cell_index, :]
    largest_connections_in_indices = np.argsort(np.abs(chosen_connections_in))[::-1]
    largest_connections_out_indices = np.argsort(np.abs(chosen_connections_out))[::-1]
    largest_connections_in = chosen_connections_in[largest_connections_in_indices][:num_connections_to_show]
    largest_connections_out = chosen_connections_out[largest_connections_out_indices][:num_connections_to_show]
    largest_connections_in_names = model.unique_cell_types[largest_connections_in_indices][:num_connections_to_show]
    largest_connections_out_names = model.unique_cell_types[largest_connections_out_indices][:num_connections_to_show]

    print('persistence:', chosen_persistence, ', offset:', chosen_offset)
    print('strength in:', chosen_strength_in, ', strength out:', chosen_strength_out)

    display_df = pd.DataFrame({
        "incoming neurons": largest_connections_in_names,
        "incoming synapses": largest_connections_in,
        "outgoing names": largest_connections_out_names,
        "outgoing synapses": largest_connections_out,
    })

    print(display_df)

    # --- make XT plots of the cell responses next to the target ---
    ctp_dir = analysis_folder / ns
    ctp_dir.mkdir(parents=True, exist_ok=True)
    # plot the response to every fourth stimulus
    # this is to skip the repeats over phase
    num_phase_rep = 4
    pred_plot = pred[ns][::num_phase_rep, :, :].detach().cpu().numpy()
    # pu.plot_response_over_space(pred_plot[:, time_point, :], column_info_df.x, column_info_df.z, save_folder=model_folder / 'analysis' / ctp)
    pu.xt_plot(pred_plot, column_info_df.phi.to_numpy(), save_folder=ctp_dir)

# --- get the responses to flashes


if do_sine_sweep:
    # do a full sine wave sweep
    # --- get a sign wave stimulus ---
    # generate a sine wave on a cylinder around the fly
    dt = model_config['model']['dt']
    stim_time = 2
    input_neurons = ['L1', 'L2', 'L3']  # Convert list to tuple for compatibility
    output_neurons = ['T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d']
    temporal_frequencies = [0.25, 0.5, 1, 2, 4, 8, 16]

    sine_waves, sine_wave_tfs, contrast_scaling_factor, blocks = su.create_sine_wave(column_info_df.phi.to_numpy(), column_info_df.theta.to_numpy(), 
        temporal_frequency=temporal_frequencies, dt=dt, stim_time=stim_time, pre_stim_time=0, post_stim_time=0)
    sine_waves = torch.tensor(sine_waves, device=device, dtype=dtype)
    sine_wave_tfs = torch.tensor(sine_wave_tfs, device=device, dtype=dtype)

    pred = model.predict(sine_waves, input_neurons=input_neurons, output_neurons=output_neurons)[0]

    num_phase = 4
    sine_wave_dir = ['right', 'left', 'up', 'down']
    num_tf = len(temporal_frequencies)
    num_dir = len(sine_wave_dir)

    ave_resp = {}
    for ons in output_neurons:
        ave_resp[ons] = np.nanmean(pred['firing_rate'][ons].detach().cpu().numpy(), axis=(1, 2))
        ave_resp[ons] = ave_resp[ons][::num_phase] # skip the different phases for now
        ave_resp[ons] = ave_resp[ons].reshape((num_dir, num_tf)).T

    for ni, n in enumerate(output_neurons):
        plt.figure()
        fig, ax = plt.subplots()
        ax.plot(np.log(temporal_frequencies), ave_resp[n])
        ax.set_xticks(np.log(temporal_frequencies))
        ax.set_xticklabels(temporal_frequencies)
        plt.title(n)
        plt.legend(sine_wave_dir)
        plt.ylim(bottom=0, top=None)
        plt.savefig(model_folder / 'analysis' / (n + '_sine_wave_sweep.png'))
        plt.close()




