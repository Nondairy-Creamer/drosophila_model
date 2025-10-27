import torch
import numpy as np
from pathlib import Path
import sys, pathlib
import pandas as pd
import shutil
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # adds /workspace
import neural_models as nm
import data_utilities as du
import plotting_utilities as pu
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

model_folder = Path('models/20251027_150715')

config = du.load_config(model_folder / 'config.toml')
model = nm.NeuralRnn.load(model_folder / 'model.pt')
cell_inputs = torch.load(model_folder / 'cell_inputs.pt').to(device=model.device)
target = torch.load(model_folder / 'target.pt').to(device=model.device)

connectome_folder = Path(config['data']['connectome_folder'])
cell_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_cell_info.parquet')

cells_to_plot = ['L1', 'Mi1', 'T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d']
column_info_df, column_ids = du.get_column_df(cell_info_df, cells_to_plot)
pred, readout = model.predict_and_readout(cell_inputs)
stim_fig_folder = Path('figures/analysis')
shutil.rmtree(stim_fig_folder, ignore_errors=True)
stim_fig_folder.mkdir(parents=True, exist_ok=True)
time_point = 20

pred_list = []
for ctp in cells_to_plot:
    cell_to_plot_bool = (cell_info_df.type == ctp) & cell_info_df.column_id.isin(column_ids)
    cell_x = column_info_df.x
    cell_z = column_info_df.z

    # plot the response to every fourth stimulus
    # this is to skip the repeats over phase
    half_way = pred.shape[1] // 2
    pred_plot = pred[::4, half_way:, cell_to_plot_bool].detach().cpu().numpy()
    pred_list.append(pred_plot)
    pu.plot_response_over_space(pred_plot[:, time_point, :], cell_x, cell_z, save_folder=model_folder / 'analysis' / ctp)

    ave_resp = np.mean(pred_plot, axis=2)
    plt.figure()
    plt.imshow(ave_resp, interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.savefig(model_folder / 'analysis' / ctp / 'ave_resp.png')
    plt.close()

for i in range(2, len(pred_list) - 1, 2):
    ave_resp = np.mean(pred_list[i], axis=(1, 2)) - np.mean(pred_list[i + 1], axis=(1, 2))
    plt.figure()
    # plt.imshow(ave_resp, interpolation='nearest', aspect='auto')
    plt.plot(ave_resp)
    # plt.colorbar()
    plt.savefig(model_folder / 'analysis' / ('ave_resp_sub_' + str(i) + '.png'))
    plt.close()

pu.plot_readout_target(readout, target, list(model.readout_neurons.keys()), save_folder=model_folder / 'analysis')

print('analysis completed')