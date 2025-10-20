import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from pathlib import Path
import sys, pathlib
import pandas as pd
import shutil
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # adds /workspace
import neural_models as nm
import data_utilities as du

model_folder = Path('models/2025/10/20/15_16_23')

config = du.load_config(model_folder / 'config.toml')
model = nm.NeuralRnn.load(model_folder / 'model.pt')
cell_inputs = torch.load(model_folder / 'cell_inputs.pt').to(device=model.device)

connectome_folder = Path(config['data']['connectome_folder'])
column_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_column_info.parquet')
cell_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_cell_info.parquet')

cells_to_plot = ['L1', 'Mi1', 'T4a']
pred, readout = model.predict_and_readout(cell_inputs)
stim_fig_folder = Path('figures/analysis')
shutil.rmtree(stim_fig_folder, ignore_errors=True)
stim_fig_folder.mkdir(parents=True, exist_ok=True)
time_point = 15

for ctp in cells_to_plot:
    plot_column_df = column_info_df[column_info_df.type == ctp]
    cell_to_plot_bool = column_info_df.type == ctp
    cell_x = plot_column_df.x
    cell_z = plot_column_df.z

    # plot the response to each stimulus
    for pi, p in enumerate(pred[:4]):
        # plot the cell response
        c = p[time_point, plot_column_df.root_ind.to_numpy()].detach().cpu().numpy()
        plt.figure()
        sc = plt.scatter(cell_x, cell_z, c=c)
        cb = plt.colorbar(sc)
        plt.savefig(stim_fig_folder / (ctp + '_resp_' + str(pi) + '.png'))
        plt.close()

print('analysis completed')