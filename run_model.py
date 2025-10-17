import numpy as np
import torch
import model_utilities as mu
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import neural_models as nm
import time
import data_utilities as du
from datetime import datetime

# Load configuration from TOML file
config = du.load_config()

# Clear previous figures (delete non-hidden files)
fig_dir = Path('figures')
fig_dir.mkdir(parents=True, exist_ok=True)
for p in fig_dir.iterdir():
    if p.is_file() and not p.name.startswith('.'):
        p.unlink()

# Extract parameters from config
device = torch.device(config['model']['device'])
dtype = getattr(torch, config['model']['dtype'])
dt = config['simulation']['dt']
max_time = config['simulation']['max_time']
num_steps = config['simulation']['num_steps']
alpha = config['simulation']['alpha']
time_points = torch.arange(0, max_time, step=dt)
num_time = len(time_points)
use_bfgs = config['simulation']['use_bfgs']
input_neurons = ['L1', 'L2', 'L3']  # Convert list to tuple for compatibility

# Generate temporal frequencies
temporal_frequencies = list(np.arange(
    config['stimulus']['temporal_freq_start'],
    config['stimulus']['temporal_freq_end'],
    config['stimulus']['temporal_freq_step'],
))

# --- load in connectivity and cell info data ---
connectome_folder = Path(config['data']['connectome_folder'])
column_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_column_info.parquet')
cell_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_cell_info.parquet')
synapse_counts_df = pd.read_parquet(connectome_folder / 'optic_lobes_synapse_counts.parquet')
num_neurons = len(cell_info_df.index)

# create the model
model = nm.NeuralRnn(synapse_counts_coo=synapse_counts_df[['pre_matrix_ind', 'post_matrix_ind', 'synapse_counts']].to_numpy(), 
                     cell_types=cell_info_df.type.to_numpy(), nonlinearity=config['model']['nonlinearity'],
                     device=config['model']['device'], dtype=config['model']['dtype'], readout_neurons=config['model']['readout_neurons'],
                     use_sparse=config['model']['use_sparse'], scale_in_out=config['model']['scale_in_out'])

# --- get a sign wave stimulus ---
# generate a sine wave on a cylinder around the fly
# get the column info for all L1s and use that to know the location and column ID for each ommatidia
l1_df = column_info_df[column_info_df['type'] == 'L1']

sine_waves = mu.create_sine_wave(l1_df.phi.to_numpy(), l1_df.theta.to_numpy(), 
    temporal_frequency=temporal_frequencies, dt=dt, max_time=max_time)
# we're going to put the sinewaves into L1, L2, L3 so invert contast
sine_waves = torch.tensor(sine_waves, device=device, dtype=dtype)
cell_inputs = mu.map_stimulus_to_cells(sine_waves, column_info_df, cell_info_df.root_id, l1_df.column_id, input_neurons=input_neurons)

# --- train the model ---
if use_bfgs:
    opt = torch.optim.LBFGS(model.parameters(), line_search_fn='strong_wolfe')
else:
    opt = torch.optim.Adam(model.parameters(), lr=alpha)
    
# Define the target
all_tf = out = [i for i in temporal_frequencies for _ in (0, 1)]
all_tf = all_tf + [-i for i in all_tf]
target = torch.zeros(cell_inputs.shape[0], device=device, dtype=dtype)
target[0 : len(all_tf)] = torch.tensor(all_tf, device=device, dtype=dtype)

# find a good estimate of the readout weights
model.estimate_readout_weights(cell_inputs, target)
timestamp = datetime.now().strftime('%Y/%m/%d/%H_%M_%S')
model_save_folder = Path('models/' + timestamp)
loss_list = []

for step in range(num_steps):
    def closure():
        opt.zero_grad()
        readout = model.predict_and_readout(cell_inputs)[1]
        loss = torch.nn.functional.mse_loss(readout, target)
        loss.backward()
        return loss

    if use_bfgs:
        loss = opt.step(closure)
    else:
        loss = closure()
        opt.step()

    # log once per outer step
    loss_list.append(loss.item())

    print(f'step {step + 1} / {num_steps} | loss = {loss.item()}')

    if (step % config['simulation']['save_every'] == 0) | (step == num_steps - 1):
        with torch.no_grad():
            model.save(model_save_folder / 'model.pt')
            du.save_config(config, model_save_folder / 'config.toml')
            torch.save(cell_inputs.detach().cpu(), model_save_folder / "cell_inputs.pt")
            pred, readout = model.predict_and_readout(cell_inputs)

            if step == 0:
                cell_to_plot = 'L1'
                plot_column_df = column_info_df[column_info_df.type == cell_to_plot]
                cell_to_plot_bool = column_info_df.type == cell_to_plot
                cell_x = plot_column_df.x
                cell_z = plot_column_df.z

                c = cell_inputs[0, 0, plot_column_df.root_ind.to_numpy()].detach().cpu().numpy()
                plt.figure()
                plt.scatter(cell_x, cell_z, c=c)
                plt.savefig('figures/pro_stim.png')
                plt.close()

                c = cell_inputs[int(sine_waves.shape[0] / 2), 0, plot_column_df.root_ind.to_numpy()].detach().cpu().numpy()
                plt.figure()
                plt.scatter(cell_x, cell_z, c=c)
                plt.savefig('figures/up_stim.png')
                plt.close()

            plt.figure()
            colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
            direction = ['progressive', 'regressive', 'up', 'down']
            readout_plot = readout.detach().cpu().numpy().T.reshape((4, int(readout.shape[0]/4))).T
            target_plot = target.detach().cpu().numpy().T.reshape((4, int(target.shape[0]/4))).T
            for i in range(readout_plot.shape[1]):
                plt.plot(target_plot[:, i], color=colors[i], label=direction[i] + ' target')
                plt.plot(readout_plot[:, i], color=colors[i], ls='--', label=direction[i] + ' prediction')
            plt.legend()
            plt.savefig('figures/pred_vs_target.png')
            plt.close()

            plt.figure()
            plt.plot(loss_list, label='loss')
            plt.legend()
            plt.savefig('figures/loss.png')
            plt.close()
