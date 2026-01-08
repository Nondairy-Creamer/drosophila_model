import torch
import pandas as pd
from pathlib import Path
import neural_models as nm
import plotting_utilities as pu
import data_utilities as du

model_folder = Path('models/best')

# Load model, stimulus, and config
model = nm.NeuralRnn.load(model_folder / 'model.pt', device='cpu')
stimulus = torch.load(model_folder / 'cell_inputs.pt')
target = torch.load(model_folder / 'target.pt')
config = du.load_config(model_folder / 'config.toml')

# Load column info to get phi (x-axis for xt_plot)
connectome_folder = Path(config['data']['connectome_folder'])
column_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_column_info.parquet')

# fit a simple readout from model to flow
sine_wave_velocity = torch.zeros_like(torch.stack([stimulus, stimulus], axis=-1))
sine_wave_velocity[:, :, :, 0] = target['T4a'] - target['T4b']
sine_wave_velocity[:, :, :, 1] = target['T4c'] - target['T4d']
model.fit_optical_flow_readout(stimulus, sine_wave_velocity)

# Calculate optic flow from model, format is [stimuli, time, ommatidia, (horizontal, vertical flow)]
optic_flow = model.calculate_optical_flow(stimulus)

# # Plot XT plots with optic flow
# pu.xt_plot(
#     stimulus.detach().cpu().numpy(),
#     column_info_df['phi'].to_numpy(),
#     optic_flow=optic_flow.detach().cpu().numpy(),
#     save_folder=model_folder / 'optic_flow_analysis',
#     plot_every=4
# )
