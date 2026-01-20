import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pandas as pd
import neural_models as nm
import plotting_utilities as pu
import data_utilities as du

fit_model_readout = True
model_folder = Path('models/best_model')
local_config = du.load_config()

# Load model, stimulus, and config
model = nm.CurrentModel.load(model_folder / 'model.pt', device=local_config['model']['device'])
stimulus = torch.load(model_folder / 'cell_inputs.pt')
target = torch.load(model_folder / 'target.pt')

if fit_model_readout:
    # fit a simple readout from model to flow
    # The model stores the readout, so not necessary to do every time. Just once
    model.find_steady_state()
    sine_wave_velocity = torch.zeros_like(torch.stack([stimulus, stimulus], axis=-1))
    sine_wave_velocity[:, :, :, 0] = target['T4a'] - target['T4b']
    sine_wave_velocity[:, :, :, 1] = target['T4c'] - target['T4d']
    model.fit_optic_flow_readout(stimulus, sine_wave_velocity)
    model.save(model_folder / 'model.pt')

# Calculate optic flow from model, format is [stimuli, time, ommatidia, (horizontal, vertical flow)]
optic_flow = model.calculate_optic_flow(stimulus)[0]

# Plot XT plots with optic flow
pu.xt_plot(
    stimulus.detach().cpu().numpy(),
    model.column_phi_theta[:, 0],
    optic_flow=optic_flow.detach().cpu().numpy(),
    save_folder=model_folder / 'optic_flow_analysis',
    plot_every=4
)
