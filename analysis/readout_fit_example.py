import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pandas as pd
import neural_models as nm
import plotting_utilities as pu
import data_utilities as du

model_folder = Path('models/best_on_sintel/20260120_181237_gpu0_run0')

# Load model, stimulus, and config
print("Loading model...")
model = nm.NeuralModel.load(model_folder / 'model.pt', device='cpu')
stimulus = torch.load(model_folder / 'cell_inputs.pt')

if not model.has_decoder:
    raise ValueError("Model has no decoder. Cannot predict optic flow.")

# Run model to get decoder optic flow: [stimuli, time, ommatidia, (horizontal, vertical flow)]
with torch.no_grad():
    outputs = model.predict(stimulus, input_neurons=['L1', 'L2', 'L3'])[0]
    optic_flow = outputs['decoder']

# Plot XT plots with optic flow
pu.xt_plot(
    stimulus.detach().cpu().numpy(),
    model.column_phi_theta[:, 0],
    optic_flow=optic_flow.detach().cpu().numpy(),
    save_folder=model_folder / 'optic_flow_analysis',
    plot_every=4
)
