import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import shutil
from neural_models import NeuralRnn
from stimulus_utilities import make_lr_sine_wave
import plotting_utilities as pu

# This script loads the model and then iteratively calculates the optic flow in sequential stimuli

# Load the model
model_path = 'models/best_model/model.pt'
model = NeuralRnn.load(model_path, device='cpu')
initial_state = None

while True:
    # load in the stimulus with your code
    visual_stimulus = some_load_function()

    # Calculate optic flow predictions
    optic_flow, initial_state = model.calculate_optic_flow(visual_stimulus, initial_state=initial_state)

    # figure out the drone command based on optic flow
    done_command = get_drone_command_from_flow(optic_flow)


    # interact with the quad copter
    take_drone_action(done_command)