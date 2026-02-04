import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import neural_models as nm

# This script loads the model and then iteratively calculates the optic flow in sequential stimuli

# Load the model
model_path = 'models/best_model/model.pt'
model = nm.NeuralModel.load(model_path, device='cpu')
initial_state = None
# this variable gives you the ommatidia locations in the polar coordinates phi and theta. The model will expect
# an input for each of these columns.
# For this, it won't matter that this is a right eye, just stretch your image accross the space defined by the ommatidia
# i.e. define your image from approximately [0, -145] degrees in phi and from [20, 160] degrees in theta
# and then sample the image at the locations in column_phi_theta
column_phi_theta = model.column_phi_theta

# this is an example streaming loop where you load in an image and use the model to process it
while True:
    # load in the stimulus with your code
    # visual stimulus should be of the form [stimuli, time, ommatidia].
    # If you just have 1 camera and one timepoint then it will have shape [1, 1, 796]
    visual_stimulus = some_load_function(column_phi_theta)

    # Calculate optic flow predictions
    # set the initial state to be the last state of the model. It is first initialized with zeros which is why it has a warmup period
    # but after a few iterations should be reliable
    # optic flow has shape [stimuli, time, ommatidia, (d_phi, d_theta)] in units of approximately radians per second
    optic_flow, initial_state = model.calculate_optic_flow(visual_stimulus, initial_state=initial_state)

    # figure out the drone command based on optic flow
    done_command = get_drone_command_from_flow(optic_flow)

    # interact with the quad copter
    take_drone_action(done_command)