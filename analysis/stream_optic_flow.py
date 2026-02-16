import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import neural_models as nm

# This script loads the model and then iteratively calculates the optic flow in sequential stimuli

# Load the model
model_path = 'models/best_model/model.pt'
model = nm.NeuralModel.load(model_path, device='cpu')
initial_state_right = None
initial_state_left = None
# this variable gives you the ommatidia locations in the polar coordinates phi and theta for the right and left eye. The model will expect
# an input for each of these columns, so you should sample your images using these coordinates.
# best case you should filter with a gaussian blur with 2.5 degree std before sampling. but its probably fine to skip if its hard
column_phi_theta_right = model.column_phi_theta
column_phi_theta_left = model.column_phi_theta
column_phi_theta_left[:, 0] = -column_phi_theta_left[:, 0]

# this is an example streaming loop where you load in an image and use the model to process it
while True:
    # load in the stimulus with your code
    # visual stimulus should be of the form [stimuli, time, ommatidia].
    # If you just have one timepoint and one stimulus then it will have shape [1, 1, 796]
    visual_stimulus_right, visual_stimulus_left = some_load_function(column_phi_theta_right, column_phi_theta_left)

    # Calculate optic flow predictions
    # set the initial state to be the last state of the model.
    # optic flow has shape [stimuli, time, ommatidia, (d_right, d_up)] in units of approximately degrees per second

    # --- If you only have a single camera or if you just want flow you can get the optic flow with the following ---
    # outputs, initial_state = model.predict(visual_stimulus_right, init_state=initial_state)[0:2]
    # optic_flow = outputs['decoder']

    # --- with two inputs, you can get the model's self motion directly with ---
    # ego rotation is rotating around [x, y, z]
    # ego translation is along [x, y, z]
    ego_rotation, ego_translation, initial_state_right, initial_state_left = model.estimate_ego_motion_from_image(visual_stimulus_right, visual_stimulus_left, initial_state_right=initial_state_right, initial_state_left=initial_state_left)

    # figure out the drone command based on optic flow
    done_command = get_drone_command_from_flow(ego_rotation, ego_translation)

    # interact with the quad copter
    take_drone_action(done_command)