import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import neural_models as nm
from stimulus_utilities import build_ommatidium_sampling_matrix

# This script loads the model and then iteratively calculates the optic flow in sequential stimuli

# Load the model
model_path = 'models/best_model/model.pt'
model = nm.NeuralModel.load(model_path, device='cpu')
initial_state_right = None
initial_state_left = None
initial_filter_state_right = None
initial_filter_state_left = None
cached_mappings = None

# Ommatidia locations in polar coordinates (phi, theta) for right and left eyes.
# The model expects one input per ommatidium, so images must be sampled at these coordinates.
sample_phi_right = model.column_phi_theta[:, 0].numpy()
sample_theta = model.column_phi_theta[:, 1].numpy()
sample_phi_left = -sample_phi_right

# ------------------------------------------------------------------
# Set up the sampling matrices (do this once before the loop).
# You need to know:
#   - image_shape: (H, W) of the images you will load
#   - phi_right_lim / phi_left_lim: azimuthal FOV limits (radians)
#   - theta_lim: polar FOV limits (radians)
# Adjust these to match your camera.
# ------------------------------------------------------------------
image_height, image_width = 480, 640  # <-- replace with your image dimensions

phi_right_lim = np.array([-30, -150]) * np.pi / 180   # right camera azimuth limits
phi_left_lim  = np.array([150,   30]) * np.pi / 180   # left camera azimuth limits
theta_lim     = np.array([ 45,  135]) * np.pi / 180   # polar (elevation) limits

filter_std_deg = 2.5  # Gaussian blur std in degrees applied before sampling

W_right = build_ommatidium_sampling_matrix(
    (image_height, image_width), phi_right_lim, theta_lim,
    sample_phi_right, sample_theta, filter_std=filter_std_deg,
)
W_left = build_ommatidium_sampling_matrix(
    (image_height, image_width), phi_left_lim, theta_lim,
    sample_phi_left, sample_theta, filter_std=filter_std_deg,
)

# Ommatidia outside the camera FOV get no weight; fill them with 0.5 (mid-gray)
in_fov_right = np.asarray(W_right.sum(axis=1)).ravel() > 0
in_fov_left  = np.asarray(W_left.sum(axis=1)).ravel() > 0


def sample_image(image_gray, W, in_fov):
    """Sample a single grayscale image at ommatidium locations.

    Args:
        image_gray: (H, W) uint8 grayscale image
        W: sparse (N_ommatidia, H*W) sampling matrix
        in_fov: (N_ommatidia,) bool mask of ommatidia inside camera FOV

    Returns:
        sampled: (1, 1, N_ommatidia) float32 tensor in [0, 1],
                 shaped as [stimuli, time, ommatidia] ready for the model.
    """
    flat = image_gray.ravel().astype(np.float32)
    sampled = np.asarray(W.dot(flat)).ravel() / 255.0
    sampled[~in_fov] = 0.5
    return torch.from_numpy(sampled).unsqueeze(0).unsqueeze(0)  # [1, 1, N]


# this is an example streaming loop where you load in an image and use the model to process it
while True:
    # load in the stimulus with your code — should be a grayscale uint8 image of shape (H, W)
    image_right = some_load_function()  # e.g. cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image_left  = some_load_function()

    # Sample at ommatidium locations → [1, 1, N_ommatidia] float tensor in [0, 1]
    visual_stimulus_right = sample_image(image_right, W_right, in_fov_right)
    visual_stimulus_left  = sample_image(image_left,  W_left,  in_fov_left)

    # Calculate optic flow predictions
    # set the initial state to be the last state of the model.
    # optic flow has shape [stimuli, time, ommatidia, (d_right, d_up)] in units of approximately degrees per second

    # --- If you only have a single camera or if you just want flow you can get the optic flow with the following ---
    # result = model.predict(visual_stimulus_right, init_state=initial_state)
    # optic_flow = result['outputs']['decoder']
    # initial_state = result['model_state']['v']

    # --- with two inputs, you can get the model's self motion directly with ---
    # ego rotation is rotating around [x, y, z]
    # ego translation is along [x, y, z]
    ego_rotation, ego_translation, right_res, left_res = model.estimate_ego_motion_from_image(
        visual_stimulus_right, visual_stimulus_left,
        initial_state_right=initial_state_right, initial_state_left=initial_state_left,
        initial_filter_state_right=initial_filter_state_right, initial_filter_state_left=initial_filter_state_left,
        cached_mappings=cached_mappings,
    )
    if cached_mappings is None:
        cached_mappings = right_res['mappings']
    initial_state_right = right_res['model_state']['v']
    initial_state_left = left_res['model_state']['v']
    initial_filter_state_right = right_res['model_state']['filter_state']
    initial_filter_state_left = left_res['model_state']['filter_state']

    # figure out the drone command based on optic flow
    done_command = get_drone_command_from_flow(ego_rotation, ego_translation)

    # interact with the drone
    take_drone_action(done_command)