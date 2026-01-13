"""
Test script to estimate self-motion from rotational sine wave stimulus.

This script:
1. Loads the trained model from models/best_model
2. Generates rotational sine wave stimuli at multiple velocities (30 deg wavelength)
3. Estimates self-motion (rotation and translation) from the optic flow
"""

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


def main():
    # Load the model
    model_path = 'models/best_model/model.pt'
    print(f"Loading model from {model_path}...")
    model = NeuralRnn.load(model_path, device='cpu')
    
    print(f"Model loaded successfully")
    print(f"  Number of neurons: {model.num_neurons}")
    print(f"  Number of columns: {model.num_columns}")
    
    # Get the column positions (phi, theta) from the model
    phi = model.column_phi_theta[:, 0]  # azimuthal angles (radians)
    theta = model.column_phi_theta[:, 1]  # polar angles (radians)
    
    # Stimulus parameters
    wavelength_deg = 30
    wavelength = wavelength_deg * np.pi / 180  # convert to radians
    
    # Test multiple velocities
    velocities = [i for i in range(30, 270, 30)]  # degrees/second

    print(f"\nStimulus parameters:")
    print(f"  Spatial wavelength: {wavelength_deg} degrees")
    print(f"  Velocities to test: {velocities} deg/s")
    
    # Create folder for xt plots
    plot_folder = Path('figures/test_self_motion_stimuli')
    plot_folder.mkdir(parents=True, exist_ok=True)
    
    # Clear existing files and subdirectories in the folder
    for file in plot_folder.iterdir():
        if file.is_file():
            file.unlink()
        elif file.is_dir():
            shutil.rmtree(file)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\n{'Velocity':<15} {'True Rotation':<30} {'Predicted Rotation':<30} {'True Translation':<30} {'Predicted Translation':<30}")
    print("-"*135)
    
    for velocity_deg_per_sec in velocities:
        temporal_frequency = velocity_deg_per_sec / wavelength_deg
        
        # Generate right eye stimulus
        # Positive temporal_frequency = rightward motion on retina = leftward observer rotation
        # Negative temporal_frequency = leftward motion on retina = rightward observer rotation
        right_stimulus, right_optic_flow = make_lr_sine_wave(
            sample_phi=phi,
            sample_theta=theta,
            wavelength=wavelength,
            temporal_frequency=temporal_frequency,
            phase_init=0,
            dt=0.01,
            stim_time=1.0
        )
        
        # Generate left eye stimulus (mirrored across midline)
        # Negate phi for the other eye
        left_stimulus, left_optic_flow = make_lr_sine_wave(
            sample_phi=-phi,
            sample_theta=theta,
            wavelength=wavelength,
            temporal_frequency=temporal_frequency,  # negative with -phi = rightward motion from left eye's perspective
            phase_init=0,
            dt=0.01,
            stim_time=1.0
        )
        
        # Add examples dimension for plotting and model input
        right_stimulus = right_stimulus[None, :, :]  # [1, time, points]
        right_optic_flow = right_optic_flow[None, :, :, :]  # [1, time, points, 2]
        left_stimulus = left_stimulus[None, :, :]  # [1, time, points]
        left_optic_flow = left_optic_flow[None, :, :, :]  # [1, time, points, 2]
        
        # Convert to tensors for model input
        right_inputs = torch.tensor(
            right_stimulus, 
            dtype=model.dtype, 
            device=model.device
        )
        left_inputs = torch.tensor(
            left_stimulus, 
            dtype=model.dtype, 
            device=model.device
        )
        
        # Calculate optic flow predictions
        optic_flow_left_pred = model.calculate_optic_flow(left_inputs, eye='left')[0]
        optic_flow_right_pred = model.calculate_optic_flow(right_inputs, eye='right')[[0]]
        
        # Generate xt plots with flow predictions
        pu.xt_plot_flow_prediction(
            left_inputs,
            right_inputs,
            optic_flow_left_pred,
            optic_flow_right_pred,
            phi,
            save_folder=plot_folder,
            save_tag=str(velocity_deg_per_sec)
        )
        
        # Estimate self-motion (skip first 40 time steps for equilibration)
        rotations, translations, depths = model.estimate_self_motion_from_image(right_inputs, left_inputs, equilibration_steps=40)
        
        # Average over time
        rotation = rotations[0].mean(dim=0)  # (3,) - rotation around x, y, z axes
        translation = translations[0].mean(dim=0)  # (3,) - translation in x, y, z directions
        
        # Convert rotation to degrees (signed vector)
        rotation_deg = np.degrees(rotation.detach().cpu().numpy())  # (3,) signed rotation vector
        
        # True rotation: [0, 0, velocity] for pure yaw rotation around z-axis
        # Positive velocity → rightward motion on retina → leftward observer rotation (positive omega_z)
        # Negative velocity → leftward motion on retina → rightward observer rotation (negative omega_z)
        true_rotation = np.array([0.0, 0.0, velocity_deg_per_sec])
        
        # True translation: [0, 0, 0] for pure rotation (no translation)
        true_translation = np.array([0.0, 0.0, 0.0])
        
        # Convert translation to numpy (signed vector)
        translation_np = translation.detach().cpu().numpy()  # (3,) signed translation vector
        
        # Format vectors as strings with 2 decimal places
        true_rot_str = f"[{true_rotation[0]:.2f}, {true_rotation[1]:.2f}, {true_rotation[2]:.2f}]"
        pred_rot_str = f"[{rotation_deg[0]:.2f}, {rotation_deg[1]:.2f}, {rotation_deg[2]:.2f}]"
        true_trans_str = f"[{true_translation[0]:.2f}, {true_translation[1]:.2f}, {true_translation[2]:.2f}]"
        pred_trans_str = f"[{translation_np[0]:.4f}, {translation_np[1]:.4f}, {translation_np[2]:.4f}]"
        
        print(f"{velocity_deg_per_sec:<15.1f} {true_rot_str:<30} {pred_rot_str:<30} {true_trans_str:<30} {pred_trans_str:<30}")
    
    print("="*60)


if __name__ == "__main__":
    main()
