'''
Simple TOML configuration loader for the Drosophila neural model.
'''

from pathlib import Path
import os
import tempfile
from typing import Any, Dict
import torch
import numpy as np
import pickle
import plotting_utilities as pu
import stimulus_utilities as su

try:
    import tomllib
except ImportError:
    import tomli as tomllib

try:
    # Python <3.11 needs a writer; on 3.11+ prefer tomllib for reading and tomli_w for writing
    import tomli_w as toml_writer
except ImportError:  # pragma: no cover - tomli_w present per requirements for <3.11
    toml_writer = None

def load_config(config_path: str = 'config.toml') -> dict:
    '''
    Load configuration by merging default config with user config.
    
    Returns:
        Dictionary containing the merged configuration
    '''
    # Load default config
    default_path = Path('config_default.toml')
    if not default_path.exists():
        raise FileNotFoundError(f'Default configuration file not found: {default_path}')
    
    with open(default_path, 'rb') as f:
        config = tomllib.load(f)
    
    # Check for user config and merge if it exists
    user_path = Path(config_path)
    if user_path.exists():
        with open(user_path, 'rb') as f:
            user_config = tomllib.load(f)
        
        # Recursively merge user config into default config
        config = _merge_configs(config, user_config)
    
    # Convert "none" strings to Python None
    config = _convert_none_strings(config)
    
    return config

def _merge_configs(default: dict, user: dict) -> dict:
    '''
    Recursively merge user config into default config.
    
    Args:
        default: Default configuration dictionary
        user: User configuration dictionary
        
    Returns:
        Merged configuration dictionary
    '''
    result = default.copy()
    
    for key, value in user.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result

def _convert_none_strings(config: dict | list | Any) -> dict | list | Any:
    '''
    Recursively convert "none" strings (case-insensitive) to Python None.
    
    Args:
        config: Configuration value (dict, list, or other type)
        
    Returns:
        Configuration with "none" strings converted to None
    '''
    if isinstance(config, dict):
        return {key: _convert_none_strings(value) for key, value in config.items()}
    elif isinstance(config, list):
        return [_convert_none_strings(item) for item in config]
    elif isinstance(config, str) and config.lower() == "none":
        return None
    else:
        return config

def interpolate_to_model_dt(data, stimulus_dt, model_dt):
    """
    Linearly interpolate movie data from stimulus_dt to model_dt.
    
    Args:
        data: Tensor of shape (batch, time, ...) - can be numpy array or torch tensor
        stimulus_dt: Original time step of the data
        model_dt: Target time step for the model
        
    Returns:
        Interpolated data with new time dimension matching model_dt
        
    Raises:
        ValueError: If model_dt > stimulus_dt (cannot upsample in time)
    """
    if model_dt > stimulus_dt:
        raise ValueError(
            f"model_dt ({model_dt}) cannot be larger than stimulus_dt ({stimulus_dt}). "
            f"Cannot upsample temporal resolution."
        )
    
    # Convert to torch if numpy
    was_numpy = isinstance(data, np.ndarray)
    if was_numpy:
        data = torch.from_numpy(data)
    
    original_shape = data.shape
    batch_size, original_time = original_shape[0], original_shape[1]
    remaining_dims = original_shape[2:]
    
    # Calculate new time dimension
    total_duration = (original_time - 1) * stimulus_dt
    new_time = int(total_duration / model_dt) + 1
    
    # Flatten all dimensions after time for interpolation
    # Shape: (batch, time, *remaining) -> (batch, -1, time) for F.interpolate
    data_flat = data.reshape(batch_size, original_time, -1)  # (batch, time, features)
    data_flat = data_flat.permute(0, 2, 1)  # (batch, features, time)
    
    # Interpolate along time dimension
    data_interp = torch.nn.functional.interpolate(
        data_flat.float(), 
        size=new_time, 
        mode='linear', 
        align_corners=True
    )
    
    # Reshape back: (batch, features, new_time) -> (batch, new_time, *remaining)
    data_interp = data_interp.permute(0, 2, 1)  # (batch, new_time, features)
    new_shape = (batch_size, new_time) + remaining_dims
    data_interp = data_interp.reshape(new_shape)
    
    # Convert back to numpy if input was numpy
    if was_numpy:
        data_interp = data_interp.numpy()
    
    return data_interp


def get_dataset(data_set_type: str, config, column_phi_theta=None):
    device = torch.device(config['model']['device'])
    dtype = getattr(torch, config['model']['dtype'])
    
    model_dt = config['model']['dt']
    stim_time = config['stimulus']['stim_time']  # seconds the sine is moving
    pre_stim_time = config['stimulus']['pre_stim_time']  # seconds sine is stationary before the sine wave starts moving
    post_stim_time = config['stimulus']['post_stim_time']  # seconds sine is stationary after the sine wave finishes moving
    temporal_frequencies = config['stimulus']['temporal_frequencies']
    stimulus_seed = config['stimulus']['seed']

    if data_set_type == 'static_sine':
        if column_phi_theta is None:
            raise Exception('sine waves require column_phi_theta input')
        # --- get a sign wave stimulus ---
        # generate a sine wave on a cylinder around the fly
        stimulus, optic_flow = su.create_balanced_sine_wave(
            column_phi_theta[:, 0], column_phi_theta[:, 1],
            temporal_frequency=temporal_frequencies, dt=model_dt, stim_time=stim_time, pre_stim_time=pre_stim_time, post_stim_time=post_stim_time
        )

    elif data_set_type == 'random_sine':
        if column_phi_theta is None:
            raise Exception('sine waves require column_phi_theta input')

        # --- generate a sine wave with random tf over time
        stimulus, optic_flow = su.create_random_tf_sine_wave(
            column_phi_theta[:, 0], column_phi_theta[:, 1],
            tf_amp=config['stimulus']['tf_amp'], tf_time_scale=config['stimulus']['tf_time_scale'],
            num_repeats=config['stimulus']['num_repeats'], dt=model_dt, stim_time=stim_time, seed=stimulus_seed
        )

    elif data_set_type == 'flow_360':
        # --- grab the natural movie stimulus ---
        stimulus_path = Path('../connectome_data/flow360_data_right_eye_mapped.pkl')
        with open(stimulus_path, 'rb') as f:
            stimulus_in = pickle.load(f)

        # most stimuli are 99 points long. some are 199. shorten so they're all the same
        expected_length = 99
        stimulus_dt = 1 / stimulus_in['framerate']
        for i in reversed(range(len(stimulus_in['responses']))):
            if stimulus_in['responses'][i].shape[0] > expected_length:
                long_run = stimulus_in['responses'].pop(i)
                stimulus_in['responses'].append(long_run[:expected_length])
                stimulus_in['responses'].append(long_run[expected_length:2*expected_length])

                long_run_vect = stimulus_in['forward_flows'].pop(i)
                stimulus_in['forward_flows'].append(long_run_vect[:expected_length])
                stimulus_in['forward_flows'].append(long_run_vect[expected_length:2*expected_length])

        stimulus_out = np.stack(stimulus_in['responses'])
        optic_flow = np.stack(stimulus_in['forward_flows'])
        stimulus_out /= 127.5
        stimulus_out -= 1.0

        # Interpolate to model timestep
        stimulus = interpolate_to_model_dt(stimulus_out, stimulus_dt, model_dt)
        optic_flow = interpolate_to_model_dt(optic_flow, stimulus_dt, model_dt)

        # Add a blank stimulus (all zeros) with zero optic flow
        blank_stim = np.zeros((1, stimulus.shape[1], stimulus.shape[2]))
        blank_flow = np.zeros((1, optic_flow.shape[1], optic_flow.shape[2], optic_flow.shape[3]))
        stimulus = np.concatenate([blank_stim, stimulus], axis=0)
        optic_flow = np.concatenate([blank_flow, optic_flow], axis=0)

    elif data_set_type == 'sintel':
        sintel_path = Path('../connectome_data/sintel_data.pkl')
        with open(sintel_path, 'rb') as f:
            stimulus_in = pickle.load(f)

        expected_length = 49
        stimulus_dt = 1 / stimulus_in['framerate']
        stimulus_list = [torch.tensor(i, device=device, dtype=dtype) for i in stimulus_in['responses'] if i.shape[0] == expected_length]
        optic_flow_list = [torch.tensor(i, device=device, dtype=dtype) for i in stimulus_in['forward_flows'] if i.shape[0] == expected_length]

        stimulus = torch.stack(stimulus_list, axis=0)
        optic_flow = torch.stack(optic_flow_list, axis=0)
        optic_flow[:, :, :, 1] *= -1
        stimulus /= 127.5
        stimulus -= 1.0

        # Interpolate to model timestep
        stimulus = interpolate_to_model_dt(stimulus, stimulus_dt, model_dt)
        optic_flow = interpolate_to_model_dt(optic_flow, stimulus_dt, model_dt)

        # Add a blank stimulus (all zeros) with zero optic flow
        blank_stim = torch.zeros((1, stimulus.shape[1], stimulus.shape[2]), device=device, dtype=dtype)
        blank_flow = torch.zeros((1, optic_flow.shape[1], optic_flow.shape[2], optic_flow.shape[3]), device=device, dtype=dtype)
        stimulus = torch.cat([blank_stim, stimulus], dim=0)
        optic_flow = torch.cat([blank_flow, optic_flow], dim=0)

    # --- plot the stimuli ---
    # if column_phi_theta is not None:
    #     pu.xt_plot(column_phi_theta[:, 0], column_phi_theta[:, 1], optic_flow=optic_flow.detach().cpu().numpy(), save_folder='figures/stimulus', run_tag=seed)

    stimulus = torch.tensor(stimulus, device=device, dtype=dtype)
    optic_flow = torch.tensor(optic_flow, device=device, dtype=dtype)

    return stimulus, optic_flow

def save_config(config: Dict[str, Any], path: os.PathLike | str) -> None:
    '''
    Save a configuration dictionary to a TOML file.

    Args:
        config: Configuration dictionary to be written
        path: Output TOML path (default: 'config.toml')

    The write is performed atomically by writing to a temporary file
    in the destination directory and then replacing the target.
    '''
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Serialize using tomli_w where available. For Python 3.11+, tomllib has no writer,
    # so tomli-w remains the recommended writer.
    if toml_writer is None:
        raise RuntimeError(
            "TOML writer not available. Please install 'tomli-w' for writing TOML files."
        )

    # Write atomically
    dir_fd = os.open(str(output_path.parent), os.O_RDONLY)
    try:
        with tempfile.NamedTemporaryFile('w', delete=False, dir=str(output_path.parent), suffix='.tmp') as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(toml_writer.dumps(config))
            tmp.flush()
            os.fsync(tmp.fileno())
        # Replace target
        os.replace(tmp_path, output_path)
        os.fsync(dir_fd)
    finally:
        try:
            os.close(dir_fd)
        except Exception:
            pass
