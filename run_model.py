import torch
import numpy as np
import pandas as pd
from pathlib import Path
import plotting_utilities as pu
import neural_models as nm
import connectome_loading as cl
import data_utilities as du
from datetime import datetime
import training_utilities as tu
import stimulus_utilities as su
import model_utilities as mu
import argparse


def run_single_model(seed=None, device_override=None, run_id=None, timestamp=None, model_path=None, config_path=None):
    """
    Run a single model training with specified seed and device.

    Args:
        seed: Random seed for reproducibility. If None, uses current time.
        device_override: Override device from config (e.g., 'cuda:0', 'cuda:1', 'cpu')
        run_id: Optional identifier for this run (appended to timestamp)
        timestamp: Optional timestamp string. If None, generates a new timestamp.
        model_path: Optional path to existing model folder (e.g., 'models/best_model').
                    If provided, loads model.pt from that folder instead of creating a new one.
    """

    # Load configuration from TOML file
    config = du.load_config(config_path or 'config.toml')

    # Override device if specified
    if device_override is not None:
        config['model']['device'] = device_override

    # Clear previous figures (delete non-hidden files)
    fig_dir = Path('figures')
    fig_dir.mkdir(parents=True, exist_ok=True)
    for p in fig_dir.iterdir():
        if p.is_file() and not p.name.startswith('.'):
            p.unlink(missing_ok=True)  # another parallel process may have already deleted it

    # Extract parameters from config
    device = torch.device(config['model']['device'])
    dt = config['model']['dt']
    loss_components = config['training']['loss_components']
    uses_two_eyes = 'odometry_loss' in loss_components

    # save the config file and inputs
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if run_id is not None:
        model_save_folder = Path(f'models/{timestamp}_{run_id}')
    else:
        model_save_folder = Path('models/' + timestamp)
    model_save_folder.mkdir(parents=True, exist_ok=True)

    # --- pick input/output neurons for the configured connectome subset ---
    input_neurons = ['L1', 'L2', 'L3']
    output_neurons = ['T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d']
    output_neurons_direction = ['right', 'left', 'up', 'down', 'right', 'left', 'up', 'down']

    if len(config['model']['use_connectome_subset']) > 0:
        # Connectome subset: load the parquet bundle, filter via DataFrame, then
        # wrap into a standard ConnectomeData. (load_connectome doesn't know
        # about subsets.)
        connectome_folder = Path(config['data']['connectome_folder'])
        cell_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_cell_info.parquet')
        synapse_counts_df = pd.read_parquet(connectome_folder / 'optic_lobes_synapse_counts.parquet')
        column_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_column_info.parquet')

        cell_info_df, synapse_counts_df = du.get_connectome_subset(
            cell_info_df, synapse_counts_df, config['model']['use_connectome_subset']
        )
        input_neurons = [i for i in input_neurons if i in config['model']['use_connectome_subset']]
        output_neurons_direction = [
            output_neurons_direction[i] for i in range(len(output_neurons))
            if output_neurons[i] in config['model']['use_connectome_subset']
        ]
        output_neurons = [i for i in output_neurons if i in config['model']['use_connectome_subset']]

        connectome = cl.load_connectome_from_dataframes(
            cell_info_df, column_info_df, synapse_counts_df,
            name=config['model'].get('connectome', 'optic_lobe_right'),
            is_single_eye=True,
        )
    else:
        connectome = cl.load_connectome(
            config['model'].get('connectome', 'optic_lobe_right'),
            connectome_folder=config['data']['connectome_folder'],
        )

    # Get model class from config: compose dynamics + anatomy at runtime.
    ModelClass = nm.compose_model_class(
        dynamics_type=config['model']['dynamics_type'],
        model_type=config['model']['model_type'],
    )
    
    # Get current git commit hash for provenance tracking
    commit_hash = mu.get_git_commit_hash()
    if commit_hash:
        print(f"Git commit: {commit_hash[:8]}")

    # create or load the model
    if model_path is not None:
        model_file = Path(model_path) / 'model.pt'
        print(f"Loading existing model from {model_file}")
        model = nm.BaseRnn.load(model_file, device=str(device), freeze_params=False)
        model.add_noise_to_parameters(model.init_std * 0.1)
        if model.dt != dt:
            print(f"Using model dt={model.dt} (config had dt={dt})")
            dt = model.dt
            config['model']['dt'] = dt
    else:
        model = ModelClass(
            connectome=connectome,
            nonlinearity=config['model']['nonlinearity'],
            count_threshold=config['model']['synapse_count_threshold'],
            use_layer_norm=config['model']['use_layer_norm'],
            device=config['model']['device'],
            dtype=config['model']['dtype'],
            readouts=config['model']['readouts'],
            readout_neurons=config['model'].get('readout_neurons'),
            regularizer_parameters=config['model'].get('regularizer_parameters'),
            use_input_conv=config['model']['use_input_conv'],
            dt=dt,
            seed=seed,
            commit_hash=commit_hash,
        )

    # --- Build target dict ---
    if uses_two_eyes:
        stimulus, optic_flow, target_rotation, target_translation, train_mask = \
            su.get_stimulus_two_eyes(config['stimulus']['stimulus_type'], config, model.column_phi_theta)
        target = {'rotation': target_rotation, 'translation': target_translation, 'optic_flow': optic_flow}
    else:
        stimulus, optic_flow, train_mask = su.get_stimulus_one_eye(config['stimulus']['stimulus_type'], config, model.column_phi_theta)

        if config['training'].get('use_synthetic_data', False):
            target, target_model = su.get_synthetic_data(model, stimulus, input_neurons, output_neurons)
            target_model.save(model_save_folder / 'target_model.pt')
        else:
            target = {'optic_flow': optic_flow}

    # Save seed info to config (filter out None values since TOML doesn't support them)
    run_info = {'seed': seed, 'device': str(device), 'run_id': run_id}
    config['run_info'] = {k: v for k, v in run_info.items() if v is not None}
    du.save_config(config, model_save_folder / 'config.toml')
    torch.save({'cell_inputs': stimulus.detach().cpu(), 'train_mask': train_mask.detach().cpu()},
               model_save_folder / "cell_inputs.pt")
    if type(target) is dict:
        torch.save({k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in target.items()}, model_save_folder / "target.pt")
    else:
        torch.save(target.detach().cpu(), model_save_folder / "target.pt")

    model.save(model_save_folder / 'model_init.pt')
    model.save(model_save_folder / 'model.pt')

    # --- train the model ---
    tu.train_model(model, stimulus, target, output_neurons, input_neurons,
        loss_components=loss_components,
        model_save_folder=model_save_folder,
        rng=model.rng,
        num_steps=config['training']['num_steps'],
        save_every=config['training']['save_every'],
        alpha=config['training']['alpha'],
        clip_grad_norm=config['training'].get('clip_norm', None),
        input_noise_std=config['training']['input_noise_std'],
        contrast_std=config['training']['contrast_std'],
        luminance_std=config['training']['luminance_std'],
        batch_size=config['training']['batch_size'],
        train_mask=train_mask,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a neural model')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default=None, help='Device override (e.g., cuda:0, cuda:1, cpu)')
    parser.add_argument('--run-id', type=str, default=None, help='Optional identifier for this run')
    parser.add_argument('--timestamp', type=str, default=None, help='Tag or timestamp string for model folder naming')
    parser.add_argument('--model-folder', type=str, default=None, help='Path to existing model folder to load (e.g., models/best_model)')
    parser.add_argument('--config-path', type=str, default=None, help='Path to config TOML (default: config.toml)')

    args = parser.parse_args()

    run_single_model(
        seed=args.seed,
        device_override=args.device,
        run_id=args.run_id,
        timestamp=args.timestamp,
        model_path=args.model_folder,
        config_path=args.config_path,
    )
