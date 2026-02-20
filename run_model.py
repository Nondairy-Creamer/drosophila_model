import torch
import numpy as np
import pandas as pd
from pathlib import Path
import plotting_utilities as pu
import neural_models as nm
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

    # save the config file and inputs
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if run_id is not None:
        model_save_folder = Path(f'models/{timestamp}_{run_id}')
    else:
        model_save_folder = Path('models/' + timestamp)
    model_save_folder.mkdir(parents=True, exist_ok=True)

    # --- load in connectivity and cell info data ---
    connectome_folder = Path(config['data']['connectome_folder'])
    cell_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_cell_info.parquet')
    synapse_counts_df = pd.read_parquet(connectome_folder / 'optic_lobes_synapse_counts.parquet')
    column_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_column_info.parquet')

    # first generate the information about the columns - their location and ID
    # we'll use the columns that have all our input and output neurons
    input_neurons = ['L1', 'L2', 'L3']  # Convert list to tuple for compatibility
    output_neurons = ['T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d']
    output_neurons_direction = ['right', 'left', 'up', 'down', 'right', 'left', 'up', 'down']

    if len(config['model']['use_connectome_subset']) > 0:
        cell_info_df, synapse_counts_df = du.get_connectome_subset(cell_info_df, synapse_counts_df, config['model']['use_connectome_subset'])
        input_neurons = [i for i in input_neurons if i in config['model']['use_connectome_subset']]
        output_neurons_direction = [output_neurons_direction[i] for i in range(len(output_neurons)) if output_neurons[i] in config['model']['use_connectome_subset']]
        output_neurons = [i for i in output_neurons if i in config['model']['use_connectome_subset']]

    # Get model class from config
    model_type = config['model'].get('model_type')
    ModelClass = getattr(nm, model_type)
    
    # Get current git commit hash for provenance tracking
    commit_hash = mu.get_git_commit_hash()
    if commit_hash:
        print(f"Git commit: {commit_hash[:8]}")

    # create or load the model
    if model_path is not None:
        model_file = Path(model_path) / 'model.pt'
        print(f"Loading existing model from {model_file}")
        model = nm.NeuralModel.load(model_file, device=config['model']['device'], freeze_params=False)
        model.add_noise_to_parameters(model.init_std)
    else:
        model = ModelClass(
            synapse_counts_coo=synapse_counts_df[['pre_root_id', 'post_root_id', 'synapse_counts']].to_numpy(),
            root_ids=cell_info_df['root_id'].to_numpy(),
            cell_types=cell_info_df['type'].to_numpy(),
            neurotransmitters=cell_info_df['neurotransmitter'].to_numpy(),
            column_indices=cell_info_df['column_index'].to_numpy(),
            column_phi_theta=column_info_df[['phi', 'theta']].to_numpy(),
            column_p_q=column_info_df[['p', 'q']].to_numpy(),
            nonlinearity=config['model']['nonlinearity'],
            count_threshold=config['model']['synapse_count_threshold'],
            use_layer_norm=config['model']['use_layer_norm'],
            device=config['model']['device'],
            dtype=config['model']['dtype'],
            decoder_type=config['model']['decoder_type'],
            use_input_conv=config['model']['use_input_conv'],
            dt=dt,
            seed=seed,
            commit_hash=commit_hash,
        )

    fit_odometry = config['training'].get('fit_odometry', False)

    if fit_odometry:
        stim_right, stim_left, optic_flow, target_rotation, target_translation = \
            su.get_stimulus_two_eye(config['stimulus']['stimulus_type'], config, model.column_phi_theta)
        stimulus = torch.stack([stim_right, stim_left], dim=1)
        target = {'rotation': target_rotation, 'translation': target_translation}
    else:
        stimulus, optic_flow = su.get_stimulus_one_eye(config['stimulus']['stimulus_type'], config, model.column_phi_theta)

        if config['training']['use_synthetic_data']:
            target, target_model = su.get_synthetic_data(model, stimulus, input_neurons, output_neurons)
            target_model.save(model_save_folder / 'target_model.pt')

        elif config['model']['decoder_type'] != 'none':
            target = optic_flow
        else:
            target = {}
            for n, d in zip(output_neurons, output_neurons_direction):
                if d == 'right':
                    target[n] = torch.relu(optic_flow[:, :, :, 0])
                elif d == 'left':
                    target[n] = torch.relu(-optic_flow[:, :, :, 0])
                elif d == 'up':
                    target[n] = torch.relu(optic_flow[:, :, :, 1])
                elif d == 'down':
                    target[n] = torch.relu(-optic_flow[:, :, :, 1])

    # Save seed info to config (filter out None values since TOML doesn't support them)
    run_info = {'seed': seed, 'device': str(device), 'run_id': run_id}
    config['run_info'] = {k: v for k, v in run_info.items() if v is not None}
    du.save_config(config, model_save_folder / 'config.toml')
    torch.save(stimulus.detach().cpu(), model_save_folder / "cell_inputs.pt")
    if type(target) is dict:
        torch.save({k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in target.items()}, model_save_folder / "target.pt")
    else:
        torch.save(target.detach().cpu(), model_save_folder / "target.pt")

    # get the loss function
    loss_fn = getattr(tu, config['training']['loss_fn'])
    model.save(model_save_folder / 'model_init.pt')
    model.save(model_save_folder / 'model.pt')

    # --- train the model ---
    tu.train_model(model, stimulus, target, output_neurons, input_neurons,
        model_save_folder=model_save_folder,
        rng=model.rng,
        num_steps=config['training']['num_steps'],
        save_every=config['training']['save_every'],
        alpha=config['training']['alpha'],
        clip_grad_norm=config['training'].get('clip_norm', None),
        reset_init_every=config['training'].get('reset_init_every', 1),
        sigma_lambda=config['training'].get('sigma_lambda', 0),
        peak_lambda=config['training'].get('peak_lambda', 0),
        l1_lambda=config['training'].get('l1_lambda', 0),
        loss_fn=loss_fn, 
        input_noise_std=config['training']['input_noise_std'],
        contrast_std=config['training']['contrast_std'],
        luminance_std=config['training']['luminance_std'],
        batch_size=config['training']['batch_size'],
        pre_stim_time=config['stimulus']['pre_stim_time'],
        fit_odometry=fit_odometry,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a neural model')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default=None, help='Device override (e.g., cuda:0, cuda:1, cpu)')
    parser.add_argument('--run-id', type=str, default=None, help='Optional identifier for this run')
    parser.add_argument('--timestamp', type=str, default=None, help='Timestamp string for model folder (format: YYYYMMDD_HHMMSS)')
    parser.add_argument('--model-path', type=str, default=None, help='Path to existing model folder to load (e.g., models/best_model)')
    parser.add_argument('--config-path', type=str, default=None, help='Path to config TOML (default: config.toml)')

    args = parser.parse_args()

    run_single_model(
        seed=args.seed,
        device_override=args.device,
        run_id=args.run_id,
        timestamp=args.timestamp,
        model_path=args.model_path,
        config_path=args.config_path,
    )

