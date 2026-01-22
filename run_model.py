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


def run_single_model(seed=None, device_override=None, run_id=None, timestamp=None, model_path=None):
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
    config = du.load_config()
    stimulus_seed = 0

    # Override device if specified
    if device_override is not None:
        config['model']['device'] = device_override

    # Clear previous figures (delete non-hidden files)
    fig_dir = Path('figures')
    fig_dir.mkdir(parents=True, exist_ok=True)
    for p in fig_dir.iterdir():
        if p.is_file() and not p.name.startswith('.'):
            p.unlink()

    # Extract parameters from config
    device = torch.device(config['model']['device'])
    dt = config['model']['dt']

    # --- load in connectivity and cell info data ---
    connectome_folder = Path(config['data']['connectome_folder'])
    cell_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_cell_info.parquet')
    column_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_column_info.parquet')
    synapse_counts_df = pd.read_parquet(connectome_folder / 'optic_lobes_synapse_counts.parquet')
    
    # first generate the information about the columns - their location and ID
    # we'll use the columns that have all our input and output neurons
    input_neurons = ['L1', 'L2', 'L3']  # Convert list to tuple for compatibility
    output_neurons = ['T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d']
    output_neurons_direction = ['right', 'left', 'up', 'down', 'right', 'left', 'up', 'down']

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
        model = ModelClass.load(model_file, device=config['model']['device'], freeze_params=False)
    else:
        model = ModelClass(
            synapse_counts_coo=synapse_counts_df[['pre_matrix_ind', 'post_matrix_ind', 'synapse_counts']].to_numpy(),
            cell_types=cell_info_df['type'].to_numpy(),
            neurotransmitters=cell_info_df['neurotransmitter'].to_numpy(),
            column_indices=cell_info_df['column_index'].to_numpy(),
            column_phi_theta=column_info_df[['phi', 'theta']].to_numpy(),
            nonlinearity=config['model']['nonlinearity'],
            count_threshold=config['model']['synapse_count_threshold'],
            device=config['model']['device'],
            dtype=config['model']['dtype'],
            dt=dt,
            seed=seed,
            commit_hash=commit_hash,
        )

    stimulus, optic_flow = du.get_dataset(config['data']['data_set_type'], config, model.column_phi_theta)

    # specify the target responses you expect for each cell type. this is what we will train on
    target = {}

    # --- make stimulus for random velocity sine wave ---
    # Use torch.relu to create independent copies and avoid in-place modification bugs
    for n, d in zip(output_neurons, output_neurons_direction):
        if d == 'right':
            target[n] = torch.relu(optic_flow[:, :, :, 0])
        elif d == 'left':
            target[n] = torch.relu(-optic_flow[:, :, :, 0])
        elif d == 'up':
            target[n] = torch.relu(optic_flow[:, :, :, 1])
        elif d == 'down':
            target[n] = torch.relu(-optic_flow[:, :, :, 1])

    # Add zero baseline: no stimulus should produce no output
    zero_stimulus = torch.zeros(1, stimulus.shape[1], stimulus.shape[2], dtype=stimulus.dtype, device=stimulus.device)
    stimulus = torch.cat([stimulus, zero_stimulus], dim=0)
    for n in output_neurons:
        zero_target = torch.zeros(1, target[n].shape[1], target[n].shape[2], dtype=target[n].dtype, device=target[n].device)
        target[n] = torch.cat([target[n], zero_target], dim=0)

    # save the config file and inputs
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if run_id is not None:
        model_save_folder = Path(f'models/{timestamp}_{run_id}')
    else:
        model_save_folder = Path('models/' + timestamp)
    model_save_folder.mkdir(parents=True, exist_ok=True)

    # Save seed info to config (filter out None values since TOML doesn't support them)
    run_info = {'seed': seed, 'device': str(device), 'run_id': run_id}
    config['run_info'] = {k: v for k, v in run_info.items() if v is not None}
    du.save_config(config, model_save_folder / 'config.toml')
    torch.save(stimulus.detach().cpu(), model_save_folder / "cell_inputs.pt")
    torch.save({k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in target.items()}, model_save_folder / "target.pt")

    # get the loss function
    loss_fn = getattr(tu, config['training']['loss_fn'])
    model.save(model_save_folder / 'model_init.pt')
    model.save(model_save_folder / 'model.pt')

    # --- train the model ---
    tu.train_model(model, stimulus, target, output_neurons, input_neurons,
        model_save_folder=model_save_folder,
        num_steps=config['training']['num_steps'],
        save_every=config['training']['save_every'],
        alpha=config['training']['alpha'],
        clip_grad_norm=config['training'].get('clip_norm', None),
        reset_init_every=config['training'].get('reset_init_every', 1),
        sigma_lambda=config['training'].get('sigma_lambda', 0),
        peak_lambda=config['training'].get('peak_lambda', 0),
        loss_fn=loss_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a neural model')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default=None, help='Device override (e.g., cuda:0, cuda:1, cpu)')
    parser.add_argument('--run-id', type=str, default=None, help='Optional identifier for this run')
    parser.add_argument('--timestamp', type=str, default=None, help='Timestamp string for model folder (format: YYYYMMDD_HHMMSS)')
    parser.add_argument('--model-path', type=str, default=None, help='Path to existing model folder to load (e.g., models/best_model)')

    args = parser.parse_args()

    run_single_model(seed=args.seed, device_override=args.device, run_id=args.run_id, timestamp=args.timestamp, model_path=args.model_path)

