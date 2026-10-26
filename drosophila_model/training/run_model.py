"""Train one connectome-constrained model from config.toml.

Usage (from the repo root):

    python -m drosophila_model.training.run_model --seed 42 --device cuda:0

The model, its config, the training stimulus/targets and checkpoints are
written to ``models/<timestamp>[_<run_id>]/``. The training data must be
baked first with ``drosophila_model.data_handling.create_training_data``.
Only optic-lobe models are trained here; antenna-lobe models are built by the
olfactory analysis scripts.
"""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from drosophila_model.config import get_data_folder, load_config, load_run_config, save_config
from drosophila_model.model_definition import connectome_loading as cl
from drosophila_model.model_definition import neural_models as nm
from drosophila_model.model_definition import regions as regions_mod
from drosophila_model.training import training_utilities as tu


def build_connectome(config):
    """Load the connectome slice described by ``config['model']``.

    Applies ``[model].use_connectome_subset`` (a cell-type allow-list) when it
    is non-empty, then the per-region neuropil/side filters.
    """
    data_folder = get_data_folder(config)
    region_specs = regions_mod.build_region_specs(config['model'].get('regions', {}))
    subset = config['model']['use_connectome_subset']
    if len(subset) > 0:
        baked = data_folder / 'connectome_data' / 'baked'
        cell_info_df = pd.read_parquet(baked / 'cell_info.parquet')
        synapses_df = pd.read_parquet(baked / 'synapses.parquet')
        cell_info_df, synapses_df = cl.get_connectome_subset(cell_info_df, synapses_df, subset)
        return cl.load_connectome_from_dataframes(
            cell_info_df, synapses_df,
            region_specs=region_specs,
            data_folder=str(data_folder),
            count_threshold=config['model']['synapse_count_threshold'],
        )
    return cl.load_connectome(
        region_specs=region_specs,
        data_folder=str(data_folder),
        count_threshold=config['model']['synapse_count_threshold'],
    )


def run_single_model(seed=None, device_override=None, run_id=None, timestamp=None, model_path=None, config_path=None):
    """
    Run a single model training with specified seed and device.

    Args:
        seed: Random seed for reproducibility. If None, falls back to
            config['training']['base_seed'].
        device_override: Override device from config (e.g., 'cuda:0', 'cuda:1', 'cpu')
        run_id: Optional identifier for this run (appended to timestamp)
        timestamp: Optional timestamp string. If None, generates a new timestamp.
        model_path: Optional path to existing model folder (e.g., 'models/best_model').
                    If provided, warm-starts from that folder's model.pt (plus
                    small parameter noise drawn from ``seed``) instead of
                    creating a new model, and the new run's ``[model]`` config
                    is taken from that folder's config.toml. Training still
                    writes a new run folder.
        config_path: Path to the config TOML (default: config.toml).
    """

    # Load configuration from TOML file
    config = load_config(config_path or 'config.toml')

    # Fall back to the config's base_seed when no seed was passed on the CLI.
    if seed is None:
        seed = config['training'].get('base_seed')

    # A warm start trains the loaded model, so record that model's [model]
    # table (keeping this machine's device).
    if model_path is not None and (Path(model_path) / 'config.toml').exists():
        device_name = config['model']['device']
        config['model'] = load_run_config(model_path)['model']
        config['model']['device'] = device_name

    # Override device if specified
    if device_override is not None:
        config['model']['device'] = device_override

    other_regions = sorted(set(config['model'].get('regions', {})) - {'optic_lobe'})
    if other_regions:
        raise ValueError(
            f"run_model trains optic-lobe models only, but [model.regions] also "
            f"configures {other_regions}. Antenna-lobe models are built by "
            "analysis/model_analysis/antenna_lobe_decoding.py and olf_decoding_mimic.py."
        )

    device = torch.device(config['model']['device'])
    loss_components = config['training']['loss_components']
    tu.check_loss_components(loss_components)

    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f'{timestamp}_{run_id}' if run_id is not None else timestamp
    model_save_folder = Path('models') / folder_name
    if model_save_folder.exists() and any(model_save_folder.iterdir()):
        raise FileExistsError(
            f"Run folder {model_save_folder} already exists and is not empty; "
            "pass a different --timestamp/--run-id."
        )

    # Get current git commit hash for provenance tracking
    commit_hash = tu.get_git_commit_hash()
    if commit_hash:
        print(f"Git commit: {commit_hash[:8]}")

    model = None
    if model_path is not None:
        model_file = Path(model_path) / 'model.pt'
        print(f"Loading existing model from {model_file}")
        model = nm.BaseRnn.load(model_file, device=str(device), freeze_params=False)
        # Reseed so runs warm-started from the same model differ (parameter
        # noise, augmentation and batch order all draw from model.rng).
        model.rng = np.random.default_rng(seed)
        model.init_inputs['seed'] = seed
        model.commit_hash = commit_hash
        model.init_inputs['commit_hash'] = commit_hash
        model.add_noise_to_parameters(model.init_std * 0.1)
        if model.dt != config['model']['dt']:
            print(f"Using model dt={model.dt} (config had dt={config['model']['dt']})")
            config['model']['dt'] = model.dt
    dt = config['model']['dt']

    # --- Load the pre-baked stimulus dataset (no on-the-fly generation) ---
    # Bake with: python -m drosophila_model.data_handling.create_training_data <type>
    baked = tu.load_processed_stimulus(
        config, device=device, dtype=getattr(torch, config['model']['dtype']))
    stimulus = baked['stimulus']
    train_mask = baked['train_mask']
    num_eyes = baked['num_eyes']
    # Camera offset is a property of the rig that generated this dataset
    # (0 for single-viewpoint renders, the physical half-baseline for a
    # two-camera rig). Feed it to the two-eye ego-motion solver.
    eye_offset = baked['eye_offset']
    tu.check_loss_components(loss_components, num_eyes)

    if model is None:
        # Compose ConnectomeModel + the configured dynamics mixin at runtime.
        ModelClass = nm.compose_model_class(
            dynamics_type=config['model']['dynamics_type'],
            model_type=config['model']['model_type'],
        )
        model = ModelClass(
            connectome=build_connectome(config),
            nonlinearity=config['model']['nonlinearity'],
            count_threshold=config['model']['synapse_count_threshold'],
            use_layer_norm=config['model']['use_layer_norm'],
            device=config['model']['device'],
            dtype=config['model']['dtype'],
            region_kwargs=config['model'].get('regions', {}),
            dt=dt,
            seed=seed,
            commit_hash=commit_hash,
            **nm.dynamics_kwargs(config['model']),
        )

    num_columns = model.regions['optic_lobe'].num_columns
    if stimulus.shape[2] != num_eyes * num_columns:
        raise ValueError(
            f"Baked stimulus has {stimulus.shape[2]} columns but the model expects "
            f"{num_eyes} eye(s) x {num_columns} optic-lobe columns. Re-bake the "
            "training data with the current connectome."
        )

    target = {'optic_flow': baked['optic_flow']}
    if num_eyes == 2:
        target['rotation'] = baked['rotation']
        target['translation'] = baked['translation']

    # Record run provenance (TOML has no null, so drop None values). The eye
    # geometry is saved so analysis can replay ego-motion with the same rig.
    run_info = {
        'seed': seed, 'device': str(device), 'run_id': run_id, 'num_eyes': num_eyes,
        'eye_offset': list(eye_offset) if eye_offset is not None else None,
    }
    config['run_info'] = {k: v for k, v in run_info.items() if v is not None}
    model_save_folder.mkdir(parents=True, exist_ok=True)
    save_config(config, model_save_folder / 'config.toml')
    torch.save({'cell_inputs': stimulus.detach().cpu(), 'train_mask': train_mask.detach().cpu()},
               model_save_folder / "cell_inputs.pt")
    torch.save({k: v.detach().cpu() for k, v in target.items()}, model_save_folder / "target.pt")

    model.save(model_save_folder / 'model_init.pt')
    model.save(model_save_folder / 'model.pt')

    # --- train the model ---
    # input/output neurons come from each region's defaults or its
    # [model.regions.<name>] input_neurons / output_neurons overrides.
    tu.train_model(model, stimulus, target, output_neurons=None, input_neurons=None,
        loss_components=loss_components,
        model_save_folder=model_save_folder,
        rng=model.rng,
        num_steps=config['training']['num_steps'],
        save_every=config['training']['save_every'],
        alpha=config['training']['alpha'],
        clip_grad_norm=config['training']['clip_norm'],
        input_noise_std=config['training']['input_noise_std'],
        contrast_std=config['training']['contrast_std'],
        luminance_std=config['training']['luminance_std'],
        batch_size=config['training']['batch_size'],
        train_mask=train_mask,
        num_eyes=num_eyes,
        eye_offset=eye_offset,
    )


def main():
    parser = argparse.ArgumentParser(description='Train a neural model')
    parser.add_argument('--seed', type=int, default=None, help="Random seed for reproducibility. If omitted, uses config['training']['base_seed'].")
    parser.add_argument('--device', type=str, default=None, help='Device override (e.g., cuda:0, cuda:1, cpu)')
    parser.add_argument('--run-id', type=str, default=None, help='Optional identifier for this run')
    parser.add_argument('--timestamp', type=str, default=None, help='Tag or timestamp string for model folder naming')
    parser.add_argument('--model-folder', type=str, default=None, help='Warm-start from this model folder (e.g., models/best_model)')
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


if __name__ == '__main__':
    main()
