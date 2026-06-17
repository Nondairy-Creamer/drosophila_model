"""Bake a fixed stimulus dataset from an on-the-fly generator.

Usage:
    python drosophila_model/data_handling/create_training_data.py <stimulus_type>

Reads parameters from ``config.toml`` (``[model]``, ``[stimulus]``, ``[data]``),
generates the stimulus with the on-the-fly generators in
``stimulus_generators.py`` (two eyes when the type supports it, otherwise one
eye), and saves a single ``.pt`` with a consistent schema to
``<data_folder>/stimulus/training/<stimulus_type>.pt``.

The training file is what ``training_utilities.load_processed_stimulus`` loads, so
stimuli are no longer generated on the fly during training. Analysis scripts
still call the generators directly for variable-length, on-the-fly stimuli.
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

from drosophila_model.config import load_config, get_data_config
from drosophila_model.data_handling import stimulus_generators as sg


def load_column_phi_theta(data_folder):
    """Load right-eye column (phi, theta) without building the full model.

    Reads ``connectome_data/baked/optic_lobe_columns.parquet`` (the same source
    the analysis scripts use), so baking does not require loading the
    connectome.
    """
    col_path = Path(data_folder) / 'connectome_data' / 'baked' / 'optic_lobe_columns.parquet'
    if not col_path.exists():
        raise FileNotFoundError(
            f"Optic-lobe column file not found: {col_path}. "
            "It is produced when the connectome is baked."
        )
    column_info_df = pd.read_parquet(col_path)
    return column_info_df[['phi', 'theta']].to_numpy()


def create_training_data(stimulus_type, config=None, config_path=None,
                         data_config=None, data_config_path=None):
    """Generate ``stimulus_type`` from config and save it to ``stimulus/training``.

    Model parameters come from the main config (``config`` / ``config_path``);
    stimulus-generation parameters come from the data config (``data_config`` /
    ``data_config_path``, see :func:`drosophila_model.config.get_data_config`).

    Returns the path of the written ``.pt`` file.
    """
    if config is None:
        config = load_config(config_path or 'config.toml')
    if data_config is None:
        data_config = get_data_config(data_config_path or 'config_data.toml')

    data_folder = Path(config['data']['data_folder'])
    column_phi_theta = load_column_phi_theta(data_folder)

    two_eyes = stimulus_type in sg.TWO_EYE_STIMULUS_TYPES
    if two_eyes:
        stimulus, optic_flow, rotation, translation, train_mask = \
            sg.get_stimulus_two_eyes(stimulus_type, config, column_phi_theta,
                                     data_config=data_config)
        eye_offset = sg.camera_offset_for_stimulus(stimulus_type)
        num_eyes = 2
    else:
        stimulus, optic_flow, train_mask = \
            sg.get_stimulus_one_eye(stimulus_type, config, column_phi_theta,
                                    data_config=data_config)
        rotation = None
        translation = None
        eye_offset = None
        num_eyes = 1

    stimulus_config = dict(data_config['shared'])
    stimulus_config.update(data_config.get(stimulus_type, {}))

    training_dir = data_folder / 'stimulus' / 'training'
    training_dir.mkdir(parents=True, exist_ok=True)
    out_path = training_dir / f'{stimulus_type}.pt'

    payload = {
        'stimulus_type': stimulus_type,
        'num_eyes': num_eyes,
        'eye_offset': eye_offset,
        'model_dt': config['model']['dt'],
        'stimulus': stimulus.detach().cpu(),
        'optic_flow': optic_flow.detach().cpu(),
        'train_mask': train_mask.detach().cpu(),
        'rotation': rotation.detach().cpu() if rotation is not None else None,
        'translation': translation.detach().cpu() if translation is not None else None,
        'stimulus_config': stimulus_config,
    }
    torch.save(payload, out_path)

    print(
        f"Baked '{stimulus_type}' ({num_eyes} eye(s)) -> {out_path}\n"
        f"  stimulus {tuple(stimulus.shape)}, optic_flow {tuple(optic_flow.shape)}, "
        f"train_mask {tuple(train_mask.shape)}"
    )
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Bake a fixed stimulus dataset from an on-the-fly generator.'
    )
    parser.add_argument(
        'stimulus_type', type=str,
        help='Stimulus type / generator name to bake (e.g. random_pano, edge, flyview).'
    )
    parser.add_argument(
        '--config-path', type=str, default=None,
        help='Path to config TOML (default: config.toml)'
    )
    parser.add_argument(
        '--data-config-path', type=str, default=None,
        help='Path to data config TOML (default: config_data.toml)'
    )
    args = parser.parse_args()

    create_training_data(
        args.stimulus_type,
        config_path=args.config_path,
        data_config_path=args.data_config_path,
    )
