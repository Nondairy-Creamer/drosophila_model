"""Bake a fixed stimulus dataset from an on-the-fly generator.

Usage:
    python drosophila_model/data_handling/create_training_data.py <stimulus_type>

Reads parameters from ``config.toml`` (``[model]``, ``[stimulus]``, ``[data]``),
generates the stimulus with the on-the-fly generators in
``stimulus_generators.py`` (two eyes when the type supports it, otherwise one
eye), and splits it into a train/test pair saved with a consistent schema to
``<data_folder>/stimulus/training/<stimulus_type>_train.pt`` and
``..._test.pt``. The split is by whole source group (``StimulusData.group_id``,
e.g. the raw fly-view clip a segment was cut from), never within a group, so
correlated examples can't leak across train/test. The first
``[data].shared.train_fraction`` of groups go to ``_train.pt``, the rest to
``_test.pt``. Generators whose examples are independent give each example its
own group, so this reduces to a per-example split for them.

The ``_train.pt`` file is what ``training_utilities.load_processed_stimulus``
loads, so stimuli are no longer generated on the fly during training. The
``_test.pt`` file is held out for later analysis and is never read during
training. Analysis scripts still call the generators directly for
variable-length, on-the-fly stimuli.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from drosophila_model.config import load_config
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
                         data_config_path=None):
    """Generate ``stimulus_type`` from config and save it to ``stimulus/training``.

    Model parameters and the stimulus-generation parameters both come from the
    config (``config`` / ``config_path``); the stimulus-generation parameters
    live under ``config['data']`` (default + ``data_config_path`` overlay,
    merged by :func:`drosophila_model.config.load_config`).

    Returns the path of the written ``.pt`` file.
    """
    if config is None:
        config = load_config(config_path or 'config.toml',
                             data_config_path or 'config_data.toml')

    train_fraction = config['data']['shared']['train_fraction']
    if not 0 < train_fraction <= 1:
        raise ValueError(
            f"train_fraction must be in (0, 1], got {train_fraction}."
        )

    data_folder = Path(config['data']['data_folder'])
    column_phi_theta = load_column_phi_theta(data_folder)

    data = sg.get_stimulus(
        stimulus_type, config, column_phi_theta,
        num_eyes=sg.default_num_eyes(stimulus_type),
    )
    stimulus = data.stimulus.detach().cpu()
    optic_flow = data.optic_flow.detach().cpu()
    train_mask = data.train_mask.detach().cpu()
    rotation = data.rotation.detach().cpu() if data.rotation is not None else None
    translation = data.translation.detach().cpu() if data.translation is not None else None
    group_id = data.group_id.detach().cpu()
    eye_offset = data.eye_offset
    num_eyes = data.num_eyes

    stimulus_config = dict(config['data']['shared'])
    stimulus_config.update(config['data'].get(stimulus_type, {}))

    # Split by whole source group (e.g. raw fly-view clip), never within one, so
    # correlated examples can't leak across train/test. The first
    # `train_fraction` of groups (in first-appearance order) go to train, the
    # rest to test. Generators that produce independent examples give each
    # example its own group, so this reduces to a per-example split for them.
    groups_in_order = pd.unique(group_id.numpy())
    num_groups = len(groups_in_order)
    num_train_groups = max(1, int(np.round(num_groups * train_fraction)))
    train_groups = set(groups_in_order[:num_train_groups].tolist())
    is_train = torch.tensor([g in train_groups for g in group_id.tolist()],
                            dtype=torch.bool)

    def build_payload(sel):
        return {
            'stimulus_type': stimulus_type,
            'num_eyes': num_eyes,
            'eye_offset': eye_offset,
            'model_dt': config['model']['dt'],
            'train_fraction': train_fraction,
            'stimulus': stimulus[sel],
            'optic_flow': optic_flow[sel],
            'train_mask': train_mask[sel],
            'rotation': rotation[sel] if rotation is not None else None,
            'translation': translation[sel] if translation is not None else None,
            'group_id': group_id[sel],
            'stimulus_config': stimulus_config,
        }

    training_dir = data_folder / 'stimulus' / 'training'
    training_dir.mkdir(parents=True, exist_ok=True)

    train_path = training_dir / f'{stimulus_type}_train.pt'
    test_path = training_dir / f'{stimulus_type}_test.pt'
    torch.save(build_payload(is_train), train_path)
    torch.save(build_payload(~is_train), test_path)

    num_train = int(is_train.sum())
    print(
        f"Baked '{stimulus_type}' ({num_eyes} eye(s)) from {stimulus.shape[0]} "
        f"examples across {num_groups} source group(s)\n"
        f"  train ({num_train_groups} group(s), {num_train} examples) -> {train_path}\n"
        f"  test  ({num_groups - num_train_groups} group(s), "
        f"{stimulus.shape[0] - num_train} examples) -> {test_path}\n"
        f"  stimulus {tuple(stimulus.shape)}, optic_flow {tuple(optic_flow.shape)}, "
        f"train_mask {tuple(train_mask.shape)}"
    )
    return train_path, test_path


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
