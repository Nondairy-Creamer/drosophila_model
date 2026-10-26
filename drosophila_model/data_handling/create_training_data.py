"""Bake a fixed stimulus dataset from an on-the-fly generator.

Usage:
    python -m drosophila_model.data_handling.create_training_data <stimulus_type>

Reads ``[model]`` (``dt``, ``dtype``) and the stimulus-generation parameters
(``[data]``: ``config_default_data.toml`` + ``config_data.toml``) from the
config, generates the stimulus on the CPU with the on-the-fly generators in
``stimulus_generators.py`` (always two eyes: ``[right | left]`` columns plus
ego-motion targets), and splits it into a train/test pair saved with a
consistent schema to ``<data_folder>/stimulus/training/<stimulus_type>_train.pt``
and ``..._test.pt``.

The split is by whole source group (``StimulusData.group_id``: e.g. all
repeats of one panorama, or the phase/contrast variants of one grating
condition), never within a group, so correlated examples can't leak across
train/test. The groups are shuffled with ``[data].shared.seed`` and the first
``train_fraction`` of them go to ``_train.pt``, the rest to ``_test.pt``.
``train_fraction`` comes from the type's own ``[data.<stimulus_type>]`` table
when it sets one (e.g. the tiny ``full_field_step`` probe uses 1.0), otherwise
from ``[data].shared``; when every group goes to training no ``_test.pt`` is
written.

The analysis-only types in ``stimulus_generators.ANALYSIS_ONLY_STIMULUS_TYPES``
(``edge``, ``expanding_dot``, ``directional_sine``) are refused.

The ``_train.pt`` file is what ``training_utilities.load_processed_stimulus``
loads for training, so stimuli are not generated on the fly during training.
The ``_test.pt`` file is held out for evaluation (``split='test'``). Analysis
scripts still call the generators directly for variable-length, on-the-fly
stimuli.
"""
import argparse
import copy
import warnings

import numpy as np
import torch

from drosophila_model.config import get_data_folder, load_config
from drosophila_model.data_handling import stimulus_generators as sg
from drosophila_model.data_handling.sampling import load_column_phi_theta


def check_bakeable(stimulus_type):
    """Raise ``ValueError`` unless ``stimulus_type`` can be baked."""
    if stimulus_type in sg.ANALYSIS_ONLY_STIMULUS_TYPES:
        raise ValueError(
            f"'{stimulus_type}' cannot be baked into training data: it is "
            f"{sg.ANALYSIS_ONLY_STIMULUS_TYPES[stimulus_type]}. Bakeable types: "
            f"{', '.join(sg.bakeable_stimulus_types())}."
        )
    if stimulus_type not in sg.bakeable_stimulus_types():
        raise ValueError(
            f"Unknown stimulus_type '{stimulus_type}'. Bakeable types: "
            f"{', '.join(sg.bakeable_stimulus_types())}."
        )


def create_training_data(stimulus_type, config=None, config_path=None,
                         data_config_path=None):
    """Generate ``stimulus_type`` from config and save it to ``stimulus/training``.

    Model parameters and the stimulus-generation parameters both come from the
    config (``config`` / ``config_path``); the stimulus-generation parameters
    live under ``config['data']`` (default + ``data_config_path`` overlay,
    merged by :func:`drosophila_model.config.load_config`).

    Returns ``(train_path, test_path)``; ``test_path`` is ``None`` when every
    source group went to training.
    """
    check_bakeable(stimulus_type)
    if config is None:
        config = load_config(config_path or 'config.toml',
                             data_config_path or 'config_data.toml')

    shared = config['data']['shared']
    type_params = config['data'].get(stimulus_type, {})
    train_fraction = type_params.get('train_fraction', shared['train_fraction'])
    if not 0 < train_fraction <= 1:
        raise ValueError(
            f"train_fraction must be in (0, 1], got {train_fraction}."
        )

    data_folder = get_data_folder(config)
    column_phi_theta = load_column_phi_theta(data_folder)

    # Generate on the CPU: the dataset is saved from there anyway, and baking
    # should not need (or fill) a GPU.
    gen_config = copy.deepcopy(config)
    gen_config['model']['device'] = 'cpu'
    data = sg.get_stimulus(
        stimulus_type, gen_config, column_phi_theta,
        num_eyes=sg.default_num_eyes(stimulus_type),
    )
    stimulus = data.stimulus.detach()
    optic_flow = data.optic_flow.detach()
    train_mask = data.train_mask.detach()
    rotation = data.rotation.detach() if data.rotation is not None else None
    translation = data.translation.detach() if data.translation is not None else None
    group_id = data.group_id.detach()
    eye_offset = data.eye_offset
    num_eyes = data.num_eyes

    stimulus_config = dict(shared)
    stimulus_config.update(type_params)

    # Split by whole source group, never within one, so correlated examples
    # can't leak across train/test. The groups are shuffled (seeded) first so
    # the test set is not one systematic slice of the generation order (e.g.
    # a single direction or temporal frequency).
    groups = np.random.default_rng(shared['seed']).permutation(np.unique(group_id.numpy()))
    num_groups = len(groups)
    num_train_groups = max(1, int(np.round(num_groups * train_fraction)))
    is_train = torch.from_numpy(np.isin(group_id.numpy(), groups[:num_train_groups]))

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
    if num_train_groups < num_groups:
        torch.save(build_payload(~is_train), test_path)
    else:
        if train_fraction < 1:
            warnings.warn(
                f"'{stimulus_type}' has only {num_groups} source group(s); all of "
                f"them went to training, so there is no held-out test split."
            )
        if test_path.exists():
            # A test split from an earlier bake would overlap this training set.
            test_path.unlink()
            print(f"Removed stale {test_path}")
        test_path = None

    num_train = int(is_train.sum())
    print(
        f"Baked '{stimulus_type}' ({num_eyes} eye(s)) from {stimulus.shape[0]} "
        f"examples across {num_groups} source group(s)\n"
        f"  train ({num_train_groups} group(s), {num_train} examples) -> {train_path}\n"
        f"  test  ({num_groups - num_train_groups} group(s), "
        f"{stimulus.shape[0] - num_train} examples) -> {test_path or 'not written'}\n"
        f"  stimulus {tuple(stimulus.shape)}, optic_flow {tuple(optic_flow.shape)}, "
        f"train_mask {tuple(train_mask.shape)}"
    )
    return train_path, test_path


def main():
    parser = argparse.ArgumentParser(
        description='Bake a fixed stimulus dataset from an on-the-fly generator.'
    )
    parser.add_argument(
        'stimulus_type', type=str,
        help=f"Stimulus type to bake: one of {', '.join(sg.bakeable_stimulus_types())}."
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
    try:
        check_bakeable(args.stimulus_type)
    except ValueError as err:
        parser.error(str(err))

    create_training_data(
        args.stimulus_type,
        config_path=args.config_path,
        data_config_path=args.data_config_path,
    )


if __name__ == '__main__':
    main()
