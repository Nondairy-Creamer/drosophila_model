"""Resume training a saved run in place.

Usage (from the repo root):

    python -m drosophila_model.training.continue_training --model-folder models/<run>
    python -m drosophila_model.training.continue_training --model-folder models/<run> --new-session

Reloads the run's ``model.pt``, its saved training data (``cell_inputs.pt``,
``target.pt``) and its config, then keeps training with the same settings as
``run_model``. By default the optimizer state and loss history in
``model_opt.pt`` are restored and the run folder is updated in place.
``--new-session`` instead copies the folder to ``<run>_cont`` and starts a fresh
optimizer and loss history at the run config's ``[training].alpha``.
"""
import argparse
import shutil
from pathlib import Path

import torch

from drosophila_model.config import load_run_config
from drosophila_model.model_definition import neural_models as nm
from drosophila_model.training import training_utilities as tu


def continue_training(model_folder, new_session=False, num_steps=None):
    """Resume training the run in ``model_folder`` (see module docstring)."""
    model_folder = Path(model_folder)
    if new_session:
        new_folder = model_folder.parent / f"{model_folder.name}_cont"
        if new_folder.exists():
            raise FileExistsError(f"{new_folder} already exists; remove it or resume it directly.")
        print(f"New session: copying {model_folder} to {new_folder}")
        shutil.copytree(model_folder, new_folder)
        model_folder = new_folder

    config = load_run_config(model_folder)
    training = config['training']
    model = nm.BaseRnn.load(model_folder / 'model.pt', freeze_params=False)

    cell_inputs = torch.load(model_folder / 'cell_inputs.pt', map_location='cpu', weights_only=False)
    stimulus = cell_inputs['cell_inputs'].to(device=model.device, dtype=model.dtype)
    train_mask = cell_inputs['train_mask'].to(device=model.device)
    target = torch.load(model_folder / 'target.pt', map_location='cpu', weights_only=False)
    target = {k: v.to(device=model.device, dtype=model.dtype) for k, v in target.items()}

    # Eye geometry of the training data, recorded in [run_info] by run_model.
    run_info = config['run_info']
    num_eyes = run_info['num_eyes']
    eye_offset = (tuple(float(x) for x in run_info['eye_offset'])
                  if run_info.get('eye_offset') is not None else None)

    if new_session:
        alpha = training['alpha']
        opt_state_dict = None
        loss_history = None
    else:
        opt_state = torch.load(model_folder / 'model_opt.pt', map_location='cpu', weights_only=False)
        opt_state_dict = opt_state['optimizer_state_dict']
        loss_history = opt_state['loss']
        alpha = opt_state_dict['param_groups'][0]['lr']

    tu.train_model(model, stimulus, target, output_neurons=None, input_neurons=None,
        loss_components=training['loss_components'],
        model_save_folder=model_folder,
        rng=model.rng,
        num_steps=num_steps if num_steps is not None else training['num_steps'],
        save_every=training['save_every'],
        alpha=alpha,
        clip_grad_norm=training['clip_norm'],
        opt_state_dict=opt_state_dict,
        loss_history=loss_history,
        input_noise_std=training['input_noise_std'],
        contrast_std=training['contrast_std'],
        luminance_std=training['luminance_std'],
        batch_size=training['batch_size'],
        train_mask=train_mask,
        num_eyes=num_eyes,
        eye_offset=eye_offset,
    )


def main():
    parser = argparse.ArgumentParser(description='Resume training a saved run')
    parser.add_argument('--model-folder', required=True,
                        help='Run folder to resume (e.g. models/20260114_205655_gpu0_run1)')
    parser.add_argument('--new-session', action='store_true', default=False,
                        help='Copy the folder to <name>_cont and start a fresh optimizer '
                             'and loss history at the config alpha')
    parser.add_argument('--num-steps', type=int, default=None,
                        help="Training steps for this session (default: [training].num_steps)")
    args = parser.parse_args()
    continue_training(args.model_folder, new_session=args.new_session, num_steps=args.num_steps)


if __name__ == '__main__':
    main()
