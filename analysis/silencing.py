"""
Evaluate model loss with systematic silencing of L1, L2, and L3 neurons.

Loads a trained model and its saved stimuli, computes the baseline loss,
then silences every non-empty combination of {L1, L2, L3} and reports the
resulting loss.

Usage:
    python analysis/silencing.py <model_folder>
    python analysis/silencing.py models/best_drone --device cpu
"""
import argparse
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

import data_utilities as du
import neural_models as nm
from loss_functions import calc_loss_dict, combined_loss, mse_loss, cos_loss
from training_utilities import _compute_odometry_loss

INPUT_NEURONS = ['L1', 'L2', 'L3']


def compute_loss(model, cell_inputs, target, train_mask, config):
    """Compute the training loss for a model on saved data.

    Automatically selects the correct loss path (odometry / decoder / cell-type)
    based on the saved config and target format.
    """
    device = model.device
    dtype = model.dtype

    time_mask = train_mask.any(dim=-1)
    loss_start_step = 0
    if not time_mask[:, 0].all():
        loss_start_step = time_mask.all(dim=0).to(torch.int).argmax().item()

    fit_odometry = config['training'].get('fit_odometry', False)
    loss_fn_name = config['training'].get('loss_fn', 'combined_loss')
    loss_fn_map = {
        'combined_loss': combined_loss,
        'mse_loss': mse_loss,
        'cos_loss': cos_loss,
    }
    loss_fn = loss_fn_map.get(loss_fn_name, combined_loss)

    with torch.no_grad():
        if fit_odometry and isinstance(target, dict) and 'rotation' in target:
            return _compute_odometry_loss(
                model, cell_inputs, target, train_mask, loss_start_step,
            ).item()

        input_neurons = [n for n in INPUT_NEURONS if n in model.unique_cell_types]
        result = model.predict(cell_inputs, input_neurons=input_neurons)
        pred = result['outputs']

        if model.has_decoder:
            pred_dec = pred['decoder'][:, loss_start_step:].clone()
            target_dec = target[:, loss_start_step:].clone()
            invalid = ~train_mask[:, loss_start_step:].unsqueeze(-1).expand_as(pred_dec)
            pred_dec[invalid] = float('nan')
            target_dec[invalid] = float('nan')
            return loss_fn(target_dec, pred_dec).item()

        output_neurons = ['T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d']
        pred_loss = {k: v[:, loss_start_step:] for k, v in pred['firing_rate'].items()
                     if k in output_neurons}
        target_loss = {k: v[:, loss_start_step:] for k, v in target.items()
                       if k in output_neurons}
        return calc_loss_dict(loss_fn, pred_loss, target_loss).item()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate loss under systematic L1/L2/L3 silencing.",
    )
    parser.add_argument(
        "model_folder", type=str,
        help="Path to model folder (must contain model.pt, cell_inputs.pt, target.pt)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device override (default: from config)",
    )
    args = parser.parse_args()

    model_path = Path(args.model_folder)
    model_folder = model_path.parent if model_path.is_file() else model_path

    local_config = du.load_config()
    model_config_path = model_folder / 'config.toml'
    if model_config_path.exists():
        model_config = du.load_config(model_config_path)
    else:
        model_config = local_config

    dtype_str = local_config['model'].get('dtype', 'float32')
    pt_dtype = getattr(torch, dtype_str)

    # --- Load inputs and targets ---
    print(f"Loading model from {model_path} ...")
    model = nm.NeuralModel.load(model_path, device=args.device)
    model.eval()
    device = model.device

    cell_inputs_data = torch.load(model_folder / 'cell_inputs.pt',
                                  map_location=device, weights_only=False)
    if isinstance(cell_inputs_data, dict):
        cell_inputs = cell_inputs_data['cell_inputs'].to(device=device, dtype=model.dtype)
        train_mask = cell_inputs_data['train_mask'].to(device=device)
    else:
        cell_inputs = cell_inputs_data.to(device=device, dtype=model.dtype)
        train_mask = torch.ones(cell_inputs.shape[:3], dtype=torch.bool, device=device)

    target = torch.load(model_folder / 'target.pt',
                        map_location=device, weights_only=False)
    if isinstance(target, dict):
        target = {k: (v.to(device=device, dtype=model.dtype) if torch.is_tensor(v) else v)
                  for k, v in target.items()}
    else:
        target = target.to(device=device, dtype=model.dtype)

    print(f"  cell_inputs shape: {cell_inputs.shape}")

    # --- Baseline loss (no silencing) ---
    baseline_loss = compute_loss(model, cell_inputs, target, train_mask, model_config)
    print(f"\nBaseline loss (no silencing): {baseline_loss:.6f}")

    # --- Silencing combinations ---
    silencing_neurons = [n for n in INPUT_NEURONS if n in model.unique_cell_types]
    if not silencing_neurons:
        print("None of L1, L2, L3 found in model cell types.")
        return

    original_output_strength = model.output_strength_log.data.clone()

    results = [("none", baseline_loss)]

    all_combos = []
    for r in range(1, len(silencing_neurons) + 1):
        all_combos.extend(combinations(silencing_neurons, r))

    for combo in all_combos:
        combo_list = list(combo)
        label = "+".join(combo_list)

        model.output_strength_log.data.copy_(original_output_strength)
        model.silence_neurons(combo_list)

        loss = compute_loss(model, cell_inputs, target, train_mask, model_config)
        results.append((label, loss))
        print(f"  Silenced {label:12s} -> loss = {loss:.6f}")

    # Restore original weights
    model.output_strength_log.data.copy_(original_output_strength)

    # --- Summary table ---
    print("\n" + "=" * 40)
    print(f"{'Silenced':<16s} {'Loss':>12s} {'Δ Loss':>12s}")
    print("-" * 40)
    for label, loss in results:
        delta = loss - baseline_loss
        print(f"{label:<16s} {loss:>12.6f} {delta:>+12.6f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
