"""
Evaluate model loss with systematic neuron silencing.

Loads a trained model and its saved stimuli, computes the baseline loss,
then silences every non-empty combination of {L1, L2, L3}, plus grouped
T4 (T4a-d), T5 (T5a-d), and combined T4+T5 silencing.

Usage:
    python analysis/silencing.py <model_folder>
    python analysis/silencing.py models/best_drone --device cpu
"""
import argparse
import sys
from itertools import combinations
from pathlib import Path


import torch
import numpy as np

from drosophila_model.config import load_config
from drosophila_model.model_definition import neural_models as nm
from drosophila_model.training import loss_functions as lf

INPUT_NEURONS = ['L1', 'L2', 'L3']
T4_NEURONS = ['T4a', 'T4b', 'T4c', 'T4d']
T5_NEURONS = ['T5a', 'T5b', 'T5c', 'T5d']


def compute_loss(model, cell_inputs, target, train_mask, config):
    """Compute the training loss for a model on saved data.

    Uses the ``loss_components`` list from *config* to look up and call each
    loss function by name from ``loss_functions``, mirroring the training loop.
    """
    time_mask = train_mask.any(dim=-1)
    loss_start_step = 0
    if not time_mask[:, 0].all():
        loss_start_step = time_mask.all(dim=0).to(torch.int).argmax().item()

    loss_names = config['training'].get('loss_components', ['flow_loss'])

    with torch.no_grad():
        if 'odometry_loss' in loss_names and isinstance(target, dict) and 'rotation' in target:
            N = cell_inputs.shape[2] // 2
            oob = ~train_mask.any(dim=1).any(dim=0)
            pred_rot, pred_trans, right_res, left_res = \
                model.regions['optic_lobe'].estimate_ego_motion_from_image(
                    cell_inputs[:, :, :N], cell_inputs[:, :, N:], oob_columns=oob)
            outputs = {}
            right_optic = right_res['outputs']['optic_lobe']
            left_optic = left_res['outputs']['optic_lobe']
            for key in right_optic:
                if key in ('voltage', 'firing_rate', 'synaptic_current'):
                    continue
                outputs[key] = torch.cat(
                    [right_optic[key], left_optic[key]], dim=2)
            outputs['pred_rotation'] = pred_rot
            outputs['pred_translation'] = pred_trans
        else:
            input_neurons = [n for n in INPUT_NEURONS if n in model.unique_cell_types]
            result = model.predict(cell_inputs, input_neurons=input_neurons)
            outputs = result['outputs']['optic_lobe']

        loss = sum(
            getattr(lf, n)(model, outputs, target, train_mask, loss_start_step)
            for n in loss_names
        ) / len(loss_names)
        return loss.item()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate loss under systematic neuron silencing.",
    )
    parser.add_argument(
        "model_folder", type=str,
        help="Path to model folder (must contain model.pt, cell_inputs.pt, target.pt)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device override (default: from config)",
    )
    parser.add_argument(
        "--num-stimuli", type=int, default=50,
        help="Number of stimuli to randomly sample (default: 50, 0 = use all)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for stimulus sampling (default: 0)",
    )
    args = parser.parse_args()

    model_path = Path(args.model_folder)
    model_folder = model_path.parent if model_path.is_file() else model_path

    local_config = load_config()
    model_config_path = model_folder / 'config.toml'
    if model_config_path.exists():
        model_config = load_config(model_config_path)
    else:
        model_config = local_config

    dtype_str = local_config['model'].get('dtype', 'float32')
    pt_dtype = getattr(torch, dtype_str)

    # --- Load inputs and targets ---
    print(f"Loading model from {model_path} ...")
    model = nm.BaseRnn.load(model_path, device=args.device)
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

    # --- Subsample stimuli ---
    num_stimuli = cell_inputs.shape[0]
    if args.num_stimuli and args.num_stimuli < num_stimuli:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(num_stimuli, size=args.num_stimuli, replace=False)
        idx.sort()
        cell_inputs = cell_inputs[idx]
        train_mask = train_mask[idx]
        if isinstance(target, dict):
            target = {k: (v[idx] if torch.is_tensor(v) else v) for k, v in target.items()}
        else:
            target = target[idx]
        print(f"  Sampled {args.num_stimuli}/{num_stimuli} stimuli (seed={args.seed})")

    print(f"  cell_inputs shape: {cell_inputs.shape}")

    # --- Baseline loss (no silencing) ---
    baseline_loss = compute_loss(model, cell_inputs, target, train_mask, model_config)
    print(f"\nBaseline loss (no silencing): {baseline_loss:.6f}")

    # --- Build silencing experiments ---
    def present(neurons):
        return [n for n in neurons if n in model.unique_cell_types]

    silencing_groups = []

    # All non-empty combinations of L1/L2/L3
    input_neurons = present(INPUT_NEURONS)
    for r in range(1, len(input_neurons) + 1):
        for combo in combinations(input_neurons, r):
            silencing_groups.append(list(combo))

    # T4 as a group, T5 as a group, T4+T5 together
    t4_present = present(T4_NEURONS)
    t5_present = present(T5_NEURONS)
    if t4_present:
        silencing_groups.append(t4_present)
    if t5_present:
        silencing_groups.append(t5_present)
    if t4_present and t5_present:
        silencing_groups.append(t4_present + t5_present)

    if not silencing_groups:
        print("No silenceable neurons found in model cell types.")
        return

    original_output_strength = model.output_strength_log.data.clone()
    original_input_strength = model.input_strength_log.data.clone()

    results = [("none", baseline_loss)]

    for group in silencing_groups:
        label = "+".join(group)

        model.output_strength_log.data.copy_(original_output_strength)
        model.input_strength_log.data.copy_(original_input_strength)
        model.silence_neurons(group)

        loss = compute_loss(model, cell_inputs, target, train_mask, model_config)
        results.append((label, loss))
        print(f"  Silenced {label:30s} -> loss = {loss:.6f}")

    # Restore original weights
    model.output_strength_log.data.copy_(original_output_strength)
    model.input_strength_log.data.copy_(original_input_strength)

    # --- Summary table ---
    max_label = max(len(label) for label, _ in results)
    width = max_label + 30
    print("\n" + "=" * width)
    print(f"{'Silenced':<{max_label}s} {'Loss':>12s} {'Δ Loss':>12s}")
    print("-" * width)
    for label, loss in results:
        delta = loss - baseline_loss
        print(f"{label:<{max_label}s} {loss:>12.6f} {delta:>+12.6f}")
    print("=" * width)


if __name__ == "__main__":
    main()
