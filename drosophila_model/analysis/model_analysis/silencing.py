"""
Evaluate model loss with systematic neuron silencing.

Loads a trained model and its saved training stimuli (``cell_inputs.pt`` /
``target.pt``, i.e. the training split, not held-out data), computes the
baseline training loss, then silences every non-empty combination of
{L1, L2, L3}, plus grouped T4 (T4a-d), T5 (T5a-d), and combined T4+T5
silencing.

Usage:
    python -m drosophila_model.analysis.model_analysis.silencing models/<model>/
    python -m drosophila_model.analysis.model_analysis.silencing models/<model>/ --device cpu
"""
import argparse
from itertools import combinations
from pathlib import Path


import torch
import numpy as np

from drosophila_model.config import load_run_config
from drosophila_model.model_definition import neural_models as nm
from drosophila_model.training import training_utilities as tu
from drosophila_model.analysis.model_analysis.analysis_utilities import run_eye_offset

INPUT_NEURONS = ['L1', 'L2', 'L3']
T4_NEURONS = ['T4a', 'T4b', 'T4c', 'T4d']
T5_NEURONS = ['T5a', 'T5b', 'T5c', 'T5d']


def compute_loss(model, cell_inputs, target, train_mask, loss_components,
                 num_eyes, eye_offset):
    """Training loss of ``model`` on saved training data.

    Runs the model exactly as ``fit`` does (the two-eye ego-motion
    path, solved with the run's ``eye_offset``, for two-eye data) and sums the
    ``loss_components``. The model is in eval mode, so training-only
    regularizers are not included.
    """
    with torch.no_grad():
        outputs, _ = tu.model_outputs(model, cell_inputs, train_mask, num_eyes,
                                      eye_offset=eye_offset)
        components = tu.compute_loss_components(
            model, outputs, target, train_mask, loss_components)
    return float(sum(components.values()))


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
    run_config = load_run_config(model_folder)

    # --- Load inputs and targets ---
    print(f"Loading model from {model_path} ...")
    model = nm.BaseRnn.load(model_path, device=args.device)
    model.eval()
    device = model.device

    cell_inputs_data = torch.load(model_folder / 'cell_inputs.pt',
                                  map_location=device, weights_only=False)
    cell_inputs = cell_inputs_data['cell_inputs'].to(device=device, dtype=model.dtype)
    train_mask = cell_inputs_data['train_mask'].to(device=device)

    target = torch.load(model_folder / 'target.pt',
                        map_location=device, weights_only=False)
    target = {k: v.to(device=device, dtype=model.dtype) for k, v in target.items()}

    num_eyes = run_config['run_info']['num_eyes']
    eye_offset = run_eye_offset(run_config)
    loss_components = run_config['training']['loss_components']

    # --- Subsample stimuli ---
    num_stimuli = cell_inputs.shape[0]
    if args.num_stimuli and args.num_stimuli < num_stimuli:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(num_stimuli, size=args.num_stimuli, replace=False)
        idx.sort()
        cell_inputs = cell_inputs[idx]
        train_mask = train_mask[idx]
        target = {k: v[idx] for k, v in target.items()}
        print(f"  Sampled {args.num_stimuli}/{num_stimuli} stimuli (seed={args.seed})")

    print(f"  cell_inputs shape: {cell_inputs.shape}")

    def loss_of(mdl):
        return compute_loss(mdl, cell_inputs, target, train_mask, loss_components,
                            num_eyes, eye_offset)

    # --- Baseline loss (no silencing) ---
    baseline_loss = loss_of(model)
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

    results = [("none", baseline_loss)]

    for group in silencing_groups:
        label = "+".join(group)
        with model.silenced(group):
            loss = loss_of(model)
        results.append((label, loss))
        print(f"  Silenced {label:30s} -> loss = {loss:.6f}")

    # --- Summary table ---
    max_label = max(len(label) for label, _ in results)
    width = max_label + 30
    print("\n" + "=" * width)
    print(f"{'Silenced':<{max_label}s} {'Loss':>12s} {'Delta Loss':>12s}")
    print("-" * width)
    for label, loss in results:
        delta = loss - baseline_loss
        print(f"{label:<{max_label}s} {loss:>12.6f} {delta:>+12.6f}")
    print("=" * width)


if __name__ == "__main__":
    main()
