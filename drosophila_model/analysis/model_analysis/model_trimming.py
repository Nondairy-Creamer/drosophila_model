"""
Model trimming: binary search for the most cell types that can be removed before loss degrades.

Importance score = incoming_synapses * outgoing_synapses * input_strength * output_strength

Cell types are ranked by importance. We binary search over the count of
least-important cell types to remove, finding the maximum number removable
before the training loss exceeds ``LOSS_THRESHOLD`` x the original loss. Every
candidate is built exactly like the saved model: the cell types are removed
structurally (``model.remove_cell_types``), which keeps the original model's
initial state (``steady_state``) for the retained neurons. Losses are the
training objective evaluated on the model's own training data
(``cell_inputs.pt`` / ``target.pt``).

The trimmed model is saved, with the run's config.toml, cell_inputs.pt and
target.pt, to ``<model_folder>_trimmed/`` (or ``--output``).

Usage:
    python -m drosophila_model.analysis.model_analysis.model_trimming models/<model>/
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

from drosophila_model.config import load_run_config
from drosophila_model.model_definition import neural_models as nm
from drosophila_model.analysis.model_analysis import analysis_utilities as au
from drosophila_model.analysis.model_analysis.silencing import compute_loss

# Stop when loss increases to 110% of starting value (10% worse)
LOSS_THRESHOLD = 1.10
# Cell types that are never removed.
PROTECTED_TYPES = {'L1', 'L2', 'L3', 'T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d'}
# Files of the run folder copied next to the trimmed model.pt.
RUN_FILES = ('config.toml', 'cell_inputs.pt', 'target.pt')


def importance_order(model):
    """Cell-type indices sorted from least to most important."""
    # Get average connections matrix (num_cell_types, num_cell_types)
    # ave_con[post_type, pre_type] = average synapse count from pre_type to post_type
    ave_con = au.compute_average_connections(model)

    # Get input and output strengths per cell type (already at cell type level)
    # Dynamics without per-cell-type gains (CellAnn, Arbitrary) are ranked on
    # connectivity alone.
    if hasattr(model, 'input_strength_log'):
        input_strength = model.input_strength_log.exp().detach().cpu().numpy()  # (num_cell_types,) or (num_transmitters, num_cell_types)
        output_strength = model.output_strength_log.exp().detach().cpu().numpy()  # (num_cell_types,)
    else:
        input_strength = np.ones(model.num_cell_types)
        output_strength = np.ones(model.num_cell_types)

    # ConductanceDynamics stores input strength per (transmitter, cell type).
    # Collapse to one value per cell type for the importance heuristic.
    if input_strength.ndim == 2:
        input_strength = input_strength.mean(axis=0)  # (num_cell_types,)

    # Effective incoming weight to type k: input_strength[k] * Σ_i |ave_con[k,i]| * output_strength[i]
    effective_incoming = input_strength * (np.abs(ave_con) @ output_strength)  # (num_cell_types,)

    # Effective outgoing weight from type k: output_strength[k] * Σ_j |ave_con[j,k]| * input_strength[j]
    effective_outgoing = output_strength * (np.abs(ave_con).T @ input_strength)  # (num_cell_types,)

    # Calculate importance score per cell type
    importance = effective_incoming * effective_outgoing
    print(f"Importance range: {importance.min():.6f} to {importance.max():.6f}")

    # Sort cell types by importance (ascending - least important first)
    return np.argsort(importance)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove the least important cell types of a model while "
                    f"its training loss stays below {LOSS_THRESHOLD:.2f}x the original.")
    parser.add_argument('model_folder', type=str, help='Path to the model folder')
    parser.add_argument('--output', type=str, default=None,
                        help='Output folder for the trimmed model '
                             '(default: <model_folder>_trimmed)')
    args = parser.parse_args()

    model_path = Path(args.model_folder)
    model_folder = model_path.parent if model_path.is_file() else model_path
    run_config = load_run_config(model_folder)

    # --- Load the model ---
    print("Loading model...")
    model = nm.BaseRnn.load(model_path)
    model.eval()
    device = model.device
    print('running on', device)
    dtype = model.dtype

    # --- Load training data ---
    print("Loading training data...")
    cell_inputs_data = torch.load(model_folder / 'cell_inputs.pt', map_location=device)
    stimulus = cell_inputs_data['cell_inputs'].to(device=device, dtype=dtype)
    train_mask = cell_inputs_data['train_mask'].to(device=device)
    target = torch.load(model_folder / 'target.pt', map_location=device)
    target = {k: v.to(device=device, dtype=dtype) for k, v in target.items()}

    num_eyes = run_config['run_info']['num_eyes']
    eye_offset = au.run_eye_offset(run_config)
    loss_components = run_config['training']['loss_components']

    def loss_of(mdl):
        return compute_loss(mdl, stimulus, target, train_mask, loss_components,
                            num_eyes, eye_offset)

    # --- Calculate importance score for each cell type ---
    print("Calculating cell type importance scores...")
    print(f"Total cell types: {model.num_cell_types}")
    sorted_indices = importance_order(model)

    # --- Calculate initial loss ---
    print("\nCalculating initial loss...")
    initial_loss = loss_of(model)
    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Target loss ({LOSS_THRESHOLD:.0%} of initial): {initial_loss * LOSS_THRESHOLD:.6f}")

    # --- Build ordered candidate list (least important first, excluding protected types) ---
    candidates = [model.unique_cell_types[i] for i in sorted_indices
                  if model.unique_cell_types[i] not in PROTECTED_TYPES]
    print(f"Eligible cell types for removal: {len(candidates)}")

    # --- Binary search for max cell types removable within threshold ---
    print("\nBinary search for maximum removable cell types...")
    print("-" * 60)

    lo, hi = 0, len(candidates)
    trimmed_model, final_ratio = model, 1.0
    while lo < hi:
        mid = (lo + hi + 1) // 2  # bias high so lo converges upward
        candidate = model.remove_cell_types(candidates[:mid])
        loss_ratio = loss_of(candidate) / initial_loss
        passed = loss_ratio < LOSS_THRESHOLD
        print(f"Removed {mid:4d} / {len(candidates)} | Loss ratio: {loss_ratio:.4f} | "
              f"{'PASS' if passed else 'FAIL'}")

        if passed:
            # lo only grows, so the last passing candidate is the final model.
            lo = mid
            trimmed_model, final_ratio = candidate, loss_ratio
        else:
            hi = mid - 1

    max_removable = lo
    removed_cell_types = candidates[:max_removable]

    print("-" * 60)
    print(f"\nMax removable cell types: {max_removable} / {len(candidates)}")
    print(f"Final loss ratio: {final_ratio:.4f}")
    print(f"Retained cell types: {model.num_cell_types - max_removable}")
    print(f"\nRemoved cell types: {sorted(removed_cell_types)}")

    if max_removable == 0:
        print("\nNo cell type can be removed within the loss threshold; nothing saved.")
        return

    print(f"Neurons: {model.num_neurons} -> {trimmed_model.num_neurons}")
    print(f"Cell types: {model.num_cell_types} -> {trimmed_model.num_cell_types}")

    # --- Save the trimmed model next to the run's config and training data ---
    out_dir = Path(args.output) if args.output else model_folder.with_name(model_folder.name + '_trimmed')
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in RUN_FILES:
        shutil.copy2(model_folder / name, out_dir / name)
    trimmed_model.save(out_dir / 'model.pt')
    print(f"\nTrimmed model saved to: {out_dir}")

    print("\nDone!")


if __name__ == "__main__":
    main()
