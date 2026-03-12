"""
Model trimming: binary search for maximum neurons we can silence before loss degrades.

Importance score = incoming_synapses * outgoing_synapses * input_strength * output_strength

Neurons are ranked by importance. We binary search over the count of least-important
neurons to silence, finding the maximum number removable before loss exceeds a threshold.
"""
import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import neural_models as nm
import data_utilities as du
import model_utilities as mu
from loss_functions import mse_loss

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('model_folder', type=str, help='Path to the model folder')
args = parser.parse_args()

# --- Configuration ---
model_path = args.model_folder
loss_threshold = 1.10  # Stop when loss increases to 110% of starting value (10% worse)
local_config = du.load_config()

# --- Load the model ---
print("Loading model...")
device = local_config['model']['device']
print('running on', device)
model = nm.NeuralModel.load(model_path, device=device)
dtype = model.dtype

# --- Load training data ---
print("Loading training data...")
model_data_path = Path(model_path)
cell_inputs_data = torch.load(model_data_path / 'cell_inputs.pt', map_location=device)
if isinstance(cell_inputs_data, dict):
    stimulus = cell_inputs_data['cell_inputs'].to(device=device, dtype=dtype)
else:
    stimulus = cell_inputs_data.to(device=device, dtype=dtype)

target_loaded = torch.load(model_data_path / 'target.pt', map_location=device)
if isinstance(target_loaded, dict):
    target = {k: (v.to(device=device, dtype=dtype) if torch.is_tensor(v) else v)
              for k, v in target_loaded.items()}
else:
    target = target_loaded.to(device=device, dtype=dtype)

# --- Calculate importance score for each cell type ---
print("Calculating cell type importance scores...")

# Get average connections matrix (num_cell_types, num_cell_types)
# ave_con[post_type, pre_type] = average synapse count from pre_type to post_type
ave_con = mu.compute_average_connections(model)

# Get input and output strengths per cell type (already at cell type level)
input_strength = model.input_strength_log.exp().detach().cpu().numpy()  # (num_cell_types,)
output_strength = model.output_strength_log.exp().detach().cpu().numpy()  # (num_cell_types,)

# Effective incoming weight to type k: input_strength[k] * Σ_i |ave_con[k,i]| * output_strength[i]
effective_incoming = input_strength * (np.abs(ave_con) @ output_strength)  # (num_cell_types,)

# Effective outgoing weight from type k: output_strength[k] * Σ_j |ave_con[j,k]| * input_strength[j]
effective_outgoing = output_strength * (np.abs(ave_con).T @ input_strength)  # (num_cell_types,)

# Calculate importance score per cell type
importance = effective_incoming * effective_outgoing

# Sort cell types by importance (ascending - least important first)
sorted_indices = np.argsort(importance)

print(f"Total cell types: {model.num_cell_types}")
print(f"Importance range: {importance.min():.6f} to {importance.max():.6f}")

# --- Calculate initial loss ---
print("\nCalculating initial loss...")
input_neurons = ['L1', 'L2', 'L3']

target_tensor = target['optic_flow'] if isinstance(target, dict) else target

with torch.no_grad():
    pred = model.predict(stimulus, input_neurons=input_neurons)['outputs']
    initial_loss = mse_loss(target_tensor, pred['decoder']).item()

print(f"Initial loss: {initial_loss:.6f}")
print(f"Target loss (110% of initial): {initial_loss * loss_threshold:.6f}")

# --- Build ordered candidate list (least important first, excluding protected types) ---
skipped_types = {'L1', 'L2', 'L3', 'T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d'}
candidates = [model.unique_cell_types[i] for i in sorted_indices
              if model.unique_cell_types[i] not in skipped_types]

print(f"Eligible cell types for silencing: {len(candidates)}")


def evaluate_silencing(count):
    """Silence the `count` least-important candidates and return the loss ratio."""
    silenced = candidates[:count]
    model_copy = nm.NeuralModel.load(model_path, device=device)
    model_copy.silence_neurons(silenced, invert_silencing=False)
    with torch.no_grad():
        pred = model_copy.predict(stimulus, input_neurons=input_neurons)['outputs']
        loss = mse_loss(target_tensor, pred['decoder']).item()
    return loss / initial_loss


# --- Binary search for max neurons removable within threshold ---
print("\nBinary search for maximum removable neurons...")
print("-" * 60)

lo, hi = 0, len(candidates)

while lo < hi:
    mid = (lo + hi + 1) // 2  # bias high so lo converges upward
    loss_ratio = evaluate_silencing(mid)
    print(f"Silenced {mid:4d} / {len(candidates)} | Loss ratio: {loss_ratio:.4f} | "
          f"{'PASS' if loss_ratio < loss_threshold else 'FAIL'}")

    if loss_ratio < loss_threshold:
        lo = mid
    else:
        hi = mid - 1

max_removable = lo
final_ratio = evaluate_silencing(max_removable) if max_removable > 0 else 1.0
silenced_cell_types = candidates[:max_removable]

print("-" * 60)
print(f"\nMax removable neurons: {max_removable} / {len(candidates)}")
print(f"Final loss ratio: {final_ratio:.4f}")
print(f"Retained cell types: {len(candidates) - max_removable + len(skipped_types)}")
print(f"\nSilenced cell types: {sorted(silenced_cell_types)}")

print("\nDone!")
