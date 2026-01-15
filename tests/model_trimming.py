"""
Model trimming: iteratively silence least important neurons and track loss.

Importance score = incoming_synapses * outgoing_synapses * input_strength * output_strength
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from neural_models import NeuralRnn
import data_utilities as du
import training_utilities as tu

# --- Configuration ---
model_path = 'models/best_model/model.pt'
loss_threshold = 1.10  # Stop when loss increases to 110% of starting value (10% worse)

# --- Load the model ---
print("Loading model...")
model = NeuralRnn.load(model_path, device='cpu')
device = model.device
dtype = model.dtype

# --- Load training data ---
print("Loading training data...")
stimulus = torch.load('models/best_model/cell_inputs.pt', map_location=device).to(dtype=dtype)
target = torch.load('models/best_model/target.pt', map_location=device)
target = {k: v.to(dtype=dtype) for k, v in target.items()}

# --- Calculate importance score for each neuron ---
print("Calculating neuron importance scores...")

# Get synapse counts (sparse matrix)
synapse_counts = model.synapse_counts  # (num_neurons, num_neurons), sparse

# Sum incoming synapses per neuron (sum over rows = sum of column)
# synapse_counts[i, j] = synapses from j to i, so sum over dim=1 gives incoming to each neuron
incoming_synapses = torch.sparse.sum(synapse_counts.abs(), dim=1).to_dense()  # (num_neurons,)

# Sum outgoing synapses per neuron (sum over columns = sum of row)
outgoing_synapses = torch.sparse.sum(synapse_counts.abs(), dim=0).to_dense()  # (num_neurons,)

# Get input and output strengths per neuron (mapped from cell type)
input_strength = model.input_strength_log.exp()[model.neuron_to_type_index]  # (num_neurons,)
output_strength = model.output_strength_log.exp()[model.neuron_to_type_index]  # (num_neurons,)

# Calculate importance score
importance = incoming_synapses * outgoing_synapses * input_strength * output_strength
importance = importance.detach().cpu().numpy()

# Sort neurons by importance (ascending - least important first)
sorted_indices = np.argsort(importance)

print(f"Total neurons: {model.num_neurons}")
print(f"Importance range: {importance.min():.6f} to {importance.max():.6f}")

# --- Calculate initial loss ---
print("\nCalculating initial loss...")
input_neurons = ['L1', 'L2', 'L3']
output_neurons = list(target.keys())

with torch.no_grad():
    pred = model.predict(stimulus, input_neurons=input_neurons, output_neurons=output_neurons)[0]
    initial_loss = tu.calc_loss(tu.cos_loss, pred['firing_rate'], target).item()

print(f"Initial loss: {initial_loss:.6f}")
print(f"Target loss (110% of initial): {initial_loss * loss_threshold:.6f}")

# --- Iteratively silence neurons ---
print("\nStarting neuron trimming...")
print("-" * 60)

silenced_neurons = set()
current_loss = initial_loss
neurons_silenced = 0

for neuron_idx in sorted_indices:
    # Get the cell type of this neuron
    cell_type = model.cell_types[neuron_idx]
    
    # Skip if this cell type is already in our silenced set
    # (since silencing works at cell type level, not individual neuron)
    if cell_type in silenced_neurons:
        continue
    
    # Silence this cell type
    silenced_neurons.add(cell_type)
    
    # Create a fresh model copy and apply silencing
    model_copy = NeuralRnn.load(model_path, device='cpu')
    model_copy.silence_neurons(list(silenced_neurons), invert_silencing=False)
    
    # Calculate loss with silenced neurons
    with torch.no_grad():
        pred = model_copy.predict(stimulus, input_neurons=input_neurons, output_neurons=output_neurons)[0]
        current_loss = tu.calc_loss(tu.cos_loss, pred['firing_rate'], target).item()
    
    neurons_silenced += 1
    loss_ratio = current_loss / initial_loss
    
    print(f"Silenced {neurons_silenced} / {model.num_cell_types} cell types | Latest: {cell_type:10s} | "
          f"Loss: {current_loss:.6f} | Ratio: {loss_ratio:.4f}")
    
    # Check if we've reached the threshold
    if loss_ratio >= loss_threshold:
        print("-" * 60)
        print(f"\nStopped: Loss reached {loss_ratio:.2%} of initial value")
        print(f"Total cell types silenced: {neurons_silenced}")
        print(f"Silenced cell types: {sorted(silenced_neurons)}")
        break

else:
    print("-" * 60)
    print(f"\nCompleted all neurons without reaching threshold")
    print(f"Final loss ratio: {current_loss / initial_loss:.2%}")

print("\nDone!")
