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
import model_utilities as mu

# --- Configuration ---
model_path = 'models/best_model/model.pt'
loss_threshold = 1.10  # Stop when loss increases to 110% of starting value (10% worse)
local_config = du.load_config()

# --- Load the model ---
print("Loading model...")
device = local_config['model']['device']
print('running on', device)
model = NeuralRnn.load(model_path, device=device)
dtype = model.dtype

# --- Load training data ---
print("Loading training data...")
stimulus = torch.load('models/best_model/cell_inputs.pt', map_location=device).to(dtype=dtype)
target = torch.load('models/best_model/target.pt', map_location=device)
target = {k: v.to(dtype=dtype) for k, v in target.items()}

# --- Calculate importance score for each cell type ---
print("Calculating cell type importance scores...")

# Get average connections matrix (num_cell_types, num_cell_types)
# ave_con[post_type, pre_type] = average synapse count from pre_type to post_type
ave_con = mu.compute_average_connections(model)

# Sum average incoming connections per cell type (sum over pre_types)
incoming_connections = np.abs(ave_con).sum(axis=1)  # (num_cell_types,)

# Sum average outgoing connections per cell type (sum over post_types)
outgoing_connections = np.abs(ave_con).sum(axis=0)  # (num_cell_types,)

# Get input and output strengths per cell type (already at cell type level)
input_strength = model.input_strength_log.exp().detach().cpu().numpy()  # (num_cell_types,)
output_strength = model.output_strength_log.exp().detach().cpu().numpy()  # (num_cell_types,)

# Calculate importance score per cell type
importance = incoming_connections * outgoing_connections * input_strength * output_strength

# Sort cell types by importance (ascending - least important first)
sorted_indices = np.argsort(importance)

print(f"Total cell types: {model.num_cell_types}")
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

# --- Iteratively silence cell types ---
print("\nStarting cell type trimming...")
print("-" * 60)

silenced_cell_types = ['5-HTPMPV01', 'OA-AL2i2', 'cM08a', 'cLM01', 'cLLP02', 'OA-AL2i3', 'PDt', 'OA-AL2i4', 'cL16', 'CB2685', 'AVLP593', 'DGI', 'cL10', 'LPi05', 'LPT28', 'CB0143', 'MeTu4d', 'LPT48_vCal3', 'mALD1', 'DNg33', 'LPT27', 'Dm4', 'Li29', 'LPT21', 'LT1c', 'mALC3', 'aMe17c', 'LTe28', 'LPi08', 'MeTu2a', 'LPC1', 'DN1a', 'LTe15', 'PLP248', 'LPi04', 'VS7', 'MTe13', 'VCH', 'LTe37', 'PLP081', 'LTe21', 'cL07', 'Dm13', 'Sm24', 'VS5', 'PLP177', 'VS2', 'LT33', 'LPT47_vCal2', 'CL104', 'LC19', 'LPT31', 'LT1b', 'LPTe02', 'PLP163', 'cL02b', 'MTe41', 'cM01b', 'Pm11', 'LC27', 'LPC2', 'Li22', 'LT72', 'LC25', 'Sm26', 'LC35', 'cM16', 'TmY4', 'LMt1', 'cLP04', 'LTe42a', 'LMa5', 'LTe74', 'Sm29', 'MTe09', 'DN1-l', 'PLP037b', 'Nod1', 'MTe18', 'Dm3v', 'LTe18', 'cM05', 'Pm05', 'LTe20', 'MTe08', 'Pm14', 'LT51', 'Tlp5', 'LPTe01', 'Li08', 'Sm36', 'MTe48', 'LPT26', 'LC45', 'TmY14', 'Pm12', 'DCH', 'Pm04', 'LTe49b', 'MTe02', 'LTe56', 'AVLP534', 'cLLPM02', 'PLP020', 'LC37a', 'MTe25', 'aMe17a1', 'Li17', 'VST1', 'PLP252', 'cL22c', 'Li26', 'PLP239', 'LT40', 'LC34', 'MTe44', 'LT65', 'Li14', 'PLP251', 'Li124_56', 'PLP231', 'Sm16', 'LTe41', 'LT86', 'Tm2', 'LMt2', 'DmDRA1', 'LCe01a', 'LCe07', 'MTe27', 'LTe38b', 'Sm20', 'OA-AL2i1', 'PLP104', 'LT76', 'Tm5d', 'APDN3', 'Li18', 'MTe26', 'LTe23', 'LTe72', 'cL02d', 'LC6', 'cM12', 'LMa3', 'LTe09', 'cL22b', 'Sm38', 'cL15', 'LPT23', 'Li09', 'LTe67', 'LLPC2', 'Li16', 'H2', 'cL08', 'LTe42b', 'VST2', 'cM01c', 'Li15', 'LC40', 'Sm19', 'CB3238', 'T2a', 'LLPC1', 'Li19', 'Sm37', 'LT77', 'Dm1', 'cLP02', 'HSE', 'LTe22', 'LT57', 'MC65', 'LC15', 'Sm43', 'LMa1', 'LTe10', 'LPT58', 'Li32', 'PLP215', 'LTe13', 'DNc02', 'VSm', 'Li07', 'Sm28', 'DNp11', 'cM03', 'MLt5', 'Tm16', 'LT59', 'MTe40', 'Sm30', 'LTe47', 'LCe04', 'LT68', 'AOTU065', 'MTe42', 'LTe70', 'LC39', 'LT69', 'CB3866', 'cLP01', 'cL18', 'LPi02', 'Mi14', 'MeTu4a', 'PLP249', 'Sm40', 'Dm8b', 'LC21', 'LTe58', 'LTe52b', 'Sm21', 'Sm18', 'Li03', 'Sm42', 'LPi06', 'Lai', 'Pm13', 'CT1', 'MTe17', 'TmY20', 'LLPC4', 'LC22', 'LTe36', 'LT63', 'VS6', 'LC43', 'AOTU053', 'cL02a', 'LTe05', 'LT67', 'CB3790', 'MTe19', 'LT75', 'MTe30', 'LT78', 'LC46', 'Li24', 'LTe40', 'MeTu4b', 'LMa2', 's-CPDN3A', 'LPLC1', 'MLt8', 'cL21', 'LT85', 'LTe49d', 'LTe06', 'Sm04', 'LTe68', 'LT58', 'LTe33', 'LHPV2i2a', 'Li31', 'LC36', 'Sm17', 'LT43', 'MTe28', 'TmY11']
skipped_types = ['L1', 'L2', 'L3', 'T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d']
current_loss = initial_loss

for cell_type_idx in sorted_indices:
    # Get the cell type name
    cell_type = model.unique_cell_types[cell_type_idx]
    
    if cell_type in skipped_types:
        continue

    if cell_type in silenced_cell_types:
        continue

    # Add to silenced list
    silenced_cell_types.append(cell_type)
    
    # Create a fresh model copy and apply silencing
    model_copy = NeuralRnn.load(model_path, device=device)
    model_copy.silence_neurons(silenced_cell_types, invert_silencing=False)
    
    # Calculate loss with silenced cell types
    with torch.no_grad():
        pred = model_copy.predict(stimulus, input_neurons=input_neurons, output_neurons=output_neurons)[0]
        current_loss = tu.calc_loss(tu.cos_loss, pred['firing_rate'], target).item()
    
    loss_ratio = current_loss / initial_loss
    
    print(f"Silenced {len(silenced_cell_types)} / {model.num_cell_types} cell types | Latest: {cell_type:10s} | "
          f"Loss: {current_loss:.6f} | Ratio: {loss_ratio:.4f}")
    
    # Check if we've reached the threshold
    if loss_ratio >= loss_threshold:
        print("-" * 60)
        print(f"\nStopped: Loss reached {loss_ratio:.2%} of initial value")
        print(f"Total cell types silenced: {len(silenced_cell_types)}")
        print(f"Silenced cell types: {sorted(silenced_cell_types)}")
        break

else:
    print("-" * 60)
    print(f"\nCompleted all cell types without reaching threshold")
    print(f"Final loss ratio: {current_loss / initial_loss:.2%}")

print("\nDone!")
