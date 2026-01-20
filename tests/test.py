import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from neural_models import CurrentModel
from pathlib import Path
import data_utilities as du
import stimulus_utilities as su
import training_utilities as tu
import matplotlib.pyplot as plt
import time

config = du.load_config()
stimulus_seed = 0

# --- load the model ---
model_path = 'models/best_model/model.pt'
model = CurrentModel.load(model_path, device='cpu')
device = model.device
dtype = model.dtype

# --- grab the natural movie stimulus ---
stimulus, optic_flow = du.get_dataset('flow_360', config)

# --- run the model on the data ---
estimated_optic_flow, initial_state = model.calculate_optic_flow(stimulus)

cos_sim = 1 - tu.cos_loss(estimated_optic_flow, optic_flow)

print(f"Cosine similarity: {cos_sim}")

# --- Bar plots for specific neurons ---
neurons_of_interest = ['L1', 'L2', 'L3', 'L4', 'L5', 'Mi1', 'Tm3', 'Tm1', 'Tm2', 'Tm9', 'Mi4', 'Mi9', 'C2', 'C3', 'CT1']

# Get model parameters (per cell type)
persistence = model.persistence_logit.sigmoid().detach().cpu().numpy()
input_strength = model.input_strength_log.exp().detach().cpu().numpy()
output_strength = model.output_strength_log.exp().detach().cpu().numpy()

# Map neuron names to indices in unique_cell_types
unique_cell_types = model.unique_cell_types
neuron_indices = []
valid_neurons = []
for neuron in neurons_of_interest:
    if neuron in unique_cell_types:
        idx = np.where(unique_cell_types == neuron)[0][0]
        neuron_indices.append(idx)
        valid_neurons.append(neuron)
    else:
        print(f"Warning: {neuron} not found in model")

# Extract values for neurons of interest
persistence_vals = persistence[neuron_indices]
input_strength_vals = input_strength[neuron_indices]
output_strength_vals = output_strength[neuron_indices]
input_output_product = input_strength_vals * output_strength_vals

# Create bar plots
fig, axes = plt.subplots(4, 1, figsize=(12, 14))
x = np.arange(len(valid_neurons))

# Plot 1: Persistence
axes[0].bar(x, persistence_vals, color='steelblue', edgecolor='black')
axes[0].set_xticks(x)
axes[0].set_xticklabels(valid_neurons, rotation=45, ha='right')
axes[0].set_ylabel('Persistence')
axes[0].set_title('Persistence by Neuron Type')
axes[0].set_ylim(0, 1)

# Plot 2: Input Strength
axes[1].bar(x, input_strength_vals, color='forestgreen', edgecolor='black')
axes[1].set_xticks(x)
axes[1].set_xticklabels(valid_neurons, rotation=45, ha='right')
axes[1].set_ylabel('Input Strength')
axes[1].set_title('Input Strength by Neuron Type')

# Plot 3: Output Strength
axes[2].bar(x, output_strength_vals, color='coral', edgecolor='black')
axes[2].set_xticks(x)
axes[2].set_xticklabels(valid_neurons, rotation=45, ha='right')
axes[2].set_ylabel('Output Strength')
axes[2].set_title('Output Strength by Neuron Type')

# Plot 4: Input * Output Product
axes[3].bar(x, input_output_product, color='mediumpurple', edgecolor='black')
axes[3].set_xticks(x)
axes[3].set_xticklabels(valid_neurons, rotation=45, ha='right')
axes[3].set_ylabel('Input × Output')
axes[3].set_title('Input Strength × Output Strength by Neuron Type')

plt.tight_layout()
plt.savefig('figures/neuron_parameters.png')
plt.show()
print("Saved neuron parameter plots to figures/neuron_parameters.png")
