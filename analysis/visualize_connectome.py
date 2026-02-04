import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import neural_models as nm
import data_utilities as du
import model_utilities as mu

# Load the model
model_folder = Path('models/20260126_131210')
model = nm.NeuralModel.load(model_folder / 'model.pt', device='cpu')

# Compute average connections between cell types
ave_con = mu.compute_average_connections(model)

# Get cell type labels
cell_types = model.unique_cell_types

# Create the figure
fig, ax = plt.subplots(figsize=(14, 12))

# Apply signed cube root transform for compression (handles negative values naturally)
ave_con_transformed = np.cbrt(ave_con)

# Plot the connection matrix
vmax = np.abs(ave_con_transformed).max()
im = ax.imshow(ave_con_transformed, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')

# Add colorbar with original scale tick labels
cbar = fig.colorbar(im, ax=ax)
# Create ticks in transformed space, label with original values
cbar_ticks = np.linspace(-vmax, vmax, 9)
cbar_labels = [f'{v**3:.0f}' for v in cbar_ticks]  # Convert back to original scale
cbar.set_ticks(cbar_ticks)
cbar.set_ticklabels(cbar_labels)
cbar.set_label('Average synapse count', fontsize=14)
cbar.ax.tick_params(labelsize=12)

# Set tick labels
ax.set_xticks(np.arange(len(cell_types)))
ax.set_yticks(np.arange(len(cell_types)))
ax.set_xticklabels(cell_types, rotation=90, fontsize=12)
ax.set_yticklabels(cell_types, fontsize=12)

# Labels: ave_con[post, pre] so rows=post, cols=pre
ax.set_xlabel('Presynaptic cell type', fontsize=14)
ax.set_ylabel('Postsynaptic cell type', fontsize=14)
ax.set_title(f'Average synapse counts between cell types\n({len(cell_types)} cell types)', fontsize=16)

plt.tight_layout()
plt.savefig(model_folder / 'connectome_matrix.png', dpi=150)
plt.show()

print(f"Saved connectome matrix to {model_folder / 'connectome_matrix.png'}")
