"""
Create a movie of the natural movie stimulus with true and predicted optic flow.
2x2 layout:
  - Top left: Stimulus intensity (scatter plot)
  - Top right: True optic flow (quiver plot)
  - Bottom left: Blank (reserved for future use)
  - Bottom right: Model's predicted optic flow (quiver plot)
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import imageio as iio_v2

import data_utilities as du
import stimulus_utilities as su
from neural_models import NeuralRnn

# --- Configuration ---
config = du.load_config()
connectome_folder = Path(config['data']['connectome_folder'])
output_folder = Path('figures/natural_movie')
output_folder.mkdir(parents=True, exist_ok=True)

frame_rate = 30  # fps for the output movie
stim_index = 0   # which stimulus to visualize (first one)

# --- Load the model ---
print("Loading model...")
model_path = 'models/best_model/model.pt'
model = NeuralRnn.load(model_path, device='cpu')
device = model.device
dtype = model.dtype

# --- Load the natural movie stimulus and optic flow ---
print("Loading natural movie data...")
stimulus, optic_flow = su.load_natural_movie()
# stimulus: (num_stimuli, time, columns)
# optic_flow: (num_stimuli, time, columns, 2) where [:,:,:,0] is LR, [:,:,:,1] is UD

print(f"Stimulus shape: {stimulus.shape}")
print(f"Optic flow shape: {optic_flow.shape}")

# Convert to tensors for model
stimulus_tensor = torch.tensor(stimulus, device=device, dtype=dtype)
optic_flow_tensor = torch.tensor(optic_flow, device=device, dtype=dtype)

# --- Run model to get predicted optic flow ---
print("Running model prediction...")
with torch.no_grad():
    estimated_optic_flow = model.calculate_optic_flow(stimulus_tensor)[0]

# Convert predictions to numpy
pred_flow = estimated_optic_flow.cpu().numpy()
print(f"Predicted optic flow shape: {pred_flow.shape}")

# --- Load column positions ---
column_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_column_info.parquet')
cell_x = -column_info_df.x.to_numpy()  # flip x for visualization
cell_z = column_info_df.z.to_numpy()

# --- Extract first stimulus ---
stim_single = stimulus[stim_index, :, :]  # (time, columns)
true_flow_single = optic_flow[stim_index, :, :, :]  # (time, columns, 2)
pred_flow_single = pred_flow[stim_index, :, :, :]  # (time, columns, 2)
num_time, num_cols = stim_single.shape

print(f"Creating movie for stimulus {stim_index} with {num_time} frames...")

# --- Set up normalizations ---
# Stimulus normalization (min to max)
stim_min, stim_max = np.nanmin(stim_single), np.nanmax(stim_single)
stim_norm = mcolors.Normalize(vmin=stim_min, vmax=stim_max)

# True optic flow
true_flow_u = true_flow_single[:, :, 0]  # horizontal (LR)
true_flow_v = true_flow_single[:, :, 1]  # vertical (UD)

# Predicted optic flow
pred_flow_u = pred_flow_single[:, :, 0]  # horizontal (LR)
pred_flow_v = pred_flow_single[:, :, 1]  # vertical (UD)

# Use same normalization for both true and predicted (based on true flow)
true_magnitude = np.hypot(true_flow_u, true_flow_v)
max_mag = np.nanpercentile(true_magnitude, 99) if np.any(true_magnitude) else 1.0
flow_norm = mcolors.Normalize(vmin=0.0, vmax=max_mag)

# --- Create figure (2x2 layout) ---
fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.24), dpi=100)  # divisible by 16
ax_stim = axes[0, 0]      # top left: stimulus
ax_true = axes[0, 1]      # top right: true optic flow
ax_blank = axes[1, 0]     # bottom left: blank
ax_pred = axes[1, 1]      # bottom right: predicted optic flow

# Set up axes limits with some padding
x_margin = 0.1 * (cell_x.max() - cell_x.min())
z_margin = 0.1 * (cell_z.max() - cell_z.min())

for ax in [ax_stim, ax_true, ax_pred]:
    ax.set_xlim(cell_x.min() - x_margin, cell_x.max() + x_margin)
    ax.set_ylim(cell_z.min() - z_margin, cell_z.max() + z_margin)
    ax.set_aspect('equal')
    ax.set_xlabel('x (posterior ↔ anterior)')
    ax.set_ylabel('z (ventral ↔ dorsal)')

# Blank axis - turn off
ax_blank.axis('off')
ax_blank.set_title('(reserved)')

# Initialize stimulus scatter
sc_stim = ax_stim.scatter(cell_x, cell_z, c=stim_single[0, :], cmap='gray', 
                          norm=stim_norm, s=90, edgecolors='none')
ax_stim.set_title('Stimulus')
cbar_stim = fig.colorbar(sc_stim, ax=ax_stim, label='intensity')

# Arrow scale (same for both true and predicted)
arrow_scale = 0.1 / max_mag if max_mag > 0 else 1.0

# Initialize TRUE optic flow quiver
q_true = ax_true.quiver(cell_x, cell_z, 
                        true_flow_u[0, :], true_flow_v[0, :],
                        np.hypot(true_flow_u[0, :], true_flow_v[0, :]),
                        cmap='viridis', norm=flow_norm,
                        angles='xy', scale_units='xy', scale=1/arrow_scale,
                        width=0.003)
ax_true.set_title('True Optic Flow')
cbar_true = fig.colorbar(q_true, ax=ax_true, label='magnitude')

# Initialize PREDICTED optic flow quiver
q_pred = ax_pred.quiver(cell_x, cell_z, 
                        pred_flow_u[0, :], pred_flow_v[0, :],
                        np.hypot(pred_flow_u[0, :], pred_flow_v[0, :]),
                        cmap='viridis', norm=flow_norm,
                        angles='xy', scale_units='xy', scale=1/arrow_scale,
                        width=0.003)
ax_pred.set_title('Predicted Optic Flow')
cbar_pred = fig.colorbar(q_pred, ax=ax_pred, label='magnitude')

plt.tight_layout()


def fig_to_rgb(fig):
    """Convert matplotlib figure to RGB array."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[..., :3].copy()


# --- Create movie ---
out_path = output_folder / f'natural_movie_comparison_stim_{stim_index}.mp4'
print(f"Writing movie to {out_path}...")

with iio_v2.get_writer(str(out_path), codec="libx264", fps=frame_rate) as writer:
    for t in range(num_time):
        # Update stimulus
        sc_stim.set_array(stim_single[t, :])
        
        # Update true optic flow
        u_true_t = true_flow_u[t, :]
        v_true_t = true_flow_v[t, :]
        mag_true_t = np.hypot(u_true_t, v_true_t)
        q_true.set_UVC(u_true_t, v_true_t, mag_true_t)
        
        # Update predicted optic flow
        u_pred_t = pred_flow_u[t, :]
        v_pred_t = pred_flow_v[t, :]
        mag_pred_t = np.hypot(u_pred_t, v_pred_t)
        q_pred.set_UVC(u_pred_t, v_pred_t, mag_pred_t)
        
        # Update title with frame number
        ax_stim.set_title(f'Stimulus (frame {t+1}/{num_time})')
        
        # Capture frame
        writer.append_data(fig_to_rgb(fig))
        
        if (t + 1) % 20 == 0:
            print(f"  Frame {t+1}/{num_time}")

plt.close(fig)
print(f"Done! Movie saved to {out_path}")
