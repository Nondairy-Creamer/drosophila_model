"""
Create a movie comparing true vs predicted optic flow.
2x2 layout:
  - Top left: Stimulus intensity (scatter plot)
  - Top right: True optic flow (quiver plot)
  - Bottom left: Blank (reserved)
  - Bottom right: Predicted optic flow (quiver plot)
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import imageio as iio_v2

import data_utilities as du
import neural_models as nm
import training_utilities as tu


def _resolve_model_paths(model_path: Path) -> tuple[Path, Path]:
    if model_path.is_dir():
        model_folder = model_path
        model_file = model_folder / 'model.pt'
    else:
        model_file = model_path
        model_folder = model_path.parent

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    return model_folder, model_file


def _extract_optic_flow(target_loaded: object) -> torch.Tensor:
    if torch.is_tensor(target_loaded):
        return target_loaded

    if isinstance(target_loaded, dict):
        for key in ('decoder', 'optic_flow', 'flow'):
            if key in target_loaded:
                value = target_loaded[key]
                if torch.is_tensor(value):
                    return value
        raise ValueError(
            "Target dict does not contain a tensor under keys: decoder, optic_flow, flow."
        )

    raise TypeError("Unsupported target format. Expected tensor or dict of tensors.")


def _extract_pred_flow(pred_output: object) -> torch.Tensor:
    if torch.is_tensor(pred_output):
        return pred_output

    if isinstance(pred_output, tuple):
        pred_output = pred_output[0]

    if isinstance(pred_output, dict):
        if 'decoder' in pred_output:
            return pred_output['decoder']
        raise ValueError("Predict output dict missing 'decoder' key.")

    raise TypeError("Unsupported predict output format.")


def fig_to_rgb(fig):
    """Convert matplotlib figure to RGB array."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[..., :3].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Make optic flow movie for a model.")
    parser.add_argument("model_path", type=str, help="Path to model folder or model.pt")
    parser.add_argument("--stim-index", type=int, default=5, help="Stimulus index to visualize")
    args = parser.parse_args()

    model_folder, model_file = _resolve_model_paths(Path(args.model_path))
    config = du.load_config(model_folder / 'config.toml')
    print("Loading model...")
    model_type = config['model'].get('model_type')
    ModelClass = getattr(nm, model_type)
    model = ModelClass.load(model_file, device='cpu')
    model.use_decoder = True
    device = model.device
    dtype = model.dtype
    frame_rate = max(1, int((1 / model.dt) / 2))

    print("Loading inputs and targets...")
    cell_inputs = torch.load(model_folder / 'cell_inputs.pt', map_location=device)
    cell_inputs = cell_inputs.to(device=device, dtype=dtype)
    target_loaded = torch.load(model_folder / 'target.pt', map_location=device)
    true_flow = _extract_optic_flow(target_loaded).to(device=device, dtype=dtype)

    print("Running model prediction...")
    with torch.no_grad():
        pred_output = model.predict(cell_inputs)
    pred_flow = _extract_pred_flow(pred_output).to(device=device, dtype=dtype)

    print("Cosine similarity", 1 - tu.cos_loss(true_flow, pred_flow))

    # Convert to numpy for plotting
    stimulus = cell_inputs.detach().cpu().numpy()
    true_flow = true_flow.detach().cpu().numpy()
    pred_flow = pred_flow.detach().cpu().numpy()

    print(f"Stimulus shape: {stimulus.shape}")
    print(f"True flow shape: {true_flow.shape}")
    print(f"Pred flow shape: {pred_flow.shape}")

    # Use model column locations in spherical coordinates and project to 3D (radius=1)
    column_xyz = model.get_column_xyz()
    cell_x = column_xyz[:, 0]
    cell_z = column_xyz[:, 2]

    # Extract stimulus and flow for a single stimulus
    stim_index = args.stim_index
    stim_single = stimulus[stim_index, :, :]
    true_flow_single = true_flow[stim_index, :, :, :]
    pred_flow_single = pred_flow[stim_index, :, :, :]
    num_time, _ = stim_single.shape

    print(f"Creating movie for stimulus {stim_index} with {num_time} frames...")

    # Normalizations
    stim_min, stim_max = np.nanmin(stim_single), np.nanmax(stim_single)
    stim_norm = mcolors.Normalize(vmin=stim_min, vmax=stim_max)

    true_flow_u = true_flow_single[:, :, 0]
    true_flow_v = true_flow_single[:, :, 1]
    pred_flow_u = pred_flow_single[:, :, 0]
    pred_flow_v = pred_flow_single[:, :, 1]

    true_magnitude = np.hypot(true_flow_u, true_flow_v)
    max_mag = np.nanpercentile(true_magnitude, 99) if np.any(true_magnitude) else 1.0
    flow_norm = mcolors.Normalize(vmin=0.0, vmax=max_mag)

    # --- Create figure (2x2 layout) ---
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.24), dpi=100)
    ax_stim = axes[0, 0]
    ax_true = axes[0, 1]
    ax_blank = axes[1, 0]
    ax_pred = axes[1, 1]

    # Set up axes limits with some padding (invert x-axis: +X on left)
    x_margin = 0.1 * (cell_x.max() - cell_x.min())
    z_margin = 0.1 * (cell_z.max() - cell_z.min())

    for ax in [ax_stim, ax_true, ax_pred]:
        ax.set_xlim(cell_x.max() + x_margin, cell_x.min() - x_margin)
        ax.set_ylim(cell_z.min() - z_margin, cell_z.max() + z_margin)
        ax.set_aspect('equal')
        ax.set_xlabel('x (anterior ← → posterior)')
        ax.set_ylabel('z (ventral ↔ dorsal)')

    ax_blank.axis('off')
    ax_blank.set_title('(reserved)')

    # Initialize stimulus scatter
    sc_stim = ax_stim.scatter(
        cell_x, cell_z, c=stim_single[0, :], cmap='gray', norm=stim_norm, s=90, edgecolors='none'
    )
    ax_stim.set_title('Stimulus')
    fig.colorbar(sc_stim, ax=ax_stim, label='intensity')

    # Arrow scale (same for both true and predicted)
    arrow_scale = 0.2 / max_mag if max_mag > 0 else 1.0

    # Initialize TRUE optic flow quiver (negate horizontal for inverted x-axis)
    q_true = ax_true.quiver(
        cell_x, cell_z,
        -true_flow_u[0, :], true_flow_v[0, :],
        np.hypot(true_flow_u[0, :], true_flow_v[0, :]),
        cmap='viridis', norm=flow_norm,
        angles='xy', scale_units='xy', scale=1 / arrow_scale,
        width=0.003
    )
    ax_true.set_title('True Optic Flow')
    fig.colorbar(q_true, ax=ax_true, label='magnitude')

    # Initialize PREDICTED optic flow quiver (negate horizontal for inverted x-axis)
    q_pred = ax_pred.quiver(
        cell_x, cell_z,
        -pred_flow_u[0, :], pred_flow_v[0, :],
        np.hypot(pred_flow_u[0, :], pred_flow_v[0, :]),
        cmap='viridis', norm=flow_norm,
        angles='xy', scale_units='xy', scale=1 / arrow_scale,
        width=0.003
    )
    ax_pred.set_title('Predicted Optic Flow')
    fig.colorbar(q_pred, ax=ax_pred, label='magnitude')

    plt.tight_layout()

    # --- Create movie ---
    output_folder = model_folder / 'optic_flow_movie'
    output_folder.mkdir(parents=True, exist_ok=True)
    out_path = output_folder / f'optic_flow_comparison_stim_{stim_index}.mp4'
    print(f"Writing movie to {out_path}...")

    with iio_v2.get_writer(str(out_path), codec="libx264", fps=frame_rate) as writer:
        for t in range(num_time):
            sc_stim.set_array(stim_single[t, :])

            u_true_t = true_flow_u[t, :]
            v_true_t = true_flow_v[t, :]
            mag_true_t = np.hypot(u_true_t, v_true_t)
            q_true.set_UVC(-u_true_t, v_true_t, mag_true_t)

            u_pred_t = pred_flow_u[t, :]
            v_pred_t = pred_flow_v[t, :]
            mag_pred_t = np.hypot(u_pred_t, v_pred_t)
            q_pred.set_UVC(-u_pred_t, v_pred_t, mag_pred_t)

            ax_stim.set_title(f'Stimulus (frame {t+1}/{num_time})')
            writer.append_data(fig_to_rgb(fig))

            if (t + 1) % 20 == 0:
                print(f"  Frame {t+1}/{num_time}")

    plt.close(fig)
    print(f"Done! Movie saved to {out_path}")


if __name__ == "__main__":
    main()
