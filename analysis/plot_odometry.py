"""
Plot predicted vs true odometry (rotation and translation) for a two-eye model.

Layout (3 x 2):
  Left column: Rotation x, y, z  (deg/s)
  Right column: Translation x, y, z

Usage:
    python analysis/plot_odometry.py models/20260226_190611_gpu8_run0/
    python analysis/plot_odometry.py models/best_model --stim-index 5
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import data_utilities as du
import neural_models as nm

from analysis_utilities import resolve_model_paths


AXIS_LABELS = ['x (roll)', 'y (pitch)', 'z (yaw)']


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot predicted vs true odometry.")
    parser.add_argument(
        "model_path", type=str, nargs="?", default="models/best_model",
        help="Path to model folder or model.pt",
    )
    parser.add_argument("--stim-index", type=int, default=5,
                        help="Stimulus index to visualize")
    args = parser.parse_args()

    model_file, model_folder = resolve_model_paths(Path(args.model_path))
    local_config = du.load_config()
    device = local_config['model'].get('device', 'cpu')
    print(f"Loading model from {model_file} on {device}...")
    model = nm.NeuralModel.load(model_file, device=device)
    device = model.device
    dtype = model.dtype
    dt = model.dt

    stim_index = args.stim_index

    print("Loading inputs and targets...")
    cell_inputs_all = torch.load(model_folder / 'cell_inputs.pt', map_location=device)
    cell_inputs_all = cell_inputs_all.to(device=device, dtype=dtype)
    target_loaded = torch.load(model_folder / 'target.pt', map_location=device)

    if not isinstance(target_loaded, dict) or 'rotation' not in target_loaded:
        print("Error: target.pt does not contain odometry data (expected dict with 'rotation' / 'translation').")
        sys.exit(1)

    true_rot_all = target_loaded['rotation'].to(device=device, dtype=dtype)
    true_trans_all = target_loaded['translation'].to(device=device, dtype=dtype)

    stim_input = cell_inputs_all[stim_index:stim_index + 1]  # (1, T, 2*N)
    true_rot = true_rot_all[stim_index].cpu().numpy()          # (T, 3)
    true_trans = true_trans_all[stim_index].cpu().numpy()      # (T, 3)

    N = stim_input.shape[2] // 2
    right_input = stim_input[:, :, :N]
    left_input = stim_input[:, :, N:]

    print(f"Running ego-motion estimation for stimulus {stim_index} ({stim_input.shape[1]} frames)...")
    with torch.no_grad():
        pred_rot, pred_trans, _, _, _ = model.estimate_ego_motion_from_image(right_input, left_input)

    pred_rot = pred_rot[0].cpu().numpy()    # (T, 3)
    pred_trans = pred_trans[0].cpu().numpy() # (T, 3)

    T = true_rot.shape[0]
    time = np.arange(T) * dt

    # --- Plot ---
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)

    for i in range(3):
        ax_rot = axes[i, 0]
        ax_rot.plot(time, true_rot[:, i], label='true', linewidth=1)
        ax_rot.plot(time, pred_rot[:, i], label='predicted', linewidth=1, alpha=0.8)
        ax_rot.set_ylabel(f'Rotation {AXIS_LABELS[i]}')
        if i == 0:
            ax_rot.legend(fontsize=8)
            ax_rot.set_title('Rotation (deg/s)')

        ax_trans = axes[i, 1]
        ax_trans.plot(time, true_trans[:, i], label='true', linewidth=1)
        ax_trans.plot(time, pred_trans[:, i], label='predicted', linewidth=1, alpha=0.8)
        ax_trans.set_ylabel(f'Translation {AXIS_LABELS[i]}')
        if i == 0:
            ax_trans.legend(fontsize=8)
            ax_trans.set_title('Translation')

    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 1].set_xlabel('Time (s)')

    fig.suptitle(f'Odometry — stimulus {stim_index}', fontsize=13)
    plt.tight_layout()

    out_path = model_folder / f'odometry_stim_{stim_index}.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
