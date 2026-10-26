"""
Plot predicted vs true odometry (rotation and translation) for a two-eye model.

Replays the model's own training stimuli (``cell_inputs.pt`` / ``target.pt``,
the training split) through the two-eye ego-motion solver with the run's
camera offset, excluding columns that are never in view (as training does).

Layout (3 x 2):
  Left column: Rotation x, y, z  (deg/s)
  Right column: Translation x, y, z

Usage:
    python -m drosophila_model.analysis.model_analysis.plot_odometry models/<model>/
    python -m drosophila_model.analysis.model_analysis.plot_odometry models/<model>/ --stim-index 5
    python -m drosophila_model.analysis.model_analysis.plot_odometry models/<model>/ --num-stimuli 10
"""
import argparse
import sys
from pathlib import Path


import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from drosophila_model.config import load_run_config
from drosophila_model.model_definition import neural_models as nm
from drosophila_model.analysis.model_analysis.analysis_utilities import run_eye_offset


AXIS_LABELS = ['x (roll)', 'y (pitch)', 'z (yaw)']


def plot_odometry_single(model, cell_inputs_all, train_mask_all, true_rot_all, true_trans_all,
                         stim_index, dt, output_dir, eye_offset):
    """Generate and save an odometry plot for a single stimulus."""
    stim_input = cell_inputs_all[stim_index:stim_index + 1]
    # Columns never in view are excluded from the solve, as in training.
    oob = ~train_mask_all[stim_index].any(dim=0)
    true_rot = true_rot_all[stim_index].cpu().numpy()
    true_trans = true_trans_all[stim_index].cpu().numpy()

    N = stim_input.shape[2] // 2
    right_input = stim_input[:, :, :N]
    left_input = stim_input[:, :, N:]

    print(f"  Stimulus {stim_index} ({stim_input.shape[1]} frames)...")
    with torch.no_grad():
        pred_rot, pred_trans, _, _ = model.regions['optic_lobe'].estimate_ego_motion_from_image(
            right_input, left_input, oob_columns=oob, camera_offset=eye_offset)

    pred_rot = pred_rot[0].cpu().numpy()
    pred_trans = pred_trans[0].cpu().numpy()

    # Normalize each translation axis by its own std (unknown scale factor).
    # Skip if either trace has zero std to avoid division by zero.
    for i in range(3):
        true_std = np.std(true_trans[:, i])
        pred_std = np.std(pred_trans[:, i])
        if true_std > 0 and pred_std > 0:
            true_trans[:, i] /= true_std
            pred_trans[:, i] /= pred_std

    T = true_rot.shape[0]
    time = np.arange(T) * dt

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
            ax_trans.set_title('Translation (normalized)')

    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 1].set_xlabel('Time (s)')

    fig.suptitle(f'Odometry — stimulus {stim_index}', fontsize=13)
    plt.tight_layout()

    out_path = output_dir / f'odometry_stim_{stim_index}.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot predicted vs true odometry.")
    parser.add_argument(
        "model_folder", type=str,
        help="Path to model folder or model.pt",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--stim-index", type=int, default=None,
                       help="Single stimulus index to visualize")
    group.add_argument("--num-stimuli", type=int, default=10,
                       help="Number of stimuli to plot, evenly sampled (default: 10)")
    args = parser.parse_args()

    model_path = Path(args.model_folder)
    model_folder = model_path.parent if model_path.is_file() else model_path
    print(f"Loading model from {model_path}...")
    model = nm.BaseRnn.load(model_path)
    device = model.device
    dtype = model.dtype
    dt = model.dt

    eye_offset = run_eye_offset(load_run_config(model_folder))

    print("Loading inputs and targets...")
    cell_inputs_data = torch.load(model_folder / 'cell_inputs.pt', map_location=device)
    cell_inputs_all = cell_inputs_data['cell_inputs'].to(device=device, dtype=dtype)
    train_mask_all = cell_inputs_data['train_mask'].to(device=device)
    target_loaded = torch.load(model_folder / 'target.pt', map_location=device)

    if 'rotation' not in target_loaded:
        sys.exit("Error: target.pt has no odometry data ('rotation' / 'translation'); "
                 "the model was not trained on two-eye data.")

    true_rot_all = target_loaded['rotation'].to(device=device, dtype=dtype)
    true_trans_all = target_loaded['translation'].to(device=device, dtype=dtype)

    num_total = cell_inputs_all.shape[0]

    if args.stim_index is not None:
        if not 0 <= args.stim_index < num_total:
            sys.exit(f"--stim-index {args.stim_index} is out of range (0..{num_total - 1}).")
        stim_indices = [args.stim_index]
    else:
        n = min(args.num_stimuli, num_total)
        stim_indices = np.linspace(0, num_total - 1, n, dtype=int).tolist()

    output_dir = model_folder / 'figures' / 'plot_odometry'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Plotting odometry for {len(stim_indices)} stimuli (out of {num_total} total)...")
    for idx in stim_indices:
        plot_odometry_single(model, cell_inputs_all, train_mask_all, true_rot_all, true_trans_all,
                             idx, dt, output_dir, eye_offset)

    print("Done!")


if __name__ == "__main__":
    main()
