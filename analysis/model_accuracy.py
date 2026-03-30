"""
Break down the odometry loss function into its individual components.

Loads a model and its saved stimuli (cell_inputs.pt, target.pt), then
computes each loss term from training_utilities.odometry_loss per stimulus:

  - rot_rate    : normalized MSE on rotation rate
  - trans_rate  : cosine loss on translation rate
  - rot_pos     : normalized MSE on integrated rotation position
  - trans_pos   : cosine loss on integrated translation position
  - flow        : normalized MSE on optic flow
  - flow_cos    : cosine distance on optic flow (NaN-safe)
  - odom_total  : mean of the four odometry terms
  - total       : odom_total + flow

Usage:
    python analysis/model_accuracy.py models/best_model
    python analysis/model_accuracy.py models/best_model --stim-index 3
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

import neural_models as nm
from loss_functions import normalized_mse, cosine_loss


def compute_loss_components(model, cell_inputs, target_rot, target_trans,
                            target_flow, train_mask, loss_start_step=0):
    """Compute each odometry loss component for a batch of stimuli.

    Returns a dict of tensors, each of shape ``(num_stimuli,)``.
    """
    device = model.device
    dtype = model.dtype
    dt = model.dt

    N = cell_inputs.shape[2] // 2
    right_input = cell_inputs[:, :, :N]
    left_input = cell_inputs[:, :, N:]

    oob = ~train_mask.any(dim=1).any(dim=0)

    with torch.no_grad():
        pred_rot, pred_trans, right_res, left_res = model.estimate_ego_motion_from_image(
            right_input, left_input, oob_columns=oob,
        )
        flow_key = next(k for k, m in model._readout_modules.items() if m.predicts_flow)
        pred_flow = torch.cat([right_res['outputs'][flow_key], left_res['outputs'][flow_key]], dim=2)

    num_stimuli = cell_inputs.shape[0]
    results = {k: [] for k in [
        'rot_rate', 'trans_rate', 'rot_pos', 'trans_pos',
        'flow', 'flow_cos', 'odom_total', 'total',
    ]}

    for i in range(num_stimuli):
        pr = pred_rot[i:i+1, loss_start_step:]
        tr = target_rot[i:i+1, loss_start_step:]
        pt = pred_trans[i:i+1, loss_start_step:]
        tt = target_trans[i:i+1, loss_start_step:]

        rot_rate = normalized_mse(pr, tr)
        trans_rate = cosine_loss(pt, tt)

        pr_pos = nm.integrate_angular_velocity(pr, dt)
        tr_pos = nm.integrate_angular_velocity(tr, dt)
        rot_pos = normalized_mse(pr_pos, tr_pos)

        pt_pos = nm.integrate_translational_velocity(pt, pr, dt)
        tt_pos = nm.integrate_translational_velocity(tt, tr, dt)
        trans_pos = cosine_loss(pt_pos, tt_pos)

        pf = pred_flow[i:i+1, loss_start_step:]
        tf = target_flow[i:i+1, loss_start_step:]
        fm = train_mask[i:i+1, loss_start_step:].unsqueeze(-1).expand_as(tf)
        valid_flow = fm & ~torch.isnan(tf)
        if valid_flow.any():
            flow_loss = normalized_mse(pf[valid_flow], tf[valid_flow])
        else:
            flow_loss = pf.new_tensor(0.0)

        flow_cos = cosine_loss(pf, tf)

        odom_total = (rot_rate + trans_rate + rot_pos + trans_pos) / 4
        total = odom_total + flow_loss

        results['rot_rate'].append(rot_rate.item())
        results['trans_rate'].append(trans_rate.item())
        results['rot_pos'].append(rot_pos.item())
        results['trans_pos'].append(trans_pos.item())
        results['flow'].append(flow_loss.item())
        results['flow_cos'].append(flow_cos.item())
        results['odom_total'].append(odom_total.item())
        results['total'].append(total.item())

    return {k: np.array(v) for k, v in results.items()}


def plot_flow_cosine(results, stim_indices, output_path):
    """Bar plot of per-stimulus flow cosine distance."""
    flow_cos = results['flow_cos']
    x = np.arange(len(stim_indices))

    fig, ax = plt.subplots(figsize=(max(6, len(stim_indices) * 0.4), 4))
    ax.bar(x, flow_cos, color='steelblue', edgecolor='white', linewidth=0.5)
    ax.axhline(flow_cos.mean(), color='crimson', linestyle='--', linewidth=1,
               label=f'mean = {flow_cos.mean():.4f}')
    ax.set_xticks(x)
    ax.set_xticklabels(stim_indices, fontsize=7, rotation=45 if len(stim_indices) > 20 else 0)
    ax.set_xlabel('Stimulus index')
    ax.set_ylabel('Cosine distance (1 - cos sim)')
    ax.set_title('Flow cosine distance per stimulus')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\nFlow cosine plot saved to {output_path}")


def print_results(results, stim_indices):
    """Pretty-print the per-stimulus loss breakdown and summary statistics."""
    components = ['rot_rate', 'trans_rate', 'rot_pos', 'trans_pos',
                  'flow', 'flow_cos', 'odom_total', 'total']
    headers = ['rot_rate', 'trans_rate', 'rot_pos', 'trans_pos',
               'flow', 'flow_cos', 'odom', 'total']
    col_w = 12

    header_line = f"{'stim':>6s}" + "".join(f"{h:>{col_w}s}" for h in headers)
    print(header_line)
    print("-" * len(header_line))

    for idx, si in enumerate(stim_indices):
        vals = "".join(f"{results[c][idx]:>{col_w}.6f}" for c in components)
        print(f"{si:>6d}{vals}")

    print("-" * len(header_line))
    means = "".join(f"{results[c].mean():>{col_w}.6f}" for c in components)
    stds = "".join(f"{results[c].std():>{col_w}.6f}" for c in components)
    print(f"{'mean':>6s}{means}")
    print(f"{'std':>6s}{stds}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Break down odometry loss into components per stimulus.",
    )
    parser.add_argument(
        "model_folder", type=str,
        help="Path to model folder or model.pt",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--stim-index", type=int, default=None,
        help="Evaluate a single stimulus index only",
    )
    group.add_argument(
        "--num-stim", type=int, default=50,
        help="Number of stimuli to evaluate (default: 50)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device override (default: from config)",
    )
    args = parser.parse_args()

    model_path = Path(args.model_folder)
    model_folder = model_path.parent if model_path.is_file() else model_path

    print(f"Loading model from {model_path}...")
    model = nm.NeuralModel.load(model_path, device=args.device)
    model.eval()
    print('git hash', model.commit_hash)
    device = model.device
    dtype = model.dtype

    print("Loading inputs and targets...")
    cell_inputs_data = torch.load(model_folder / 'cell_inputs.pt', map_location=device)
    if isinstance(cell_inputs_data, dict):
        cell_inputs = cell_inputs_data['cell_inputs'].to(device=device, dtype=dtype)
        train_mask = cell_inputs_data['train_mask'].to(device=device)
    else:
        cell_inputs = cell_inputs_data.to(device=device, dtype=dtype)
        train_mask = torch.ones(cell_inputs.shape[:3], dtype=torch.bool, device=device)

    target_loaded = torch.load(model_folder / 'target.pt', map_location=device)
    if not isinstance(target_loaded, dict) or 'rotation' not in target_loaded:
        print("Error: target.pt does not contain odometry data "
              "(expected dict with 'rotation' / 'translation' / 'optic_flow').")
        sys.exit(1)

    target_rot = target_loaded['rotation'].to(device=device, dtype=dtype)
    target_trans = target_loaded['translation'].to(device=device, dtype=dtype)
    target_flow = target_loaded['optic_flow'].to(device=device, dtype=dtype)

    # Determine loss_start_step from train_mask (pre-stim frames are masked out)
    time_mask = train_mask.any(dim=-1)
    loss_start_step = 0
    if not time_mask[:, 0].all():
        loss_start_step = time_mask.all(dim=0).to(torch.int).argmax().item()

    num_total = cell_inputs.shape[0]
    if args.stim_index is not None:
        stim_indices = [args.stim_index]
        cell_inputs = cell_inputs[args.stim_index:args.stim_index + 1]
        target_rot = target_rot[args.stim_index:args.stim_index + 1]
        target_trans = target_trans[args.stim_index:args.stim_index + 1]
        target_flow = target_flow[args.stim_index:args.stim_index + 1]
        train_mask = train_mask[args.stim_index:args.stim_index + 1]
    else:
        n = min(args.num_stim, num_total)
        stim_indices = list(range(n))
        cell_inputs = cell_inputs[:n]
        target_rot = target_rot[:n]
        target_trans = target_trans[:n]
        target_flow = target_flow[:n]
        train_mask = train_mask[:n]

    print(f"Evaluating {len(stim_indices)} stimuli (loss_start_step={loss_start_step})...\n")

    results = compute_loss_components(
        model, cell_inputs, target_rot, target_trans, target_flow,
        train_mask, loss_start_step=loss_start_step,
    )

    print_results(results, stim_indices)

    output_dir = model_folder / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_flow_cosine(results, stim_indices, output_dir / 'flowcosine_loss.png')


if __name__ == "__main__":
    main()
