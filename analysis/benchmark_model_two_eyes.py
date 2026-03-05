"""
Benchmark a two-eye model on random_pano and drone_view stimuli.

Loads a trained model, randomly selects 100 stimuli of each type via
get_stimulus_two_eyes, runs estimate_ego_motion_from_image on the full
batch simultaneously, and prints per-stimulus cosine similarity between
true and predicted odometry.

Usage:
    python analysis/benchmark_model_two_eyes.py models/best_pano/
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

import data_utilities as du
import neural_models as nm
import stimulus_utilities as su

from analysis_utilities import resolve_model_paths


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two flattened arrays."""
    a_flat, b_flat = a.ravel(), b.ravel()
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a_flat, b_flat) / denom)


def run_benchmark(model, config, stimulus_type: str, num_stimuli: int, device: str):
    """Generate stimuli, run the model in one batch, return per-stimulus cosine sims."""
    column_phi_theta = model.column_phi_theta
    N = column_phi_theta.shape[0]

    stimulus, optic_flow, true_rot, true_trans, train_mask = su.get_stimulus_two_eyes(
        stimulus_type, config, column_phi_theta=column_phi_theta, add_blank=False,
    )

    n_available = stimulus.shape[0]
    n_use = min(num_stimuli, n_available)
    if n_use < num_stimuli:
        print(f"  Warning: only {n_available} stimuli available, using {n_use}")

    rng = np.random.default_rng()
    indices = rng.choice(n_available, size=n_use, replace=False)

    # Find first valid time step (pre-stim is a contiguous block at the start)
    time_mask = train_mask.any(dim=-1)  # (examples, time)
    loss_start_step = 0
    if not time_mask[:, 0].all():
        loss_start_step = time_mask.all(dim=0).to(torch.int).argmax().item()
    oob = ~train_mask.any(dim=1).any(dim=0).to(device=device)  # (2*N,)

    # Run all selected stimuli through the model in a single batched call
    right_input = stimulus[indices, :, :N].to(device=device, dtype=model.dtype)
    left_input = stimulus[indices, :, N:].to(device=device, dtype=model.dtype)

    with torch.no_grad():
        pred_rot, pred_trans, _, _ = model.estimate_ego_motion_from_image(
            right_input, left_input, oob_columns=oob,
        )

    rot_cosines = []
    trans_cosines = []

    for batch_idx, orig_idx in enumerate(indices):
        pred_rot_np = pred_rot[batch_idx, loss_start_step:].cpu().numpy()
        pred_trans_np = pred_trans[batch_idx, loss_start_step:].cpu().numpy()
        true_rot_np = true_rot[orig_idx, loss_start_step:].cpu().numpy()
        true_trans_np = true_trans[orig_idx, loss_start_step:].cpu().numpy()

        rot_cos = _cos_sim(true_rot_np, pred_rot_np)
        trans_cos = _cos_sim(true_trans_np, pred_trans_np)
        rot_cosines.append(rot_cos)
        trans_cosines.append(trans_cos)

        print(f"  [{orig_idx:3d}] rot cos={rot_cos:+.4f}  trans cos={trans_cos:+.4f}")

    return rot_cosines, trans_cosines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark two-eye model on random_pano and drone_view stimuli."
    )
    parser.add_argument("model_folder", type=str, help="Path to model folder or model.pt")
    parser.add_argument("--num-stimuli", type=int, default=100,
                        help="Number of stimuli per type (default: 100)")
    args = parser.parse_args()

    model_file, model_folder = resolve_model_paths(Path(args.model_folder))
    computer_config = du.load_config()
    model_config = du.load_config(model_folder / 'config.toml')
    device = computer_config["model"].get("device", "cpu")
    model_config["model"]["device"] = device

    print(f"Loading model from {model_file} on {device}...")
    model = nm.NeuralModel.load(model_file, device=device)
    model.eval()

    stimulus_types = ["random_pano", "drone_view"]

    for stim_type in stimulus_types:
        print(f"\n{'='*60}")
        print(f"Stimulus: {stim_type}")
        print(f"{'='*60}")

        rot_cosines, trans_cosines = run_benchmark(
            model, model_config, stim_type, args.num_stimuli, device,
        )

        median_rot = np.median(rot_cosines)
        median_trans = np.median(trans_cosines)
        print(f"\n  Summary ({stim_type}):")
        print(f"    Rotation  cos sim: median={median_rot:+.4f}  "
              f"std={np.std(rot_cosines):.4f}  "
              f"min={np.min(rot_cosines):+.4f}  max={np.max(rot_cosines):+.4f}")
        print(f"    Translation cos sim: median={median_trans:+.4f}  "
              f"std={np.std(trans_cosines):.4f}  "
              f"min={np.min(trans_cosines):+.4f}  max={np.max(trans_cosines):+.4f}")


if __name__ == "__main__":
    main()
