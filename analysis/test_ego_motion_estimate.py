"""
Test ego motion estimation from panoramic stimuli.

Generates random-velocity panoramic stimuli for both eyes, runs them through
the model's estimate_ego_motion_from_image, and compares the predicted
rotation to the ground-truth body-frame angular velocity.

Usage:
    python analysis/test_ego_motion_estimate.py <model_path> [--device cpu]
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import neural_models as nm
import stimulus_utilities as su


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test ego motion estimation against ground truth."
    )
    parser.add_argument(
        "model_folder",
        type=str,
        help="Path to model folder or model.pt file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load the model on (default: cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stimulus generation (default: 42)",
    )
    parser.add_argument(
        "--num-repeats",
        type=int,
        default=1,
        help="Number of repeats per panorama (default: 1)",
    )
    parser.add_argument(
        "--num-stimuli",
        type=int,
        default=None,
        help="Number of repeats per panorama (default: 1)",
    )
    args = parser.parse_args()

    print(f"Loading model from {args.model_folder} ...")
    model = nm.NeuralModel.load(args.model_folder, device=args.device)
    model.eval()

    # Column locations for right eye and left eye
    right_phi_theta = model.column_phi_theta          # (N, 2)
    left_phi_theta = right_phi_theta.copy()
    left_phi_theta[:, 0] = -left_phi_theta[:, 0]     # negate phi for left eye

    # Concatenate both eyes so they experience the same rotation sequence
    both_phi = np.concatenate([right_phi_theta[:, 0], left_phi_theta[:, 0]])
    both_theta = np.concatenate([right_phi_theta[:, 1], left_phi_theta[:, 1]])
    num_columns = right_phi_theta.shape[0]

    # Generate stimuli with ground-truth rotation vectors
    print("Generating panoramic stimuli ...")
    stimulus, optic_flow_true, rotation_vectors, _ = su.get_random_velocity_pano(
        both_phi, both_theta,
        velocity_amp=100, velocity_time_scale=0.5,
        seed=args.seed, dt=model.dt, stim_time=1.0,
        num_repeats=args.num_repeats,
    )
    if args.num_stimuli is not None:
        stimulus = stimulus[:args.num_stimuli]
        optic_flow_true = optic_flow_true[:args.num_stimuli]
        rotation_vectors = rotation_vectors[:args.num_stimuli]
    # stimulus: (examples, time, 2*N)
    # rotation_vectors: (examples, time, 3)  — body-frame omega in rad/s

    # Split into right and left eye inputs
    right_stim = torch.tensor(
        stimulus[:, :, :num_columns], device=args.device, dtype=model.dtype
    )
    left_stim = torch.tensor(
        stimulus[:, :, num_columns:], device=args.device, dtype=model.dtype
    )

    # -------------------------------------------------------------------------
    # DEBUG: Check optic flow predictions against ground truth
    # -------------------------------------------------------------------------
    optic_flow_true_tensor = torch.tensor(
        optic_flow_true, device=args.device, dtype=model.dtype
    )
    # optic_flow_true_tensor: (stimuli, time, 2*N, 2) — columns are [right; left]
    # Split into per-eye ground truth
    of_true_right = optic_flow_true_tensor[:, :, :num_columns, :]
    of_true_left = optic_flow_true_tensor[:, :, num_columns:, :]

    print("\nRunning per-eye model predictions for optic flow debug ...")
    with torch.no_grad():
        outputs_right = model.predict(right_stim, init_state=None)['outputs']
        outputs_left = model.predict(left_stim, init_state=None)['outputs']

    of_pred_right = outputs_right['decoder']   # (stimuli, time, N, 2)
    of_pred_left = outputs_left['decoder']     # (stimuli, time, N, 2)

    # The left eye decoder predicts flow in the left-eye frame (phi negated).
    # The ground truth left-eye flow was generated in the global frame, so
    # its d_right component has the opposite sign from the left-eye-frame.
    # Flip the left decoder's d_right to match the global-frame ground truth.
    of_pred_left_global = of_pred_left.clone()
    of_pred_left_global[:, :, :, 0] = -of_pred_left_global[:, :, :, 0]

    print("\nOptic flow cosine similarity (predicted vs true), per stimulus:")
    print(f"  {'Stim':>5s}  {'Right eye':>10s}  {'Left eye':>10s}  {'Combined':>10s}")
    for s in range(right_stim.shape[0]):
        # Mask NaN columns in ground truth
        valid_r = ~torch.isnan(of_true_right[s]).any(dim=-1).any(dim=0)  # (N,)
        valid_l = ~torch.isnan(of_true_left[s]).any(dim=-1).any(dim=0)   # (N,)

        pr = of_pred_right[s, :, valid_r, :].reshape(-1)
        tr = of_true_right[s, :, valid_r, :].reshape(-1)
        cos_r = torch.nn.functional.cosine_similarity(pr, tr, dim=0).item()

        pl = of_pred_left_global[s, :, valid_l, :].reshape(-1)
        tl = of_true_left[s, :, valid_l, :].reshape(-1)
        cos_l = torch.nn.functional.cosine_similarity(pl, tl, dim=0).item()

        pc = torch.cat([pr, pl])
        tc = torch.cat([tr, tl])
        cos_c = torch.nn.functional.cosine_similarity(pc, tc, dim=0).item()

        print(f"  {s:5d}  {cos_r:10.4f}  {cos_l:10.4f}  {cos_c:10.4f}")

    # Also print magnitude stats
    valid_all = ~torch.isnan(optic_flow_true_tensor).any(dim=-1)  # (stim, time, 2*N)
    of_pred_combined = torch.cat([of_pred_right, of_pred_left_global], dim=2)
    pred_mags = of_pred_combined[valid_all.unsqueeze(-1).expand_as(of_pred_combined)].abs().mean()
    true_mags = optic_flow_true_tensor[valid_all.unsqueeze(-1).expand_as(optic_flow_true_tensor)].abs().mean()
    print(f"\n  Mean |predicted flow|: {pred_mags.item():.4f}")
    print(f"  Mean |true flow|:      {true_mags.item():.4f}")
    print(f"  Ratio:                 {pred_mags.item() / (true_mags.item() + 1e-12):.4f}")
    # -------------------------------------------------------------------------

    # Ground truth: convert rad/s → deg/s to match decoder-output units
    omega_true = torch.tensor(
        rotation_vectors * (180.0 / np.pi), device=args.device, dtype=model.dtype
    )

    # Estimate ego motion
    print("Running ego motion estimation ...")
    with torch.no_grad():
        rotations, translations, _, _ = model.estimate_ego_motion_from_image(
            right_stim, left_stim
        )
    # rotations: (stimuli, time, 3) — predicted rotation in deg/s

    # --- Cosine similarity ---
    cos_sim = torch.nn.functional.cosine_similarity(
        rotations, omega_true, dim=-1
    )  # (stimuli, time)

    print(f"\nStimuli: {rotations.shape[0]},  Time steps: {rotations.shape[1]}")
    print(f"\nCosine similarity (predicted vs true rotation):")
    print(f"  Mean:   {cos_sim.mean().item():.4f}")
    print(f"  Median: {cos_sim.median().item():.4f}")
    print(f"  Min:    {cos_sim.min().item():.4f}")
    print(f"  Max:    {cos_sim.max().item():.4f}")

    # --- Magnitude comparison ---
    pred_mag = rotations.norm(dim=-1)
    true_mag = omega_true.norm(dim=-1)
    mag_ratio = pred_mag / (true_mag + 1e-8)

    print(f"\nRotation magnitude ratio (predicted / true):")
    print(f"  Mean:   {mag_ratio.mean().item():.4f}")
    print(f"  Median: {mag_ratio.median().item():.4f}")

    # --- Translation magnitude (should be ~0 for pure rotation) ---
    trans_mag = translations.norm(dim=-1)
    print(f"\nTranslation magnitude (should be ~0 for pure rotation):")
    print(f"  Mean:   {trans_mag.mean().item():.6f}")

    # --- Per-stimulus summary ---
    print(f"\nPer-stimulus mean cosine similarity:")
    for s in range(cos_sim.shape[0]):
        print(f"  Stimulus {s:3d}: {cos_sim[s].mean().item():.4f}")

    # -------------------------------------------------------------------------
    # TEST: Ego motion estimation with ground-truth optic flow
    # -------------------------------------------------------------------------
    # Instead of using the decoder's predicted flow, plug the true optic flow
    # (from the stimulus generator) directly into estimate_ego_motion_from_flow.
    # This isolates the ego-motion solver from the decoder quality.
    print("\n" + "=" * 70)
    print("TEST: Ego motion estimation using ground-truth optic flow")
    print("=" * 70)

    # optic_flow_true_tensor: (stimuli, time, 2*N, 2) in deg/s
    # Column order is [right; left] matching both_phi / both_theta
    optic_flow_gt = optic_flow_true_tensor

    phi_tensor = torch.tensor(both_phi, device=args.device, dtype=model.dtype)
    theta_tensor = torch.tensor(both_theta, device=args.device, dtype=model.dtype)

    num_stimuli_gt = optic_flow_gt.shape[0]
    num_time_gt = optic_flow_gt.shape[1]
    rotations_gt = []
    translations_gt = []

    for s in range(num_stimuli_gt):
        rot_t_list = []
        trans_t_list = []
        for t in range(num_time_gt):
            # Per-time-step valid mask: different columns go in/out of the
            # panorama bounds as the observer rotates.
            valid_t = (
                ~torch.isnan(optic_flow_gt[s, t, :, 0])
                & ~torch.isnan(optic_flow_gt[s, t, :, 1])
            )
            d_right_t = optic_flow_gt[s, t, valid_t, 0]  # (valid_cols,)
            d_up_t = optic_flow_gt[s, t, valid_t, 1]
            rotation_t, translation_t, _ = nm.NeuralModel.estimate_ego_motion_from_flow(
                phi_tensor[valid_t], theta_tensor[valid_t], d_right_t, d_up_t
            )
            rot_t_list.append(rotation_t)
            trans_t_list.append(translation_t)
        rotations_gt.append(torch.stack(rot_t_list))
        translations_gt.append(torch.stack(trans_t_list))

    rotations_gt = torch.stack(rotations_gt)       # (stimuli, time, 3)
    translations_gt = torch.stack(translations_gt)  # (stimuli, time, 3)

    # omega_true is already in deg/s; ground-truth flow is in deg/s so
    # the returned rotation estimate is also in deg/s — directly comparable.
    cos_sim_gt = torch.nn.functional.cosine_similarity(
        rotations_gt, omega_true, dim=-1
    )

    print(f"\nStimuli: {rotations_gt.shape[0]},  Time steps: {rotations_gt.shape[1]}")
    print(f"\nCosine similarity (GT-flow estimate vs true rotation):")
    print(f"  Mean:   {cos_sim_gt.mean().item():.4f}")
    print(f"  Median: {cos_sim_gt.median().item():.4f}")
    print(f"  Min:    {cos_sim_gt.min().item():.4f}")
    print(f"  Max:    {cos_sim_gt.max().item():.4f}")

    pred_mag_gt = rotations_gt.norm(dim=-1)
    true_mag_gt = omega_true.norm(dim=-1)
    mag_ratio_gt = pred_mag_gt / (true_mag_gt + 1e-8)

    print(f"\nRotation magnitude ratio (GT-flow estimate / true):")
    print(f"  Mean:   {mag_ratio_gt.mean().item():.4f}")
    print(f"  Median: {mag_ratio_gt.median().item():.4f}")

    trans_mag_gt = translations_gt.norm(dim=-1)
    print(f"\nTranslation magnitude (should be ~0 for pure rotation):")
    print(f"  Mean:   {trans_mag_gt.mean().item():.6f}")

    print(f"\nPer-stimulus mean cosine similarity (GT flow):")
    for s in range(cos_sim_gt.shape[0]):
        print(f"  Stimulus {s:3d}: {cos_sim_gt[s].mean().item():.4f}")


if __name__ == "__main__":
    main()
