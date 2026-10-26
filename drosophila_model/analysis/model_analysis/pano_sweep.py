"""
Panoramic odometry benchmark.

Generates random panoramic stimuli with smoothly varying rotation (one per
panorama in ``all_panoramic.pt``, for the first ``--num-panos`` panoramas),
runs the model's two-eye ego-motion pipeline, and saves ground-truth vs
predicted odometry figures (rate and integrated pose) for each stimulus.

Usage:
    python -m drosophila_model.analysis.model_analysis.pano_sweep models/<model>/
    python -m drosophila_model.analysis.model_analysis.pano_sweep models/<model>/ --num-panos 10 --stim-time 10
"""
import argparse
from pathlib import Path


import numpy as np
import torch

from drosophila_model.model_definition import neural_models as nm
from drosophila_model.data_handling import stimulus_generators as su
from drosophila_model.analysis.model_analysis.analysis_utilities import (
    cos_sim, save_integrated_figure, save_rate_figure,
)


def run_pano_sweep(
    model_path: str | Path,
    num_panos: int = 5,
    stim_time: float = 5.0,
    pre_stim_time: float = 0.5,
    velocity_amp: float = 100.0,
    velocity_time_scale: float = 0.3,
    seed: int = 0,
) -> dict:
    """Run pano sweep and return results.

    Panorama ``i`` (``i < num_panos``) gets its own random velocity trace,
    all drawn from ``seed``.

    Returns:
        dict with keys 'rot_cosines' (list[float]) and
        'median_rot_cos' (float).
    """
    model_path = Path(model_path)
    model_folder = model_path.parent if model_path.is_file() else model_path

    print(f"Loading model from {model_path}...")
    model = nm.BaseRnn.load(model_path)
    model.eval()

    model_dt = model.dt
    device = model.device
    dtype = model.dtype
    column_phi_theta = model.regions['optic_lobe'].column_phi_theta
    N = column_phi_theta.shape[0]

    both_phi = np.concatenate([column_phi_theta[:, 0], -column_phi_theta[:, 0]])
    both_theta = np.concatenate([column_phi_theta[:, 1], column_phi_theta[:, 1]])

    pre_stim_frames = int(round(pre_stim_time / model_dt))

    output_dir = model_folder / "figures" / "pano_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    # random_pano is a single-viewpoint render: its registered offset is 0.
    camera_offset = su.camera_offset_for_stimulus('random_pano')

    num_available = len(su._load_all_panos())
    if num_panos > num_available:
        print(f"Only {num_available} panoramas available; using all of them.")
        num_panos = num_available

    # One generator call gives each of the first num_panos panoramas its own
    # velocity trace, ordered pano-major with 2 * num_repeats (symmetrised)
    # repeats per panorama, so example 2 * i is panorama i's first repeat.
    print(f"\nGenerating random-velocity panoramas (seed={seed})...")
    stimulus_both, _, rotation_vectors, oob_mask = su.get_random_velocity_pano(
        both_phi, both_theta,
        velocity_amp=velocity_amp,
        velocity_time_scale=velocity_time_scale,
        num_repeats=1,
        dt=model_dt,
        stim_time=stim_time,
        seed=seed,
        pano_indices=list(range(num_panos)),
    )

    rot_cosines = []

    for pi in range(num_panos):
        print(f"\n--- Pano {pi} ---")

        stim = stimulus_both[2 * pi]
        rot_vectors = rotation_vectors[2 * pi]
        oob = oob_mask[2 * pi]

        true_rot_rate = rot_vectors * (180.0 / np.pi)
        true_trans_rate = np.zeros_like(true_rot_rate)

        stimulus_right_np = stim[:, :N]
        stimulus_left_np = stim[:, N:]
        oob_right = oob[:, :N]
        oob_left = oob[:, N:]

        # Prepend pre-stimulus interval
        if pre_stim_frames > 0:
            pre = np.full((pre_stim_frames, N), 0.5)
            stimulus_right_np = np.concatenate([pre, stimulus_right_np], axis=0)
            stimulus_left_np = np.concatenate([pre, stimulus_left_np], axis=0)
            pre_rot = np.zeros((pre_stim_frames, 3))
            true_rot_rate = np.concatenate([pre_rot, true_rot_rate], axis=0)
            true_trans_rate = np.concatenate([pre_rot, true_trans_rate], axis=0)
            pre_oob = np.zeros((pre_stim_frames, N), dtype=bool)
            oob_right = np.concatenate([pre_oob, oob_right], axis=0)
            oob_left = np.concatenate([pre_oob, oob_left], axis=0)

        right_input = torch.tensor(stimulus_right_np, device=device, dtype=dtype).unsqueeze(0)
        left_input = torch.tensor(stimulus_left_np, device=device, dtype=dtype).unsqueeze(0)

        oob_always_right = np.all(oob_right, axis=0)
        oob_always_left = np.all(oob_left, axis=0)
        oob_columns = torch.tensor(
            np.concatenate([oob_always_right, oob_always_left]), device=device,
        )

        num_frames = stimulus_right_np.shape[0]
        print(f"  Running model on {num_frames} frames...")
        with torch.no_grad():
            pred_rot, pred_trans, _, _ = model.regions['optic_lobe'].estimate_ego_motion_from_image(
                right_input, left_input, oob_columns=oob_columns,
                camera_offset=camera_offset,
            )

        pred_rot_np = pred_rot[0].cpu().numpy()
        pred_trans_np = pred_trans[0].cpu().numpy()

        rot_cos = cos_sim(true_rot_rate[pre_stim_frames:], pred_rot_np[pre_stim_frames:])
        rot_cosines.append(rot_cos)
        print(f"  Rotation cosine similarity: {rot_cos:.4f}")

        title = f"Pano {pi}"
        save_rate_figure(
            true_rot_rate, pred_rot_np, true_trans_rate, pred_trans_np,
            model_dt, title, output_dir / f"pano_{pi}_rate.png",
        )
        save_integrated_figure(
            true_rot_rate, pred_rot_np, true_trans_rate, pred_trans_np,
            model_dt, title, output_dir / f"pano_{pi}_integrated.png",
        )

    median_rot = float(np.median(rot_cosines))

    print(f"\n{'='*50}")
    print(f"Pano sweep summary ({num_panos} stimuli):")
    for i, c in enumerate(rot_cosines):
        print(f"  Pano {i}: rot cos = {c:.4f}")
    print(f"  Median: {median_rot:.4f}  Std: {np.std(rot_cosines):.4f}")
    print(f"Figures saved to {output_dir}")

    return {"rot_cosines": rot_cosines, "median_rot_cos": median_rot}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run model on random panoramic stimuli and save odometry figures."
    )
    parser.add_argument("model_folder", type=str, help="Path to model folder or model.pt")
    parser.add_argument("--num-panos", type=int, default=5,
                        help="Number of random pano stimuli to generate (default: 5)")
    parser.add_argument("--stim-time", type=float, default=5.0,
                        help="Stimulus duration in seconds (default: 5.0)")
    parser.add_argument("--pre-stim-time", type=float, default=0.5,
                        help="Pre-stimulus duration in seconds (default: 0.5)")
    parser.add_argument("--velocity-amp", type=float, default=100.0,
                        help="RMS rotation speed in deg/s (default: 100.0)")
    parser.add_argument("--velocity-time-scale", type=float, default=0.3,
                        help="Velocity smoothing time scale (default: 0.3)")
    args = parser.parse_args()

    run_pano_sweep(
        args.model_folder,
        num_panos=args.num_panos,
        stim_time=args.stim_time,
        pre_stim_time=args.pre_stim_time,
        velocity_amp=args.velocity_amp,
        velocity_time_scale=args.velocity_time_scale,
    )


if __name__ == "__main__":
    main()
