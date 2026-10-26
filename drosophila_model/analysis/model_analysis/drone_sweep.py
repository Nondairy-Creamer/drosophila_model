"""
Drone odometry benchmark.

Loads drone video clips (the baked ``<data_folder>/stimulus/baked/<dataset>.pt``
cache), runs the model's two-eye ego-motion pipeline on each full-length clip
with the dataset's registered camera offset, and saves ground-truth vs
predicted odometry figures (rate and integrated pose) to
``<model_folder>/figures/drone_sweep/``.

Usage (from the repo root):
    python -m drosophila_model.analysis.model_analysis.drone_sweep models/my_model/
    python -m drosophila_model.analysis.model_analysis.drone_sweep models/my_model/ --num-clips 5
    python -m drosophila_model.analysis.model_analysis.drone_sweep models/my_model/ --dataset rpi_drone
"""
import argparse
from pathlib import Path


import numpy as np
import torch

from drosophila_model.config import get_data_folder, load_config
from drosophila_model.model_definition import neural_models as nm
from drosophila_model.data_handling import drone_utilities as du
from drosophila_model.data_handling import stimulus_generators as su
from drosophila_model.analysis.model_analysis.analysis_utilities import (
    cos_sim, save_integrated_figure, save_rate_figure,
)

DRONE_DATASETS = ("small_drone", "rpi_drone")


def run_drone_sweep(
    model_path: str | Path,
    num_clips: int = 3,
    dataset: str = "small_drone",
) -> dict:
    """Run drone sweep and return results.

    ``dataset`` is a drone stimulus type (``'small_drone'`` or
    ``'rpi_drone'``); its registered camera offset
    (:func:`stimulus_generators.camera_offset_for_stimulus`) is used by the
    two-eye ego-motion solver, as in training.

    Returns:
        dict with keys 'rot_cosines', 'trans_cosines' (list[float]),
        'median_rot_cos', 'median_trans_cos' (float).
    """
    if dataset not in DRONE_DATASETS:
        raise ValueError(f"dataset must be one of {DRONE_DATASETS}, got {dataset!r}")
    model_path = Path(model_path)
    model_folder = model_path.parent if model_path.is_file() else model_path

    print(f"Loading model from {model_path}...")
    model = nm.BaseRnn.load(model_path)
    model.eval()

    model_dt = model.dt
    device = model.device
    dtype = model.dtype
    column_phi_theta = model.regions['optic_lobe'].column_phi_theta
    camera_offset = su.camera_offset_for_stimulus(dataset)

    print(f"Processing {dataset} clips...")
    drone_clips = du.get_drone_stimulus(
        model_dt, dataset=dataset, data_folder=get_data_folder(load_config()),
        sample_phi=column_phi_theta[:, 0], sample_theta=column_phi_theta[:, 1],
    )
    num_clips = min(num_clips, len(drone_clips))
    print(f"Found {len(drone_clips)} clips, evaluating {num_clips}")

    output_dir = model_folder / "figures" / "drone_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    rot_cosines = []
    trans_cosines = []

    for ci in range(num_clips):
        clip = drone_clips[ci]
        clip_id = clip["clip_id"]
        num_frames = clip["right"].shape[0]

        print(f"\n--- Clip {ci} (id={clip_id}, {num_frames} frames, "
              f"{num_frames * model_dt:.1f}s) ---")

        right_input = torch.tensor(
            clip["right"], device=device, dtype=dtype).unsqueeze(0)
        left_input = torch.tensor(
            clip["left"], device=device, dtype=dtype).unsqueeze(0)

        in_fov_right = clip["in_fov_right"].numpy()
        in_fov_left = clip["in_fov_left"].numpy()
        oob = torch.tensor(
            np.concatenate([~in_fov_right, ~in_fov_left]), device=device,
        )

        print("  Running model...")
        with torch.no_grad():
            pred_rot, pred_trans, _, _ = model.regions['optic_lobe'].estimate_ego_motion_from_image(
                right_input, left_input, oob_columns=oob,
                camera_offset=camera_offset,
            )

        pred_rot_np = pred_rot[0].cpu().numpy()
        pred_trans_np = pred_trans[0].cpu().numpy()

        true_rot_rate = clip["rotation_rate"][:num_frames]
        true_trans_rate = clip["translation_rate"][:num_frames]

        # Scale predicted translation to match ground truth magnitude
        trans_scale = np.std(true_trans_rate) / (np.std(pred_trans_np) + 1e-12)
        pred_trans_scaled = pred_trans_np * trans_scale

        rot_cos = cos_sim(true_rot_rate, pred_rot_np)
        trans_cos = cos_sim(true_trans_rate, pred_trans_np)
        rot_cosines.append(rot_cos)
        trans_cosines.append(trans_cos)
        print(f"  Rotation  cosine similarity: {rot_cos:.4f}")
        print(f"  Translation cosine similarity: {trans_cos:.4f}")

        title = f"{dataset} clip {ci} (id={clip_id})"
        save_rate_figure(
            true_rot_rate, pred_rot_np, true_trans_rate, pred_trans_scaled,
            model_dt, title, output_dir / f"{dataset}_{ci}_rate.png",
        )
        save_integrated_figure(
            true_rot_rate, pred_rot_np, true_trans_rate, pred_trans_scaled,
            model_dt, title, output_dir / f"{dataset}_{ci}_integrated.png",
        )

    median_rot = float(np.median(rot_cosines))
    median_trans = float(np.median(trans_cosines))

    print(f"\n{'='*50}")
    print(f"Drone sweep summary ({dataset}, {num_clips} clips):")
    for i, (rc, tc) in enumerate(zip(rot_cosines, trans_cosines)):
        print(f"  Clip {i}: rot cos = {rc:.4f}  trans cos = {tc:.4f}")
    print(f"  Rotation  - median: {median_rot:.4f}  std: {np.std(rot_cosines):.4f}")
    print(f"  Translation - median: {median_trans:.4f}  std: {np.std(trans_cosines):.4f}")
    print(f"Figures saved to {output_dir}")

    return {
        "rot_cosines": rot_cosines,
        "trans_cosines": trans_cosines,
        "median_rot_cos": median_rot,
        "median_trans_cos": median_trans,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run model on drone clips and save odometry figures."
    )
    parser.add_argument("model_folder", type=str, help="Path to model folder or model.pt")
    parser.add_argument("--num-clips", type=int, default=3,
                        help="Number of drone clips to evaluate (default: 3)")
    parser.add_argument("--dataset", choices=DRONE_DATASETS, default="small_drone",
                        help="Drone dataset to evaluate (default: small_drone)")
    args = parser.parse_args()

    run_drone_sweep(args.model_folder, num_clips=args.num_clips, dataset=args.dataset)


if __name__ == "__main__":
    main()
