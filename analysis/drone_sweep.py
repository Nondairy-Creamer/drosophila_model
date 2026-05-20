"""
Drone odometry benchmark.

Loads drone video clips, runs the model's ego-motion pipeline on each
full-length clip, and saves ground-truth vs predicted odometry figures
(rate and integrated pose).

Usage:
    python analysis/drone_sweep.py models/my_model/
    python analysis/drone_sweep.py models/my_model/ --num-clips 5
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


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_flat, b_flat = a.ravel(), b.ravel()
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    return float(np.dot(a_flat, b_flat) / denom) if denom > 1e-12 else 0.0


def _save_rate_figure(true_rot, pred_rot, true_trans, pred_trans,
                      model_dt, title, output_path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    true_rot, pred_rot = np.asarray(true_rot), np.asarray(pred_rot)
    true_trans, pred_trans = np.asarray(true_trans), np.asarray(pred_trans)

    T = true_rot.shape[0]
    times = np.arange(T) * model_dt
    rot_labels = ["Roll (X)", "Pitch (Y)", "Yaw (Z)"]
    trans_labels = ["X", "Y", "Z"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharex=True)
    for i in range(3):
        ax = axes[i, 0]
        ax.plot(times, true_rot[:, i], linewidth=1.2, label="Ground truth")
        ax.plot(times, pred_rot[:, i], linewidth=1.2, linestyle="--", label="Model")
        ax.set_ylabel(f"{rot_labels[i]} (deg/s)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
            ax.set_title("Rotational Velocity")

        ax = axes[i, 1]
        ax.plot(times, true_trans[:, i], linewidth=1.2, label="Ground truth")
        ax.plot(times, pred_trans[:, i], linewidth=1.2, linestyle="--", label="Model")
        ax.set_ylabel(f"{trans_labels[i]} (m/s)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
            ax.set_title("Translational Velocity")

    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 1].set_xlabel("Time (s)")
    fig.suptitle(f"{title} — Velocity", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"  Saved rate figure -> {output_path}")


def _save_integrated_figure(true_rot, pred_rot, true_trans, pred_trans,
                            model_dt, title, output_path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    true_rot, pred_rot = np.asarray(true_rot), np.asarray(pred_rot)
    true_trans, pred_trans = np.asarray(true_trans), np.asarray(pred_trans)

    true_rot_int = nm.integrate_angular_velocity(
        torch.tensor(true_rot, dtype=torch.float64), model_dt).numpy()
    pred_rot_int = nm.integrate_angular_velocity(
        torch.tensor(pred_rot, dtype=torch.float64), model_dt).numpy()
    true_pos = nm.integrate_translational_velocity(
        torch.tensor(true_trans, dtype=torch.float64),
        torch.tensor(true_rot, dtype=torch.float64), model_dt).numpy()
    pred_pos = nm.integrate_translational_velocity(
        torch.tensor(pred_trans, dtype=torch.float64),
        torch.tensor(pred_rot, dtype=torch.float64), model_dt).numpy()

    T = true_rot.shape[0]
    times = np.arange(T) * model_dt
    rot_labels = ["Roll (X)", "Pitch (Y)", "Yaw (Z)"]
    trans_labels = ["X", "Y", "Z"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharex=True)
    for i in range(3):
        ax = axes[i, 0]
        ax.plot(times, true_rot_int[:, i], linewidth=1.2, label="Ground truth")
        ax.plot(times, pred_rot_int[:, i], linewidth=1.2, linestyle="--", label="Model")
        ax.set_ylabel(f"{rot_labels[i]} (deg)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
            ax.set_title("Integrated Rotation")

        ax = axes[i, 1]
        ax.plot(times, true_pos[:, i], linewidth=1.2, label="Ground truth")
        ax.plot(times, pred_pos[:, i], linewidth=1.2, linestyle="--", label="Model")
        ax.set_ylabel(f"{trans_labels[i]} (m)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
            ax.set_title("Integrated Translation")

    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 1].set_xlabel("Time (s)")
    fig.suptitle(f"{title} — Integrated Pose", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"  Saved integrated figure -> {output_path}")


def run_drone_sweep(
    model_path: str | Path,
    num_clips: int = 3,
) -> dict:
    """Run drone sweep and return results.

    Returns:
        dict with keys 'rot_cosines', 'trans_cosines' (list[float]),
        'median_rot_cos', 'median_trans_cos' (float).
    """
    model_path = Path(model_path)
    model_folder = model_path.parent if model_path.is_file() else model_path
    local_config = du.load_config()

    print(f"Loading model from {model_path}...")
    model = nm.BaseRnn.load(model_path)
    model.eval()

    model_dt = model.dt
    device = model.device
    dtype = model.dtype

    connectome_folder = Path(local_config["data"]["connectome_folder"])
    drone_data_path = connectome_folder / "drone_data.pt"

    print("Loading drone clips...")
    drone_clips = su.get_drone_stimulus(model_dt, data_path=str(drone_data_path))
    num_clips = min(num_clips, len(drone_clips))
    print(f"Found {len(drone_clips)} clips, evaluating {num_clips}")

    output_dir = model_folder / "figures" / "drone_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    rot_cosines = []
    trans_cosines = []

    for ci in range(num_clips):
        clip = drone_clips[ci]
        clip_id = clip["clip_id"]
        N = clip["right"].shape[1]
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

        print(f"  Running model...")
        with torch.no_grad():
            pred_rot, pred_trans, _, _ = model.estimate_ego_motion_from_image(
                right_input, left_input, oob_columns=oob,
            )

        pred_rot_np = pred_rot[0].cpu().numpy()
        pred_trans_np = pred_trans[0].cpu().numpy()

        true_rot_rate = clip["rotation_rate"][:num_frames]
        true_trans_rate = clip["translation_rate"][:num_frames]

        # Scale predicted translation to match ground truth magnitude
        trans_scale = np.std(true_trans_rate) / (np.std(pred_trans_np) + 1e-12)
        pred_trans_scaled = pred_trans_np * trans_scale

        rot_cos = _cos_sim(true_rot_rate, pred_rot_np)
        trans_cos = _cos_sim(true_trans_rate, pred_trans_np)
        rot_cosines.append(rot_cos)
        trans_cosines.append(trans_cos)
        print(f"  Rotation  cosine similarity: {rot_cos:.4f}")
        print(f"  Translation cosine similarity: {trans_cos:.4f}")

        title = f"Drone clip {ci} (id={clip_id})"
        _save_rate_figure(
            true_rot_rate, pred_rot_np, true_trans_rate, pred_trans_scaled,
            model_dt, title, output_dir / f"drone_{ci}_rate.png",
        )
        _save_integrated_figure(
            true_rot_rate, pred_rot_np, true_trans_rate, pred_trans_scaled,
            model_dt, title, output_dir / f"drone_{ci}_integrated.png",
        )

    median_rot = float(np.median(rot_cosines))
    median_trans = float(np.median(trans_cosines))

    print(f"\n{'='*50}")
    print(f"Drone sweep summary ({num_clips} clips):")
    for i, (rc, tc) in enumerate(zip(rot_cosines, trans_cosines)):
        print(f"  Clip {i}: rot cos = {rc:.4f}  trans cos = {tc:.4f}")
    print(f"  Rotation  — median: {median_rot:.4f}  std: {np.std(rot_cosines):.4f}")
    print(f"  Translation — median: {median_trans:.4f}  std: {np.std(trans_cosines):.4f}")
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
    args = parser.parse_args()

    run_drone_sweep(args.model_folder, num_clips=args.num_clips)


if __name__ == "__main__":
    main()
