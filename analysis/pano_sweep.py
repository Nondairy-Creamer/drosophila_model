"""
Panoramic odometry benchmark.

Generates random panoramic stimuli with smoothly varying rotation, runs
the model's ego-motion pipeline, and saves ground-truth vs predicted
odometry figures (rate and integrated pose) for each stimulus.

Usage:
    python analysis/pano_sweep.py models/my_model/
    python analysis/pano_sweep.py models/my_model/ --num-panos 10 --stim-time 10
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

import data_utilities as du
import neural_models as nm
import ego_motion as em
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

    true_rot_int = em.integrate_angular_velocity(
        torch.tensor(true_rot, dtype=torch.float64), model_dt).numpy()
    pred_rot_int = em.integrate_angular_velocity(
        torch.tensor(pred_rot, dtype=torch.float64), model_dt).numpy()
    true_pos = em.integrate_translational_velocity(
        torch.tensor(true_trans, dtype=torch.float64),
        torch.tensor(true_rot, dtype=torch.float64), model_dt).numpy()
    pred_pos = em.integrate_translational_velocity(
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


def run_pano_sweep(
    model_path: str | Path,
    num_panos: int = 5,
    stim_time: float = 5.0,
    pre_stim_time: float = 0.5,
    velocity_amp: float = 100.0,
    velocity_time_scale: float = 0.3,
) -> dict:
    """Run pano sweep and return results.

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
    column_phi_theta = model.column_phi_theta
    N = column_phi_theta.shape[0]

    both_phi = np.concatenate([column_phi_theta[:, 0], -column_phi_theta[:, 0]])
    both_theta = np.concatenate([column_phi_theta[:, 1], column_phi_theta[:, 1]])

    pre_stim_frames = int(round(pre_stim_time / model_dt))

    output_dir = model_folder / "figures" / "pano_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    rot_cosines = []

    for pi in range(num_panos):
        seed = pi
        print(f"\n--- Pano {pi} (seed={seed}) ---")

        stimulus_both, optic_flow_both, rotation_vectors, oob_mask = su.get_random_velocity_pano(
            both_phi, both_theta,
            velocity_amp=velocity_amp,
            velocity_time_scale=velocity_time_scale,
            num_repeats=1,
            dt=model_dt,
            stim_time=stim_time,
            seed=seed,
        )

        stim = stimulus_both[0]
        rot_vectors = rotation_vectors[0]
        oob = oob_mask[0]

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
            )

        pred_rot_np = pred_rot[0].cpu().numpy()
        pred_trans_np = pred_trans[0].cpu().numpy()

        rot_cos = _cos_sim(true_rot_rate[pre_stim_frames:], pred_rot_np[pre_stim_frames:])
        rot_cosines.append(rot_cos)
        print(f"  Rotation cosine similarity: {rot_cos:.4f}")

        title = f"Pano {pi} (seed={seed})"
        _save_rate_figure(
            true_rot_rate, pred_rot_np, true_trans_rate, pred_trans_np,
            model_dt, title, output_dir / f"pano_{pi}_rate.png",
        )
        _save_integrated_figure(
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
