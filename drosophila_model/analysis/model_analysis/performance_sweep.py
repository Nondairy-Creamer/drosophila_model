"""
Quantitative performance sweep.

Reports the model's accuracy on three stimulus families with a single
comparable cosine-similarity number each:

  * sine waves   (``random_sine``)
  * random pano  (``random_pano``)
  * flyview      (``flyview``)

Two complementary metrics are computed where applicable:

  * flow cosine  — predicted optic flow vs. ground-truth optic flow
                   (the model's primary training objective). Available for
                   all three stimulus families.
  * ego cosine   — rotation (and translation) recovered by
                   ``estimate_ego_motion_from_image`` vs. ground truth.
                   Available for the two-eye families (random pano has no
                   translation, so only rotation is reported; flyview has
                   both).

No figures are produced — use ``pano_sweep`` / ``drone_sweep`` for those.

Usage:
    python analysis/model_analysis/performance_sweep.py models/my_model/
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from drosophila_model.config import load_config, get_data_config
from drosophila_model.model_definition import neural_models as nm
from drosophila_model.data_handling import stimulus_generators as su
from drosophila_model.training.loss_functions import cos_loss
from drosophila_model.analysis.model_analysis.analysis_utilities import (
    predict_optic_flow,
)

# (display name, stimulus_type) for each stimulus family.
FLOW_STIMULI = [
    ("sine_wave", "random_sine"),
    ("random_pano", "random_pano"),
    ("flyview", "flyview"),
]
# Two-eye families with ego-motion ground truth. ``has_translation`` flags
# whether the translation cosine is meaningful (random pano is pure rotation).
EGO_STIMULI = [
    ("random_pano", "random_pano", False),
    ("flyview", "flyview", True),
]


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Global cosine similarity over flattened, finite entries."""
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


def _eval_config(model_folder: Path, model: nm.BaseRnn) -> tuple[dict, dict]:
    """Load the model's config + data config, forcing device/dt to the model.

    Pre-stimulus is dropped (in the data config) so the metric reflects the
    driven response. Returns ``(config, data_config)``.
    """
    config = load_config(model_folder / "config.toml")
    config["model"]["device"] = str(model.device)
    config["model"]["dt"] = model.dt
    data_config = get_data_config()
    data_config["shared"]["pre_stim_time"] = 0.0
    return config, data_config


def run_performance_sweep(model_path: str | Path,
                          max_examples: int = 1) -> dict:
    """Run the performance sweep and return a metrics dict.

    Args:
        max_examples: number of stimulus examples to evaluate per family
            (default 1 — keep the benchmark short).

    Returns:
        dict with keys 'flow_cos' (name -> float) and 'ego_cos'
        (name -> {'rot': float, 'trans': float | None}).
    """
    model_path = Path(model_path)
    model_folder = model_path.parent if model_path.is_file() else model_path

    print(f"Loading model from {model_path}...")
    model = nm.BaseRnn.load(model_path)
    model.eval()

    config, data_config = _eval_config(model_folder, model)
    column_phi_theta = model.regions["optic_lobe"].column_phi_theta
    N = column_phi_theta.shape[0]

    flow_cos: dict[str, float] = {}
    for name, stim_type in FLOW_STIMULI:
        print(f"\n--- Flow cosine: {name} ({stim_type}) ---")
        data = su.get_stimulus(
            stim_type, config, column_phi_theta, num_eyes=1, data_config=data_config,
        )
        stimulus, optic_flow = data.stimulus, data.optic_flow
        stimulus = stimulus[:max_examples]
        optic_flow = optic_flow[:max_examples]
        print(f"  {stimulus.shape[0]} examples, {stimulus.shape[1]} frames")
        pred_flow = predict_optic_flow(model, stimulus)
        cos = 1.0 - float(cos_loss(optic_flow, pred_flow))
        flow_cos[name] = cos
        print(f"  flow cosine similarity: {cos:.4f}")

    ego_cos: dict[str, dict] = {}
    for name, stim_type, has_trans in EGO_STIMULI:
        print(f"\n--- Ego-motion cosine: {name} ({stim_type}) ---")
        data = su.get_stimulus(
            stim_type, config, column_phi_theta, num_eyes=2, data_config=data_config,
        )
        stimulus, true_rot, true_trans = data.stimulus, data.rotation, data.translation
        stimulus = stimulus[:max_examples]
        true_rot = true_rot[:max_examples]
        true_trans = true_trans[:max_examples]
        right = stimulus[:, :, :N]
        left = stimulus[:, :, N:]
        print(f"  {stimulus.shape[0]} examples, {stimulus.shape[1]} frames")
        with torch.no_grad():
            pred_rot, pred_trans, _, _ = model.regions[
                "optic_lobe"
            ].estimate_ego_motion_from_image(right, left)

        true_rot_np = true_rot.cpu().numpy()
        pred_rot_np = pred_rot.cpu().numpy()
        rot = float(np.median([
            _cos_sim(true_rot_np[i], pred_rot_np[i])
            for i in range(true_rot_np.shape[0])
        ]))
        print(f"  rotation cosine similarity: {rot:.4f}")

        trans = None
        if has_trans:
            true_trans_np = true_trans.cpu().numpy()
            pred_trans_np = pred_trans.cpu().numpy()
            trans = float(np.median([
                _cos_sim(true_trans_np[i], pred_trans_np[i])
                for i in range(true_trans_np.shape[0])
            ]))
            print(f"  translation cosine similarity: {trans:.4f}")

        ego_cos[name] = {"rot": rot, "trans": trans}

    _print_summary(flow_cos, ego_cos)
    return {"flow_cos": flow_cos, "ego_cos": ego_cos}


def _print_summary(flow_cos: dict, ego_cos: dict) -> None:
    print(f"\n{'='*52}")
    print("  Performance Summary (cosine similarity)")
    print(f"{'='*52}")
    print(f"  {'stimulus':<14}{'flow':>8}{'ego rot':>10}{'ego trans':>12}")
    names = list(dict.fromkeys(list(flow_cos) + list(ego_cos)))
    for name in names:
        flow = f"{flow_cos[name]:.4f}" if name in flow_cos else "—"
        ego = ego_cos.get(name, {})
        rot = f"{ego['rot']:.4f}" if ego.get("rot") is not None else "—"
        trans = f"{ego['trans']:.4f}" if ego.get("trans") is not None else "—"
        print(f"  {name:<14}{flow:>8}{rot:>10}{trans:>12}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report model performance (cosine similarity) on "
                    "sine waves, random pano, and flyview."
    )
    parser.add_argument("model_folder", type=str,
                        help="Path to model folder or model.pt")
    parser.add_argument("--max-examples", type=int, default=1,
                        help="Examples to evaluate per stimulus family (default: 1)")
    args = parser.parse_args()
    run_performance_sweep(args.model_folder, max_examples=args.max_examples)


if __name__ == "__main__":
    main()
