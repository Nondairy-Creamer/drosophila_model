"""
Quantitative performance sweep.

Reports the model's accuracy on three stimulus families with a single
comparable cosine-similarity number each:

  * sine waves   (``random_sine``)
  * random pano  (``random_pano``)
  * flyview      (``flyview``)

Each family is read from its baked dataset
(``<data_folder>/stimulus/training/<type>_<split>.pt``, made by
``create_training_data``). By default the held-out ``test`` split is used;
``--split train`` evaluates on training examples instead. Families whose
baked file is missing (or was baked at a different dt than the model) are
skipped. The model is run exactly as in training (two-eye data through the
two-eye ego-motion path, solved with the dataset's camera offset), and only
the time steps and columns that training scores (the dataset's
``train_mask``, after any pre-stimulus period) enter the metrics.

Two complementary metrics are computed where applicable:

  * flow cosine  — predicted optic flow vs. ground-truth optic flow
                   (the model's primary training objective). Available for
                   all three stimulus families.
  * ego cosine   — rotation (and translation) recovered by the two-eye
                   ego-motion solver vs. ground truth. Available for the
                   two-eye families with odometry (random pano has no
                   translation, so only rotation is reported; flyview has
                   both).

No figures are produced — use ``pano_sweep`` for those.

Usage:
    python -m drosophila_model.analysis.model_analysis.performance_sweep models/<model>/
    python -m drosophila_model.analysis.model_analysis.performance_sweep models/<model>/ --max-examples 10 --split train
"""
import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from drosophila_model.config import load_run_config
from drosophila_model.model_definition import neural_models as nm
from drosophila_model.training import training_utilities as tu
from drosophila_model.training.loss_functions import cos_loss, _mask_pred_target
from drosophila_model.analysis.model_analysis.analysis_utilities import cos_sim, get_flow_key

# (display name, stimulus_type) for each stimulus family.
FLOW_STIMULI = [
    ("sine_wave", "random_sine"),
    ("random_pano", "random_pano"),
    ("flyview", "flyview"),
]
# Families with ego-motion ground truth. ``has_translation`` flags whether the
# translation cosine is meaningful (random pano is pure rotation).
EGO_STIMULI = {
    "random_pano": False,
    "flyview": True,
}


def _load_family(run_config: dict, model: nm.BaseRnn, stim_type: str,
                 split: str, max_examples: int) -> dict | None:
    """Baked ``split`` of ``stim_type`` (first ``max_examples``), or None if unavailable."""
    config = copy.deepcopy(run_config)
    config["stimulus"]["stimulus_type"] = stim_type
    config["model"]["dt"] = model.dt
    try:
        # Load on the CPU and move only the evaluated examples to the model's device.
        data = tu.load_processed_stimulus(
            config, device=torch.device("cpu"), dtype=model.dtype, split=split)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  Skipping {stim_type}: {exc}")
        return None
    for key in ("stimulus", "optic_flow", "train_mask", "rotation", "translation"):
        if data[key] is not None:
            data[key] = data[key][:max_examples].to(model.device)
    return data


def run_performance_sweep(model_path: str | Path,
                          max_examples: int = 1,
                          split: str = "test") -> dict:
    """Run the performance sweep and return a metrics dict.

    Args:
        max_examples: number of stimulus examples to evaluate per family
            (default 1 — keep the benchmark short).
        split: baked split to evaluate on, ``'test'`` (held out, default) or
            ``'train'``.

    Returns:
        dict with keys 'flow_cos' (name -> float) and 'ego_cos'
        (name -> {'rot': float, 'trans': float | None}). Skipped families are
        absent.
    """
    model_path = Path(model_path)
    model_folder = model_path.parent if model_path.is_file() else model_path

    print(f"Loading model from {model_path}...")
    model = nm.BaseRnn.load(model_path)
    model.eval()
    run_config = load_run_config(model_folder)
    flow_key = get_flow_key(model)
    if flow_key is None:
        raise ValueError("Model has no flow readout configured. Cannot evaluate optic flow.")

    flow_cos: dict[str, float] = {}
    ego_cos: dict[str, dict] = {}
    for name, stim_type in FLOW_STIMULI:
        print(f"\n--- {name} ({stim_type}, {split} split) ---")
        data = _load_family(run_config, model, stim_type, split, max_examples)
        if data is None:
            continue
        stimulus, train_mask = data["stimulus"], data["train_mask"]
        print(f"  {stimulus.shape[0]} examples, {stimulus.shape[1]} frames, "
              f"{data['num_eyes']} eye(s)")
        start = tu.loss_start_step(train_mask)
        with torch.no_grad():
            outputs, _ = tu.model_outputs(
                model, stimulus, train_mask, data["num_eyes"],
                eye_offset=data["eye_offset"])

        pred_flow, true_flow = _mask_pred_target(
            outputs[flow_key], data["optic_flow"], train_mask, start)
        cos = 1.0 - float(cos_loss(true_flow, pred_flow))
        flow_cos[name] = cos
        print(f"  flow cosine similarity: {cos:.4f}")

        if name not in EGO_STIMULI or data["rotation"] is None:
            continue
        true_rot_np = data["rotation"][:, start:].cpu().numpy()
        pred_rot_np = outputs["pred_rotation"][:, start:].cpu().numpy()
        rot = float(np.median([
            cos_sim(true_rot_np[i], pred_rot_np[i])
            for i in range(true_rot_np.shape[0])
        ]))
        print(f"  rotation cosine similarity: {rot:.4f}")

        trans = None
        if EGO_STIMULI[name]:
            true_trans_np = data["translation"][:, start:].cpu().numpy()
            pred_trans_np = outputs["pred_translation"][:, start:].cpu().numpy()
            trans = float(np.median([
                cos_sim(true_trans_np[i], pred_trans_np[i])
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
        flow = f"{flow_cos[name]:.4f}" if name in flow_cos else "-"
        ego = ego_cos.get(name, {})
        rot = f"{ego['rot']:.4f}" if ego.get("rot") is not None else "-"
        trans = f"{ego['trans']:.4f}" if ego.get("trans") is not None else "-"
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
    parser.add_argument("--split", choices=("test", "train"), default="test",
                        help="Baked split to evaluate on (default: test, held out)")
    args = parser.parse_args()
    run_performance_sweep(args.model_folder, max_examples=args.max_examples,
                          split=args.split)


if __name__ == "__main__":
    main()
