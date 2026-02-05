"""
Plot XT activity from an initial model checkpoint.

Creates a 1x4 panel plot for a single stimulus:
  - stimulus input
  - L2 firing rate
  - Tm1 firing rate
  - T5 firing rate (mean over T5a-d)
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

import torch
import neural_models as nm


def _resolve_model_folder(model_path: Path) -> Path:
    return model_path if model_path.is_dir() else model_path.parent


def _prepare_xt(activity: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, list[float]]:
    valid_cols = np.isfinite(activity).all(axis=0)
    if not np.any(valid_cols):
        raise ValueError("No valid columns found for XT plot.")
    activity_valid = activity[:, valid_cols]
    phi_valid = phi[valid_cols]
    sort_inds = np.argsort(phi_valid)
    activity_sorted = activity_valid[:, sort_inds]
    phi_sorted = phi_valid[sort_inds]
    extent = [float(phi_sorted.min()), float(phi_sorted.max()), 0, activity_sorted.shape[0]]
    return activity_sorted, extent


def _get_available_types(model: nm.NeuralModel, requested: list[str]) -> list[str]:
    return [ct for ct in requested if ct in model.unique_cell_types]


def _extract_activity(outputs: dict, cell_type: str) -> np.ndarray:
    activity = outputs["firing_rate"][cell_type][0]
    return activity.detach().cpu().numpy()


def _extract_t5_activity(outputs: dict, t5_types: list[str]) -> np.ndarray:
    t5_arrays = [outputs["firing_rate"][ct][0].detach().cpu().numpy() for ct in t5_types]
    stacked = np.stack(t5_arrays, axis=0)
    return np.nanmean(stacked, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot XT panels from a model checkpoint.")
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to model folder (or any file inside it).",
    )
    parser.add_argument(
        "--stim-index",
        type=int,
        default=1,
        help="Stimulus index to plot (default: 1).",
    )
    parser.add_argument(
        "--use-init",
        action="store_true",
        help="Load model_init.pt instead of model.pt.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional output file path (default: <model_folder>/init_xt_stim_<idx>.png)",
    )
    args = parser.parse_args()

    model_folder = _resolve_model_folder(Path(args.model_path))
    model_init_path = model_folder / "model_init.pt"
    model_path = model_init_path if args.use_init else model_folder / "model.pt"
    stimulus_path = model_folder / "cell_inputs.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing {model_path}")
    if not stimulus_path.exists():
        raise FileNotFoundError(f"Missing {stimulus_path}")

    model = nm.NeuralModel.load(model_path, device="cpu")
    stimulus = torch.load(stimulus_path, map_location="cpu")
    if stimulus.dim() != 3:
        raise ValueError("Expected stimulus with shape (stimuli, time, columns).")
    if args.stim_index < 0 or args.stim_index >= stimulus.shape[0]:
        raise ValueError(f"stim-index {args.stim_index} out of range for {stimulus.shape[0]} stimuli.")

    stim = stimulus[args.stim_index : args.stim_index + 1]

    input_neurons = _get_available_types(model, ["L1", "L2", "L3"])
    if not input_neurons:
        raise ValueError("No input neurons (L1/L2/L3) available in model.")

    output_neurons = _get_available_types(model, ["L2", "Tm1", "T5a", "T5b", "T5c", "T5d"])
    if not output_neurons:
        raise ValueError("Requested output neurons not available in model.")

    outputs, _, _ = model.predict(stim, input_neurons=input_neurons, output_neurons=output_neurons, init_state=None)

    phi = model.column_phi_theta[:, 0]
    stim_xt, stim_extent = _prepare_xt(stim[0].detach().cpu().numpy(), phi)

    if "L2" not in outputs["firing_rate"]:
        raise ValueError("L2 output not found in model predictions.")
    if "Tm1" not in outputs["firing_rate"]:
        raise ValueError("Tm1 output not found in model predictions.")

    l2_xt, l2_extent = _prepare_xt(_extract_activity(outputs, "L2"), phi)
    tm1_xt, tm1_extent = _prepare_xt(_extract_activity(outputs, "Tm1"), phi)

    t5_types = [ct for ct in ["T5a", "T5b", "T5c", "T5d"] if ct in outputs["firing_rate"]]
    if not t5_types:
        raise ValueError("No T5 outputs found (T5a-d).")
    t5_xt, t5_extent = _prepare_xt(_extract_t5_activity(outputs, t5_types), phi)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=150)

    panels = [
        ("stimulus", stim_xt, stim_extent),
        ("L2 firing rate", l2_xt, l2_extent),
        ("Tm1 firing rate", tm1_xt, tm1_extent),
        ("T5 firing rate (mean)", t5_xt, t5_extent),
    ]

    for ax, (title, data, extent) in zip(axes, panels):
        im = ax.imshow(data, interpolation="nearest", aspect="auto", extent=extent)
        ax.set_title(title)
        ax.set_xlabel("phi")
        ax.set_ylabel("time")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    if args.output_path is not None:
        output_path = Path(args.output_path)
    else:
        output_path = model_folder / f"init_xt_stim_{args.stim_index}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved XT panel plot to {output_path}")


if __name__ == "__main__":
    main()
