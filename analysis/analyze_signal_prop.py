"""
Plot XT activity for light and dark pathways using sine wave stimuli.

Generates rightward and leftward sine waves, runs them through the model,
and creates XT plots for:
  Light pathway: L1, Mi1, Tm3, Mi4, Mi9, T4a, T4b
  Dark pathway:  L2, Tm1, Tm2, Tm4, Tm9, T5a, T5b
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
import stimulus_utilities as su

from analysis_utilities import resolve_model_folder, prepare_xt

# Cell types for each pathway
LIGHT_PATHWAY = ["L1", "Mi1", "Tm3", "Mi4", "Mi9", "T4a", "T4b"]
DARK_PATHWAY = ["L2", "Tm1", "Tm2", "Tm4", "Tm9", "T5a", "T5b"]


def _get_available_types(model: nm.NeuralModel, requested: list[str]) -> list[str]:
    return [ct for ct in requested if ct in model.unique_cell_types]


def _extract_activity(outputs: dict, cell_type: str) -> np.ndarray:
    activity = outputs["firing_rate"][cell_type][0]
    return activity.detach().cpu().numpy()


def _generate_sine_stimulus(
    model: nm.NeuralModel,
    direction: str,
    temporal_frequency: float,
    stim_time: float,
    pre_stim_time: float,
) -> torch.Tensor:
    """Generate a rightward or leftward sine wave stimulus for the model columns."""
    phi = model.column_phi_theta[:, 0]
    theta = model.column_phi_theta[:, 1]

    tf = temporal_frequency if direction == "right" else -temporal_frequency
    sine_wave, _ = su.make_lr_sine_wave(
        phi, theta,
        contrast=1.0,
        wavelength=30,
        temporal_frequency=tf,
        phase_init=0,
        dt=model.dt,
        stim_time=stim_time,
        pre_stim_time=pre_stim_time,
        post_stim_time=0.0,
    )
    # Shape: (1, time, columns)
    return torch.tensor(sine_wave[None, :, :], device=model.device, dtype=model.dtype)


def _run_model(
    model: nm.NeuralModel,
    stimulus: torch.Tensor,
    pathway_types: list[str],
) -> dict:
    """Run the model and return outputs for the requested cell types."""
    input_neurons = _get_available_types(model, ["L1", "L2", "L3"])
    output_neurons = _get_available_types(model, pathway_types)
    if not input_neurons:
        raise ValueError("No input neurons (L1/L2/L3) available in model.")

    with torch.no_grad():
        outputs, _, _ = model.predict(
            stimulus,
            input_neurons=input_neurons,
            output_neurons=output_neurons,
            init_state=None,
        )
    return outputs


def _plot_pathway(
    fig,
    axes,
    stimulus_xt: np.ndarray,
    stim_extent: list[float],
    outputs: dict,
    cell_types: list[str],
    phi: np.ndarray,
    pathway_label: str,
) -> None:
    """Fill one row of axes with stimulus + cell-type XT panels."""
    # Stimulus panel
    ax = axes[0]
    im = ax.imshow(stimulus_xt, interpolation="nearest", aspect="auto", extent=stim_extent, cmap="gray")
    ax.set_title(f"Stimulus")
    ax.set_ylabel(f"{pathway_label}\ntime")
    ax.set_xlabel("φ (rad)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Cell-type panels
    for idx, ct in enumerate(cell_types):
        ax = axes[idx + 1]
        if ct in outputs["firing_rate"]:
            activity = _extract_activity(outputs, ct)
            xt, extent = prepare_xt(activity, phi)
            im = ax.imshow(xt, interpolation="nearest", aspect="auto", extent=extent)
            ax.set_title(ct)
            ax.set_xlabel("φ (rad)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.set_title(f"{ct} (N/A)")
            ax.text(0.5, 0.5, "not in model", transform=ax.transAxes, ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot XT signal propagation for light/dark pathways.")
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to model folder (or any file inside it).",
    )
    parser.add_argument(
        "--use-init",
        action="store_true",
        help="Load model_init.pt instead of model.pt.",
    )
    parser.add_argument(
        "--temporal-frequency",
        type=float,
        default=4.0,
        help="Temporal frequency of the sine wave in Hz (default: 4).",
    )
    parser.add_argument(
        "--stim-time",
        type=float,
        default=2.0,
        help="Stimulus duration in seconds (default: 2.0).",
    )
    parser.add_argument(
        "--pre-stim-time",
        type=float,
        default=0.5,
        help="Pre-stimulus gray period in seconds (default: 0.5).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: <model_folder>).",
    )
    args = parser.parse_args()

    model_folder = resolve_model_folder(Path(args.model_path))
    model_path = model_folder / ("model_init.pt" if args.use_init else "model.pt")

    if not model_path.exists():
        raise FileNotFoundError(f"Missing {model_path}")

    model = nm.NeuralModel.load(model_path, device="cpu")
    phi = model.column_phi_theta[:, 0]

    output_dir = Path(args.output_dir) if args.output_dir else model_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    all_cell_types = list(set(LIGHT_PATHWAY + DARK_PATHWAY))

    for direction in ["right", "left"]:
        label = "Rightward" if direction == "right" else "Leftward"
        print(f"Generating {label.lower()} sine wave (TF={args.temporal_frequency} Hz)...")

        stimulus = _generate_sine_stimulus(
            model, direction, args.temporal_frequency, args.stim_time, args.pre_stim_time,
        )

        print(f"  Running model ({stimulus.shape[1]} time steps, {stimulus.shape[2]} columns)...")
        outputs = _run_model(model, stimulus, all_cell_types)

        stim_np = stimulus[0].detach().cpu().numpy()
        stim_xt, stim_extent = prepare_xt(stim_np, phi)

        # --- Light pathway figure ---
        light_available = _get_available_types(model, LIGHT_PATHWAY)
        n_light = len(LIGHT_PATHWAY)
        fig_light, axes_light = plt.subplots(1, n_light + 1, figsize=(3 * (n_light + 1), 4), dpi=150)
        _plot_pathway(fig_light, axes_light, stim_xt, stim_extent, outputs, LIGHT_PATHWAY, phi, "Light")
        fig_light.suptitle(f"{label} Sine Wave — Light Pathway", fontsize=14, y=1.02)
        fig_light.tight_layout()
        light_path = output_dir / f"signal_prop_{direction}_light.png"
        fig_light.savefig(light_path, bbox_inches="tight")
        plt.close(fig_light)
        print(f"  Saved {light_path}")

        # --- Dark pathway figure ---
        dark_available = _get_available_types(model, DARK_PATHWAY)
        n_dark = len(DARK_PATHWAY)
        fig_dark, axes_dark = plt.subplots(1, n_dark + 1, figsize=(3 * (n_dark + 1), 4), dpi=150)
        _plot_pathway(fig_dark, axes_dark, stim_xt, stim_extent, outputs, DARK_PATHWAY, phi, "Dark")
        fig_dark.suptitle(f"{label} Sine Wave — Dark Pathway", fontsize=14, y=1.02)
        fig_dark.tight_layout()
        dark_path = output_dir / f"signal_prop_{direction}_dark.png"
        fig_dark.savefig(dark_path, bbox_inches="tight")
        plt.close(fig_dark)
        print(f"  Saved {dark_path}")

    print("Done!")


if __name__ == "__main__":
    main()
