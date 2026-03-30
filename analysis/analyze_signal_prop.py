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
import pandas as pd
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

import torch
import neural_models as nm
import stimulus_utilities as su
import data_utilities as du
import model_utilities as mu

from analysis_utilities import prepare_xt

# Cell types for each pathway
LIGHT_PATHWAY = ["L1", "Mi1", "Tm3", "Mi4", "Mi9", "T4a", "T4b"]
DARK_PATHWAY = ["L2", "Tm1", "Tm2", "Tm4", "Tm9", "T5a", "T5b"]


def _init_model_from_config(config_path: str = "config.toml", device: str | None = None) -> nm.NeuralModel:
    """Initialize a fresh model from config, mirroring run_model.py."""
    config = du.load_config(config_path)
    if device is not None:
        config["model"]["device"] = device
    device = config["model"]["device"]

    connectome_folder = Path(config["data"]["connectome_folder"])
    cell_info_df = pd.read_parquet(connectome_folder / "optic_lobes_cell_info.parquet")
    synapse_counts_df = pd.read_parquet(connectome_folder / "optic_lobes_synapse_counts.parquet")
    column_info_df = pd.read_parquet(connectome_folder / "optic_lobes_column_info.parquet")

    if len(config["model"]["use_connectome_subset"]) > 0:
        cell_info_df, synapse_counts_df = du.get_connectome_subset(
            cell_info_df, synapse_counts_df, config["model"]["use_connectome_subset"]
        )

    model_type = config["model"].get("model_type")
    ModelClass = getattr(nm, model_type)
    commit_hash = mu.get_git_commit_hash()

    model = ModelClass(
        synapse_counts_coo=synapse_counts_df[["pre_root_id", "post_root_id", "synapse_counts"]].to_numpy(),
        root_ids=cell_info_df["root_id"].to_numpy(),
        cell_types=cell_info_df["type"].to_numpy(),
        neurotransmitters=cell_info_df["neurotransmitter"].to_numpy(),
        column_indices=cell_info_df["column_index"].to_numpy(),
        column_phi_theta=column_info_df[["phi", "theta"]].to_numpy(),
        nonlinearity=config["model"]["nonlinearity"],
        count_threshold=config["model"]["synapse_count_threshold"],
        use_layer_norm=config["model"]["use_layer_norm"],
        device=device,
        dtype=config["model"]["dtype"],
        readouts=config["model"]["readouts"],
        use_input_conv=config["model"]["use_input_conv"],
        dt=config["model"]["dt"],
        seed=None,
        commit_hash=commit_hash,
    )
    return model


def _get_available_types(model: nm.NeuralModel, requested: list[str]) -> list[str]:
    return [ct for ct in requested if ct in model.unique_cell_types]


def _extract_activity(outputs: dict, cell_type: str, data_key: str = "firing_rate") -> np.ndarray:
    activity = outputs[data_key][cell_type][0]
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
        outputs = model.predict(
            stimulus,
            input_neurons=input_neurons,
            output_neurons=output_neurons,
            init_state=None,
        )['outputs']
    return outputs


def _print_stimulus_correlation(outputs: dict, stimulus_np: np.ndarray, cell_types: list[str]) -> None:
    """Compute and print Pearson correlation between each cell type's firing rate and the raw stimulus."""
    for ct in cell_types:
        if ct not in outputs["firing_rate"]:
            continue
        activity = outputs["firing_rate"][ct][0].detach().cpu().numpy()  # (time, columns)
        # Only use columns & timesteps where activity is valid (not NaN)
        valid = np.isfinite(activity).all(axis=0)
        if not np.any(valid):
            print(f"  {ct}: no valid columns")
            continue
        act_valid = activity[:, valid].ravel()
        stim_valid = stimulus_np[:, valid].ravel()
        corr = np.corrcoef(act_valid, stim_valid)[0, 1]
        print(f"  {ct} vs stimulus correlation: {corr:+.4f}")


def _plot_input_contributions(
    model: nm.NeuralModel,
    stimulus: torch.Tensor,
    target_type: str,
    phi: np.ndarray,
    direction: str,
    output_dir: Path,
    top_n: int = 5,
) -> None:
    """Plot XT contributions from each presynaptic cell type to *target_type*, ranked by RMS."""
    if target_type not in model.unique_cell_types:
        print(f"  {target_type} not in model, skipping input contributions.")
        return

    with torch.no_grad():
        # --- run simulation, storing all firing rates ------------------
        inputs = model.apply_input_conv(stimulus) if model.use_input_conv else stimulus
        input_neurons = [n for n in ["L1", "L2", "L3"] if n in model.unique_cell_types]
        neuron_idx, col_idx = model.get_shared_columns(input_neurons)
        stim_to_cell_mapping = {
            "neuron_index": torch.tensor(neuron_idx, device=model.device, dtype=torch.int64),
            "stimulus_index": torch.tensor(col_idx, device=model.device, dtype=torch.int64),
        }

        num_time = inputs.shape[1]
        v = model.steady_state.unsqueeze(0)

        # (time, num_neurons) – store firing rate after each step
        all_fr = torch.zeros(num_time, model.num_neurons, device=model.device, dtype=model.dtype)
        for t in range(num_time):
            v = model(v, inputs[:, t, :], stim_to_cell_mapping)
            if model.use_layer_norm:
                v = model.layer_norm(v)
            all_fr[t] = model.firing_rate(v)[0]

        # --- identify target neurons and their columns -----------------
        target_mask = model.cell_types == target_type
        target_indices = np.where(target_mask)[0]
        target_columns = model.column_indices[target_indices]
        valid = ~np.isnan(target_columns)
        target_indices = target_indices[valid]
        target_columns = target_columns[valid].astype(int)

        if len(target_indices) == 0:
            print(f"  No valid {target_type} neurons, skipping.")
            return

        # --- model gain parameters ------------------------------------
        target_type_idx = int(np.where(model.unique_cell_types == target_type)[0][0])
        in_str = model.input_strength_log.exp()[target_type_idx].detach()
        output_strength = model.output_strength_log.exp().detach()

        # --- extract synapses whose post neuron is in target ----------
        syn = model.synapse_counts.coalesce()
        syn_post = syn.indices()[0].cpu().numpy()
        syn_pre = syn.indices()[1].cpu().numpy()
        syn_val = syn.values()

        target_set = set(target_indices.tolist())
        entry_mask = np.array([p in target_set for p in syn_post])
        if not entry_mask.any():
            print(f"  No synapses onto {target_type}, skipping.")
            return

        rel_post = syn_post[entry_mask]
        rel_pre = syn_pre[entry_mask]
        rel_val = syn_val[torch.tensor(entry_mask, dtype=torch.bool)]

        # map global neuron index -> local target index (0 … |target|-1)
        post_local_map = {int(ni): li for li, ni in enumerate(target_indices)}
        rel_post_local = np.array([post_local_map[p] for p in rel_post])

        # --- group by presynaptic cell type ---------------------------
        pre_types_per_entry = model.cell_types[rel_pre]
        unique_pre_types = np.unique(pre_types_per_entry)

        contributions: dict[str, np.ndarray] = {}
        rms_values: dict[str, float] = {}

        for ct in unique_pre_types:
            ct_mask = pre_types_per_entry == ct
            ct_post_local = rel_post_local[ct_mask]
            ct_pre_global = rel_pre[ct_mask]
            ct_val = rel_val[torch.tensor(ct_mask, dtype=torch.bool)]

            ct_type_idx = int(np.where(model.unique_cell_types == ct)[0][0])
            out_str = output_strength[ct_type_idx]

            # build sparse submatrix (|target| × num_neurons) for this pre type
            sparse_idx = torch.tensor(
                np.stack([ct_post_local, ct_pre_global]),
                dtype=torch.int64,
                device=model.device,
            )
            ct_syn = torch.sparse_coo_tensor(
                sparse_idx,
                ct_val,
                size=(len(target_indices), model.num_neurons),
                device=model.device,
                dtype=model.dtype,
            ).coalesce()

            # contribution: (|target|, time) = sparse @ all_fr^T
            contrib = torch.sparse.mm(ct_syn, all_fr.T)  # (|target|, time)
            contrib = (contrib * in_str * out_str).T  # (time, |target|)

            # map to (time, num_columns), averaging if >1 target neuron per col
            contrib_np = contrib.cpu().numpy()
            col_sum = np.zeros((num_time, model.num_columns))
            col_cnt = np.zeros(model.num_columns)
            for i, col in enumerate(target_columns):
                col_sum[:, col] += contrib_np[:, i]
                col_cnt[col] += 1
            contrib_xt = np.full((num_time, model.num_columns), np.nan)
            has_data = col_cnt > 0
            contrib_xt[:, has_data] = col_sum[:, has_data] / col_cnt[has_data]

            contributions[ct] = contrib_xt
            vals = contrib_xt[np.isfinite(contrib_xt)]
            rms_values[ct] = float(np.sqrt(np.mean(vals ** 2))) if len(vals) > 0 else 0.0

    # --- rank and report -----------------------------------------------
    sorted_types = sorted(rms_values, key=rms_values.get, reverse=True)[:top_n]

    print(f"  Top {top_n} synaptic inputs to {target_type} by RMS ({direction}):")
    for ct in sorted_types:
        print(f"    {ct:>10} -> {target_type}: RMS = {rms_values[ct]:.6f}")

    # --- compute sum of top N contributions ------------------------------
    sum_contrib = np.zeros_like(contributions[sorted_types[0]])
    for ct in sorted_types:
        c = contributions[ct]
        # treat NaN as 0 for summing, but keep NaN where all inputs are NaN
        sum_contrib += np.where(np.isfinite(c), c, 0.0)
    # mark columns that had no data in any contributor as NaN
    any_valid = np.zeros(sum_contrib.shape[1], dtype=bool)
    for ct in sorted_types:
        any_valid |= np.isfinite(contributions[ct]).any(axis=0)
    sum_contrib[:, ~any_valid] = np.nan

    # --- plot individual contributions + sum ---------------------------
    n_panels = len(sorted_types) + 1  # +1 for the sum panel
    fig, axes = plt.subplots(1, n_panels, figsize=(3 * n_panels, 4), dpi=150)

    for idx, ct in enumerate(sorted_types):
        ax = axes[idx]
        xt, extent = prepare_xt(contributions[ct], phi)
        im = ax.imshow(xt, interpolation="nearest", aspect="auto", extent=extent)
        ax.set_title(f"{ct} -> {target_type}\nRMS={rms_values[ct]:.4f}")
        ax.set_xlabel("φ (rad)")
        if idx == 0:
            ax.set_ylabel("time")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Sum panel
    ax = axes[-1]
    xt, extent = prepare_xt(sum_contrib, phi)
    sum_vals = sum_contrib[np.isfinite(sum_contrib)]
    sum_rms = float(np.sqrt(np.mean(sum_vals ** 2))) if len(sum_vals) > 0 else 0.0
    im = ax.imshow(xt, interpolation="nearest", aspect="auto", extent=extent)
    ax.set_title(f"Sum (top {top_n})\nRMS={sum_rms:.4f}")
    ax.set_xlabel("φ (rad)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    label = direction.capitalize()
    fig.suptitle(f"Top {top_n} inputs to {target_type} — {label}", fontsize=14, y=1.02)
    fig.tight_layout()
    out_path = output_dir / f"input_contributions_{target_type}_{direction}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def _plot_pathway(
    fig,
    axes,
    stimulus_xt: np.ndarray,
    stim_extent: list[float],
    outputs: dict,
    cell_types: list[str],
    phi: np.ndarray,
    pathway_label: str,
    data_key: str = "firing_rate",
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
        if ct in outputs[data_key]:
            activity = _extract_activity(outputs, ct, data_key=data_key)
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
        "--model-folder",
        type=str,
        nargs="?",
        default=None,
        help="Path to model folder (or any file inside it). If omitted, a fresh model is initialized from config.",
    )
    parser.add_argument(
        "--use-init",
        action="store_true",
        help="Load model_init.pt instead of model.pt.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="config.toml",
        help="Path to config TOML file (used when no model_path is provided; default: config.toml).",
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
        help="Output directory (default: <model_folder> or 'figures/' when using config).",
    )
    args = parser.parse_args()

    if args.model_folder is not None:
        model_arg = Path(args.model_folder)
        model_folder = model_arg.parent if model_arg.is_file() else model_arg
        model_path = model_folder / ("model_init.pt" if args.use_init else "model.pt")

        if not model_path.exists():
            raise FileNotFoundError(f"Missing {model_path}")

        model = nm.NeuralModel.load(model_path)
        default_output_dir = model_folder
    else:
        print(f"No model path provided — initializing from config: {args.config_path}")
        model = _init_model_from_config(config_path=args.config_path)
        default_output_dir = Path("figures")

    print('sigma max:', model.calc_sigma_max())
    phi = model.column_phi_theta[:, 0]

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir
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

        # Print correlation between L1/L2 firing rate and raw stimulus
        print(f"  Stimulus vs firing-rate correlations ({direction}):")
        _print_stimulus_correlation(outputs, stim_np, ["L1", "L2"])

        stim_xt, stim_extent = prepare_xt(stim_np, phi)

        # --- Light pathway figure (firing rate) ---
        n_light = len(LIGHT_PATHWAY)
        fig_light, axes_light = plt.subplots(1, n_light + 1, figsize=(3 * (n_light + 1), 4), dpi=150)
        _plot_pathway(fig_light, axes_light, stim_xt, stim_extent, outputs, LIGHT_PATHWAY, phi, "Light")
        fig_light.suptitle(f"{label} Sine Wave — Light Pathway", fontsize=14, y=1.02)
        fig_light.tight_layout()
        light_path = output_dir / f"signal_prop_{direction}_light.png"
        fig_light.savefig(light_path, bbox_inches="tight")
        plt.close(fig_light)
        print(f"  Saved {light_path}")

        # --- Light pathway figure (voltage) ---
        fig_light_v, axes_light_v = plt.subplots(1, n_light + 1, figsize=(3 * (n_light + 1), 4), dpi=150)
        _plot_pathway(fig_light_v, axes_light_v, stim_xt, stim_extent, outputs, LIGHT_PATHWAY, phi, "Light", data_key="voltage")
        fig_light_v.suptitle(f"{label} Sine Wave — Light Pathway (voltage)", fontsize=14, y=1.02)
        fig_light_v.tight_layout()
        light_v_path = output_dir / f"signal_prop_{direction}_light_voltage.png"
        fig_light_v.savefig(light_v_path, bbox_inches="tight")
        plt.close(fig_light_v)
        print(f"  Saved {light_v_path}")

        # --- Dark pathway figure (firing rate) ---
        n_dark = len(DARK_PATHWAY)
        fig_dark, axes_dark = plt.subplots(1, n_dark + 1, figsize=(3 * (n_dark + 1), 4), dpi=150)
        _plot_pathway(fig_dark, axes_dark, stim_xt, stim_extent, outputs, DARK_PATHWAY, phi, "Dark")
        fig_dark.suptitle(f"{label} Sine Wave — Dark Pathway", fontsize=14, y=1.02)
        fig_dark.tight_layout()
        dark_path = output_dir / f"signal_prop_{direction}_dark.png"
        fig_dark.savefig(dark_path, bbox_inches="tight")
        plt.close(fig_dark)
        print(f"  Saved {dark_path}")

        # --- Dark pathway figure (voltage) ---
        fig_dark_v, axes_dark_v = plt.subplots(1, n_dark + 1, figsize=(3 * (n_dark + 1), 4), dpi=150)
        _plot_pathway(fig_dark_v, axes_dark_v, stim_xt, stim_extent, outputs, DARK_PATHWAY, phi, "Dark", data_key="voltage")
        fig_dark_v.suptitle(f"{label} Sine Wave — Dark Pathway (voltage)", fontsize=14, y=1.02)
        fig_dark_v.tight_layout()
        dark_v_path = output_dir / f"signal_prop_{direction}_dark_voltage.png"
        fig_dark_v.savefig(dark_v_path, bbox_inches="tight")
        plt.close(fig_dark_v)
        print(f"  Saved {dark_v_path}")

        # --- Input contribution breakdown for Tm1 ---
        _plot_input_contributions(model, stimulus, "Tm1", phi, direction, output_dir)

    print("Done!")


if __name__ == "__main__":
    main()
