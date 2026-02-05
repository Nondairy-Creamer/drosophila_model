"""
Compare input convolution filters for synthetic runs.

Generates a 2x3 plot:
  - Top row: spatial input_conv_xy (init, trained, true)
  - Bottom row: temporal input_conv_t (init, trained, true)
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
import data_utilities as du
import neural_models as nm


def _resolve_model_folder(model_path: Path) -> Path:
    if model_path.is_dir():
        return model_path
    return model_path.parent


def _hex_grid_points(axial_coords: np.ndarray) -> np.ndarray:
    """Convert axial (q, r) hex coords to flat-top xy points."""
    q = axial_coords[:, 0].astype(np.float64)
    r = axial_coords[:, 1].astype(np.float64)
    grid_x = 1.5 * q
    grid_y = np.sqrt(3.0) * (r + 0.5 * q)
    return np.stack([grid_x, grid_y], axis=1)


def _centered_norm(values: np.ndarray) -> tuple[float, float]:
    max_abs = float(np.max(np.abs(values)))
    if max_abs == 0:
        max_abs = 1.0
    return -max_abs, max_abs


def _load_conv_params(model_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    model = nm.NeuralModel.load(model_path, device="cpu")
    if not model.use_input_conv:
        raise ValueError(f"Model at {model_path} does not have input convolution enabled.")
    conv_xy = model.input_conv_xy.detach().cpu().numpy()
    conv_t = model.input_conv_t.detach().cpu().numpy()
    axial_coords = model.input_conv_axial_coords.detach().cpu().numpy()
    conv_size_x = model.input_conv_size_x
    return conv_xy, conv_t, axial_coords, conv_size_x


def _get_io_neurons(config: dict) -> tuple[list[str], list[str]]:
    input_neurons = ["L1", "L2", "L3"]
    output_neurons = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]
    subset = config["model"].get("use_connectome_subset", [])
    if len(subset) > 0:
        input_neurons = [n for n in input_neurons if n in subset]
        output_neurons = [n for n in output_neurons if n in subset]
    return input_neurons, output_neurons


def _predict_decoder(model, stimulus, input_neurons, output_neurons) -> np.ndarray:
    with torch.no_grad():
        pred, _, _ = model.predict(
            stimulus,
            input_neurons=input_neurons,
            output_neurons=output_neurons,
            init_state=None,
        )
    if "decoder" not in pred:
        raise ValueError("Decoder output not found. Ensure model.use_decoder is True.")
    return pred["decoder"].detach().cpu().numpy()


def _prepare_xt(activity: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, list[float]]:
    valid_cells = ~np.any(np.isnan(activity), axis=0)
    valid_activity = activity[:, valid_cells]
    valid_phi = phi[valid_cells]
    sort_inds = np.argsort(valid_phi)
    phi_sorted = valid_phi[sort_inds]
    activity_sorted = valid_activity[:, sort_inds]
    extent = [phi_sorted.min(), phi_sorted.max(), 0, activity_sorted.shape[0]]
    return activity_sorted, extent


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze synthetic fit input convolution filters.")
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to model folder (containing model.pt, model_init.pt, target_model.pt)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional output file path (default: <model_folder>/synthetic_fit_filters.png)",
    )
    args = parser.parse_args()

    model_folder = _resolve_model_folder(Path(args.model_path))
    model_pt = model_folder / "model.pt"
    model_init_pt = model_folder / "model_init.pt"
    target_model_pt = model_folder / "target_model.pt"
    config_path = model_folder / "config.toml"
    stimulus_path = model_folder / "cell_inputs.pt"

    if not model_pt.exists():
        raise FileNotFoundError(f"Missing {model_pt}")
    if not model_init_pt.exists():
        raise FileNotFoundError(f"Missing {model_init_pt}")
    if not target_model_pt.exists():
        raise FileNotFoundError(f"Missing {target_model_pt}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}")
    if not stimulus_path.exists():
        raise FileNotFoundError(f"Missing {stimulus_path}")

    conv_init_xy, conv_init_t, axial_coords, conv_size_x = _load_conv_params(model_init_pt)
    conv_trained_xy, conv_trained_t, axial_coords_trained, conv_size_x_trained = _load_conv_params(model_pt)
    conv_true_xy, conv_true_t, axial_coords_true, conv_size_x_true = _load_conv_params(target_model_pt)

    if conv_size_x != conv_size_x_trained or conv_size_x != conv_size_x_true:
        raise ValueError("Input conv spatial sizes do not match across models.")
    if not np.array_equal(axial_coords, axial_coords_trained) or not np.array_equal(axial_coords, axial_coords_true):
        raise ValueError("Input conv axial coordinates do not match across models.")

    radius = conv_size_x // 2
    q_idx = axial_coords[:, 0] + radius
    r_idx = axial_coords[:, 1] + radius
    grid_points = _hex_grid_points(axial_coords)

    spatial_init = conv_init_xy[r_idx, q_idx]
    spatial_trained = conv_trained_xy[r_idx, q_idx]
    spatial_true = conv_true_xy[r_idx, q_idx]
    spatial_stack = np.concatenate([spatial_init, spatial_trained, spatial_true], axis=0)
    vmin_xy, vmax_xy = _centered_norm(spatial_stack)

    temporal_stack = np.concatenate([conv_init_t, conv_trained_t, conv_true_t], axis=0)
    vmin_t, vmax_t = _centered_norm(temporal_stack)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=150)
    titles = ["Initial", "Trained", "True"]
    spatial_weights = [spatial_init, spatial_trained, spatial_true]
    temporal_weights = [conv_init_t, conv_trained_t, conv_true_t]

    scatter = None
    for col in range(3):
        ax = axes[0, col]
        scatter = ax.scatter(
            grid_points[:, 0],
            grid_points[:, 1],
            c=spatial_weights[col],
            cmap="coolwarm",
            vmin=vmin_xy,
            vmax=vmax_xy,
            s=760,
            linewidths=0.6,
            edgecolors="black",
        )
        ax.set_title(titles[col])
        ax.set_xlabel("hex x")
        if col == 0:
            ax.set_ylabel("hex y\n(spatial kernel)")
        else:
            ax.set_ylabel("hex y")
        ax.set_aspect("equal", adjustable="box")
        x_min, x_max = float(grid_points[:, 0].min()), float(grid_points[:, 0].max())
        y_min, y_max = float(grid_points[:, 1].min()), float(grid_points[:, 1].max())
        pad = 0.15 * max(x_max - x_min, y_max - y_min, 1.0)
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)

    for col in range(3):
        ax = axes[1, col]
        ax.plot(temporal_weights[col], marker="o")
        ax.set_xlabel("t index")
        if col == 0:
            ax.set_ylabel("weight\n(temporal kernel)")
        else:
            ax.set_ylabel("weight")
        ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_ylim(vmin_t, vmax_t)

    fig.tight_layout()
    output_path = Path(args.output_path) if args.output_path else model_folder / "synthetic_fit_filters.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved synthetic fit filters to {output_path}")

    # --- XT activity comparison for stim_5 ---
    config = du.load_config(config_path)
    stimulus = torch.load(stimulus_path, map_location="cpu")
    if stimulus.shape[0] <= 5:
        raise ValueError("stim_5 requested but stimulus has fewer than 6 entries.")
    stim_5 = stimulus[5:6]

    input_neurons, output_neurons = _get_io_neurons(config)

    model_init = nm.NeuralModel.load(model_init_pt, device="cpu")
    model_trained = nm.NeuralModel.load(model_pt, device="cpu")
    model_true = nm.NeuralModel.load(target_model_pt, device="cpu")

    decoder_init = _predict_decoder(model_init, stim_5, input_neurons, output_neurons)[0]
    decoder_trained = _predict_decoder(model_trained, stim_5, input_neurons, output_neurons)[0]
    decoder_true = _predict_decoder(model_true, stim_5, input_neurons, output_neurons)[0]

    phi = model_trained.column_phi_theta[:, 0]

    init_sorted_x, extent = _prepare_xt(decoder_init[:, :, 0], phi)
    trained_sorted_x, _ = _prepare_xt(decoder_trained[:, :, 0], phi)
    true_sorted_x, _ = _prepare_xt(decoder_true[:, :, 0], phi)
    init_sorted_y, _ = _prepare_xt(decoder_init[:, :, 1], phi)
    trained_sorted_y, _ = _prepare_xt(decoder_trained[:, :, 1], phi)
    true_sorted_y, _ = _prepare_xt(decoder_true[:, :, 1], phi)

    xt_stack_x = np.concatenate([init_sorted_x.ravel(), trained_sorted_x.ravel(), true_sorted_x.ravel()])
    xt_stack_x = xt_stack_x[np.isfinite(xt_stack_x)]
    if xt_stack_x.size == 0:
        raise ValueError("XT decoder plots contain no finite values for output 0.")
    vmin_xt_x, vmax_xt_x = float(np.min(xt_stack_x)), float(np.max(xt_stack_x))

    xt_stack_y = np.concatenate([init_sorted_y.ravel(), trained_sorted_y.ravel(), true_sorted_y.ravel()])
    xt_stack_y = xt_stack_y[np.isfinite(xt_stack_y)]
    if xt_stack_y.size == 0:
        raise ValueError("XT decoder plots contain no finite values for output 1.")
    vmin_xt_y, vmax_xt_y = float(np.min(xt_stack_y)), float(np.max(xt_stack_y))

    fig_xt, axes_xt = plt.subplots(2, 3, figsize=(12, 7), dpi=150)
    xt_titles = ["Initial", "Trained", "True"]
    xt_row_x = [init_sorted_x, trained_sorted_x, true_sorted_x]
    xt_row_y = [init_sorted_y, trained_sorted_y, true_sorted_y]

    for col in range(3):
        ax = axes_xt[0, col]
        ax.imshow(
            xt_row_x[col],
            interpolation="nearest",
            aspect="auto",
            extent=extent,
            vmin=vmin_xt_x,
            vmax=vmax_xt_x,
        )
        ax.set_title(xt_titles[col])
        ax.set_xlabel("phi")
        ax.set_ylabel("time")

    for col in range(3):
        ax = axes_xt[1, col]
        ax.imshow(
            xt_row_y[col],
            interpolation="nearest",
            aspect="auto",
            extent=extent,
            vmin=vmin_xt_y,
            vmax=vmax_xt_y,
        )
        ax.set_xlabel("phi")
        ax.set_ylabel("time")

    fig_xt.tight_layout()
    xt_output_path = model_folder / "synthetic_fit_xt_stim_5.png"
    fig_xt.savefig(xt_output_path)
    plt.close(fig_xt)
    print(f"Saved synthetic fit XT plot to {xt_output_path}")


if __name__ == "__main__":
    main()
