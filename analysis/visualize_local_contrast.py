"""
Visualize panoramic stimulus processing: raw image, sampled image, local contrast.

Loads a panoramic image, samples it at ommatidium locations, and applies
the model's local contrast normalization (spatial pre-filter + contrast norm,
excluding the temporal LRU).

Produces a 3x1 figure:
  Row 0: Raw panoramic image cropped to the eye's field of view
  Row 1: Sampled image at ommatidium column locations
  Row 2: Local contrast after spatial normalization

Usage:
    python analysis/visualize_local_contrast.py models/my_model/
    python analysis/visualize_local_contrast.py models/my_model/ --stim-index 2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from scipy.ndimage import map_coordinates

import data_utilities as du
import neural_models as nm
import stimulus_utilities as su
from analysis_utilities import resolve_model_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize panoramic stimulus: raw, sampled, and local contrast."
    )
    parser.add_argument("model_folder", type=str, help="Path to model folder or model.pt")
    parser.add_argument("--stim-index", type=int, default=0,
                        help="Index of panoramic image to use (default: 0)")
    args = parser.parse_args()

    model_file, model_folder = resolve_model_paths(Path(args.model_folder))
    local_config = du.load_config()
    device = local_config["model"].get("device", "cpu")

    print(f"Loading model from {model_file} on {device}...")
    model = nm.NeuralModel.load(model_file, device=device)
    model.eval()

    # --- Load panoramic images and select one ---
    raw_panos = su._load_all_panos()
    num_panos = raw_panos.shape[0]
    stim_idx = args.stim_index % num_panos
    print(f"Using panorama {stim_idx} of {num_panos}")

    pano_filtered, pano_y_lim, pano_x_lim = su._filter_pano(raw_panos[stim_idx])
    pano_height, pano_width = pano_filtered.shape

    # --- Column locations ---
    column_phi = model.column_phi_theta[:, 0]
    column_theta = model.column_phi_theta[:, 1]
    column_xyz = model.get_column_xyz()
    cell_x, cell_z = column_xyz[:, 0], column_xyz[:, 2]

    # --- Sample panorama at column locations (no rotation) ---
    y_indices = ((column_theta - pano_y_lim[0])
                 / (pano_y_lim[1] - pano_y_lim[0]) * (pano_height - 1))
    x_indices = ((pano_x_lim[0] - column_phi)
                 / (pano_x_lim[0] - pano_x_lim[1]) * pano_width)
    x_indices = x_indices % pano_width

    sampled = map_coordinates(
        pano_filtered, [y_indices, x_indices], order=1, mode="nearest"
    )

    # --- Apply local contrast normalization (without temporal LRU) ---
    if not model.use_input_conv:
        print("Warning: model has no input conv; showing raw sampled values for contrast panel.")
        local_contrast = sampled
    else:
        sampled_t = torch.tensor(sampled, device=device, dtype=model.dtype).unsqueeze(0)
        with torch.no_grad():
            filtered = model.input_pre_conv(sampled_t)
            local_mean = model.input_spatial_conv(filtered)
            centered = filtered - local_mean
            local_var = model.input_spatial_conv(centered ** 2)
            local_contrast_t = -centered / torch.sqrt(local_var + 1e-12)
        local_contrast = local_contrast_t[0].cpu().numpy()

    # --- Crop raw panorama to the eye's field of view ---
    raw_pano = raw_panos[stim_idx]
    raw_height, raw_width = raw_pano.shape
    margin_rad = 5 * np.pi / 180
    phi_lo = column_phi.min() - margin_rad
    phi_hi = column_phi.max() + margin_rad
    theta_lo = column_theta.min() - margin_rad
    theta_hi = column_theta.max() + margin_rad

    theta_axis = np.linspace(pano_y_lim[0], pano_y_lim[1], raw_height)
    phi_axis = np.linspace(pano_x_lim[0], pano_x_lim[1], raw_width)

    row_mask = (theta_axis >= theta_lo) & (theta_axis <= theta_hi)
    col_mask = (phi_axis >= phi_lo) & (phi_axis <= phi_hi)
    row_idx = np.where(row_mask)[0]
    col_idx = np.where(col_mask)[0]

    if len(row_idx) == 0 or len(col_idx) == 0:
        pano_cropped = raw_pano
        elev_top = (90 - pano_y_lim[0] * 180 / np.pi)
        elev_bot = (90 - pano_y_lim[1] * 180 / np.pi)
        phi_left = phi_axis[0] * 180 / np.pi
        phi_right = phi_axis[-1] * 180 / np.pi
    else:
        r0, r1 = row_idx[0], row_idx[-1] + 1
        c0, c1 = col_idx[0], col_idx[-1] + 1
        pano_cropped = raw_pano[r0:r1, c0:c1]
        elev_top = 90 - theta_axis[r0] * 180 / np.pi
        elev_bot = 90 - theta_axis[r1 - 1] * 180 / np.pi
        phi_left = phi_axis[c0] * 180 / np.pi
        phi_right = phi_axis[c1 - 1] * 180 / np.pi

    # phi_axis goes from pi to -pi (decreasing left-to-right), matching
    # the convention: positive phi (forward) on the left, negative phi
    # (lateral/right) on the right.
    extent = [phi_left, phi_right, elev_bot, elev_top]

    # --- Plot 3×1 figure ---
    fig, axes = plt.subplots(3, 1, figsize=(8, 14))

    # Row 0: Raw panoramic image (cropped to eye FOV)
    axes[0].imshow(pano_cropped, cmap="gray", aspect="auto", extent=extent)
    axes[0].set_xlabel("Azimuth φ (deg)")
    axes[0].set_ylabel("Elevation (deg)")
    axes[0].set_title(f"Raw Panoramic Image (index {stim_idx})")

    # Row 1: Sampled image at ommatidium locations
    x_margin = 0.1 * (cell_x.max() - cell_x.min())
    z_margin = 0.1 * (cell_z.max() - cell_z.min())

    sc_samp = axes[1].scatter(
        cell_x, cell_z, c=sampled, cmap="gray", s=90, edgecolors="none",
    )
    axes[1].set_xlim(cell_x.max() + x_margin, cell_x.min() - x_margin)
    axes[1].set_ylim(cell_z.min() - z_margin, cell_z.max() + z_margin)
    axes[1].set_aspect("equal")
    axes[1].set_title("Sampled Image (Ommatidia)")
    fig.colorbar(sc_samp, ax=axes[1], label="intensity")

    # Row 2: Local contrast
    sc_lc = axes[2].scatter(
        cell_x, cell_z, c=local_contrast, cmap="gray", s=90, edgecolors="none",
    )
    axes[2].set_xlim(cell_x.max() + x_margin, cell_x.min() - x_margin)
    axes[2].set_ylim(cell_z.min() - z_margin, cell_z.max() + z_margin)
    axes[2].set_aspect("equal")
    axes[2].set_title("Local Contrast")
    fig.colorbar(sc_lc, ax=axes[2], label="intensity")

    plt.tight_layout()

    out_path = Path("figures") / f"local_contrast_{stim_idx}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
