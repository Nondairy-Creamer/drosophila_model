"""
Compute and plot the azimuthal spatial power spectrum of the natural panoramas.

Shows where the model's sine-wave test wavelengths (e.g. 30 deg, 60 deg) sit
relative to the energy distribution of natural scenes.

Usage:
    python view_spatial_frequency.py
    python view_spatial_frequency.py --max-panos 10
    python view_spatial_frequency.py --wavelengths 15 30 60 90
    python view_spatial_frequency.py --save figures/view_spatial_frequency/spatial_frequency.png
"""
import argparse
from pathlib import Path
import sys

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import torch

from drosophila_model.utilities import data_utilities as du


def _load_all_panos():
    """Load (N, H, W) panoramas from the consolidated .pt file."""
    pt_path = (
        Path(du.load_config()['data']['data_folder'])
        / 'stimulus' / 'all_panoramic.pt'
    )
    if not pt_path.exists():
        raise FileNotFoundError(
            f"Consolidated panoramic file not found at {pt_path.resolve()}. "
            "Please run the consolidation step first."
        )
    panos = torch.load(pt_path, weights_only=True)
    return panos.numpy().astype(np.float64)


def _filter_pano(pano, filter_std_rad=2.5 * np.pi / 180):
    """Gaussian low-pass + [0,1] normalization (mirrors stimulus_utilities._filter_pano)."""
    from scipy.ndimage import gaussian_filter

    pano_y_lim = -np.array([47, -58]) * np.pi / 180 + np.pi / 2
    pano_x_lim = np.array([np.pi, -np.pi])

    pano_height, pano_width = pano.shape
    y_pixels_per_rad = pano_height / (pano_y_lim[1] - pano_y_lim[0])
    x_pixels_per_rad = pano_width / (pano_x_lim[0] - pano_x_lim[1])
    sigma_y = filter_std_rad * y_pixels_per_rad
    sigma_x = filter_std_rad * x_pixels_per_rad

    pano_out = gaussian_filter(
        pano.astype(np.float64), sigma=[sigma_y, sigma_x], mode=['nearest', 'wrap']
    )
    pano_min, pano_max = pano_out.min(), pano_out.max()
    if pano_max > pano_min:
        pano_out = (pano_out - pano_min) / (pano_max - pano_min)
    else:
        pano_out = np.zeros_like(pano_out)
    return pano_out


def azimuthal_power_spectrum(pano):
    """1D power spectrum along the azimuth (columns) axis, averaged over rows.

    Args:
        pano: (H, W) image array.

    Returns:
        power: (W//2,) mean power at each positive-frequency bin (excluding DC).
    """
    fft = np.fft.rfft(pano, axis=-1)
    power = np.abs(fft) ** 2
    # average over rows, drop DC (bin 0)
    return power.mean(axis=0)[1:]


def main():
    parser = argparse.ArgumentParser(
        description="Azimuthal spatial power spectrum of natural panoramas."
    )
    parser.add_argument(
        "--pano-indices", type=int, nargs="+", default=None,
        help="Specific panorama indices to analyse (e.g. --pano-indices 0 5 12). "
             "Default: all panoramas.",
    )
    parser.add_argument(
        "--max-panos", type=int, default=None,
        help="Use only the first K panoramas (default: all). "
             "Ignored if --pano-indices is given.",
    )
    parser.add_argument(
        "--wavelengths", type=float, nargs="+", default=[30, 60],
        help="Wavelengths (degrees) to mark on the plot.",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save figure to this path instead of showing interactively "
             "(default: figures/view_spatial_frequency/spatial_frequency.png).",
    )
    args = parser.parse_args()

    if args.save is None:
        args.save = "figures/view_spatial_frequency/spatial_frequency.png"

    # ── Load panoramas ────────────────────────────────────────────────
    print("Loading panoramas …")
    all_panos = _load_all_panos()
    total, H, W = all_panos.shape

    if args.pano_indices is not None:
        panos = all_panos[args.pano_indices]
        idx_label = str(args.pano_indices)
    elif args.max_panos is not None:
        panos = all_panos[:args.max_panos]
        idx_label = f"first {panos.shape[0]}"
    else:
        panos = all_panos
        idx_label = "all"
    n_panos = panos.shape[0]
    print(f"  {n_panos}/{total} panoramas ({idx_label}), {H} x {W} pixels")

    # Azimuth spans 360 deg; frequency bin k corresponds to k cycles / 360 deg,
    # i.e. wavelength = 360 / k degrees.
    n_freq = W // 2  # number of positive-frequency bins from rfft (excluding DC)
    freq_bins = np.arange(1, n_freq + 1)  # k = 1, 2, ..., W//2
    wavelength_deg = 360.0 / freq_bins

    # ── Compute spectra ───────────────────────────────────────────────
    print("Computing power spectra …")
    raw_powers = np.zeros(n_freq)
    filt_powers = np.zeros(n_freq)

    for i in range(n_panos):
        raw_powers += azimuthal_power_spectrum(panos[i])
        filt_powers += azimuthal_power_spectrum(_filter_pano(panos[i]))
    raw_powers /= n_panos
    filt_powers /= n_panos

    # ── Plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.loglog(wavelength_deg, raw_powers, linewidth=1, label="raw panorama")
    ax.loglog(wavelength_deg, filt_powers, linewidth=1, label="after Gaussian (σ=2.5°)")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, wl in enumerate(args.wavelengths):
        c = colors[(i + 2) % len(colors)]
        ax.axvline(wl, color=c, linestyle="--", linewidth=1.2,
                   label=f"λ = {wl}°")

    ax.set_xlabel("Spatial wavelength (degrees)")
    ax.set_ylabel("Power (mean |FFT|²)")
    ax.set_title(
        f"Azimuthal spatial power spectrum of natural panoramas (n={n_panos})",
        fontweight="bold",
    )
    ax.legend()
    ax.invert_xaxis()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
