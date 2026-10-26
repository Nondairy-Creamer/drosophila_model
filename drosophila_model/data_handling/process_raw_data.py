"""Bake the large raw inputs into cached ``.pt`` files under ``stimulus/baked``.

This is the first stage of the data pipeline. The truly raw inputs (drone
videos, panorama ``.mat`` files) are large and slow to decode, so this script
reads them once from ``<data_folder>/stimulus/raw`` and writes a compact ``.pt``
cache to ``<data_folder>/stimulus/baked``:

* ``small_drone`` -> ``small_drone.pt``  (ommatidium-sampled stereo clips)
* ``rpi_drone``   -> ``rpi_drone.pt``    (ommatidium-sampled stereo clips)
* ``random_pano`` -> ``all_panoramic.pt`` ((N, H, W) raw panorama images)
* ``flyview``     -> ``flyview.pt``     (flow + depth-based odometry,
  baked from ``fib_forest_fly_view_dataset_fwd.pkl``)

The drone caches store clips spatially sampled at the connectome ommatidium
columns but still at the native video frame rate; temporal interpolation to the
model ``dt`` happens later in
:func:`drosophila_model.data_handling.drone_utilities.get_drone_stimulus`. The
panorama cache stores the unfiltered images; the Gaussian ommatidial filtering
is applied at generation time in ``stimulus_generators._filter_pano``.

The second stage, ``drosophila_model.data_handling.create_training_data``,
reads these caches (via the on-the-fly generators) and writes the fixed,
training-ready datasets to ``stimulus/training/<stimulus_type>_train.pt`` and
``<stimulus_type>_test.pt``.

Usage::

    python -m drosophila_model.data_handling.process_raw_data            # all
    python -m drosophila_model.data_handling.process_raw_data small_drone
    python -m drosophila_model.data_handling.process_raw_data rpi_drone
    python -m drosophila_model.data_handling.process_raw_data random_pano
    python -m drosophila_model.data_handling.process_raw_data flyview
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from drosophila_model.config import get_data_folder, load_config
from drosophila_model.data_handling import drone_utilities as du
from drosophila_model.data_handling.sampling import load_column_phi_theta

RAW_TYPES = ('small_drone', 'rpi_drone', 'random_pano', 'flyview')


def loadmat_v73(path):
    """Read a MATLAB v7.3 ``.mat`` file into a dict of numpy arrays."""
    import h5py
    out = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            out[k] = np.array(f[k])
    return out


def process_drone(dataset, data_folder):
    """Process a drone rig into ommatidium-sampled clips and cache them.

    The clips are sampled at the right-eye columns (and their left-eye mirror)
    of ``optic_lobe_columns.parquet``; the columns are stored in the cache and
    ``get_drone_stimulus`` checks callers' columns against them.

    Returns the path of the written ``.pt`` file.
    """
    data_folder = Path(data_folder)
    raw_dir = data_folder / 'stimulus' / 'raw'
    baked_dir = data_folder / 'stimulus' / 'baked'
    column_phi_theta = load_column_phi_theta(data_folder)
    sample_phi, sample_theta = column_phi_theta[:, 0], column_phi_theta[:, 1]

    if dataset == 'small_drone':
        clips = du.process_small_drone_dataset(
            raw_dir / 'drone_videos', raw_dir / 'drone_poses',
            sample_phi, sample_theta)
    elif dataset == 'rpi_drone':
        clips = du.process_rpi_drone_dataset(
            raw_dir / 'rpi_drone', raw_dir / 'rpi_drone' / 'dome_calib',
            sample_phi, sample_theta)
    else:
        raise ValueError(f"Unknown drone dataset '{dataset}'.")

    if not clips:
        raise RuntimeError(f"No clips produced for '{dataset}' from {raw_dir}.")

    baked_dir.mkdir(parents=True, exist_ok=True)
    out_path = baked_dir / f'{dataset}.pt'
    torch.save({
        'dataset': dataset,
        'clips': clips,
        'sample_phi': np.asarray(sample_phi),
        'sample_theta': np.asarray(sample_theta),
    }, out_path)

    print(f"Cached '{dataset}' ({len(clips)} clips) -> {out_path}")
    return out_path


def _find_pano_mat_files(raw_dir):
    """Locate the panorama ``.mat`` files, raising a clear error if missing.

    Looks in the dedicated ``panoramic/`` subfolder first (where the archive is
    meant to be extracted), then falls back to ``raw_dir`` itself.
    """
    search_dirs = [raw_dir / 'panoramic', raw_dir]
    for d in search_dirs:
        mats = sorted(d.glob('*.mat'))
        if mats:
            return mats

    rar_files = sorted(raw_dir.glob('panorama_*.rar'))
    if rar_files:
        raise FileNotFoundError(
            f"Found {rar_files[0].name} but no extracted .mat files. "
            f"Extract the archive into {raw_dir / 'panoramic'} and re-run."
        )
    raise FileNotFoundError(
        f"No panorama .mat files found in {raw_dir / 'panoramic'} or {raw_dir}."
    )


def _extract_pano_image(mat_path):
    """Extract the raw panorama image array from a single ``.mat`` file.

    Mirrors the historical loader: take the 2D arrays in the file (dropping the
    first, which is MATLAB metadata), pick the largest, and transpose it.
    """
    pano_data = loadmat_v73(str(mat_path))
    pano_candidates = [np.asarray(v) for v in pano_data.values() if np.ndim(v) == 2][1:]
    if not pano_candidates:
        raise ValueError(
            f"Panoramic .mat file {mat_path} did not contain a 2D array. "
            "Keys: " + ", ".join(pano_data.keys())
        )
    return max(pano_candidates, key=lambda a: a.size).T


def process_random_pano(data_folder):
    """Consolidate the raw panorama ``.mat`` images into a single ``.pt``.

    Returns the path of the written ``all_panoramic.pt`` file.
    """
    data_folder = Path(data_folder)
    raw_dir = data_folder / 'stimulus' / 'raw'
    baked_dir = data_folder / 'stimulus' / 'baked'
    mat_files = _find_pano_mat_files(raw_dir)
    print(f"[random_pano] Found {len(mat_files)} panorama .mat files.")

    panos = []
    for mat_path in mat_files:
        print(f"[random_pano] Loading {mat_path.name}...")
        panos.append(_extract_pano_image(mat_path))

    panos = np.stack(panos).astype(np.float32)
    baked_dir.mkdir(parents=True, exist_ok=True)
    out_path = baked_dir / 'all_panoramic.pt'
    torch.save(torch.from_numpy(panos), out_path)

    print(f"Cached 'random_pano' {tuple(panos.shape)} -> {out_path}")
    return out_path


def process_raw_data(raw_type, config=None, config_path=None):
    """Bake ``raw_type`` from its raw inputs into a ``stimulus/baked`` cache."""
    if config is None:
        config = load_config(config_path or 'config.toml')
    data_folder = get_data_folder(config)

    if raw_type in ('small_drone', 'rpi_drone'):
        return process_drone(raw_type, data_folder)
    if raw_type == 'random_pano':
        return process_random_pano(data_folder)
    if raw_type == 'flyview':
        from drosophila_model.data_handling import stimulus_generators as sg
        return sg.bake_fly_view_dataset(data_folder=data_folder)
    raise ValueError(
        f"Unknown raw_type '{raw_type}'; expected one of {RAW_TYPES}.")


def main():
    parser = argparse.ArgumentParser(
        description='Bake raw inputs into cached .pt files under stimulus/baked.'
    )
    parser.add_argument(
        'raw_type', type=str, nargs='?', default=None, choices=RAW_TYPES,
        help=f"Raw dataset to process ({', '.join(RAW_TYPES)}). "
             'Omit to process all of them.'
    )
    parser.add_argument(
        '--config-path', type=str, default=None,
        help='Path to config TOML (default: config.toml)'
    )
    args = parser.parse_args()

    raw_types = [args.raw_type] if args.raw_type is not None else list(RAW_TYPES)
    for raw_type in raw_types:
        process_raw_data(raw_type, config_path=args.config_path)


if __name__ == '__main__':
    main()
