"""Bake the large raw inputs into cached ``.pt`` files under ``stimulus/baked``.

This is the first stage of the data pipeline. The raw panorama ``.mat`` files
are large and slow to decode, so this script reads them once, straight out of
the ``panorama_001-100.rar`` archive in ``<data_folder>/stimulus/raw`` (no
extraction step), and writes a compact ``.pt`` cache to
``<data_folder>/stimulus/baked``:

* ``random_pano`` -> ``all_panoramic.pt`` ((N, H, W) raw panorama images)

The panorama cache stores the unfiltered images; the Gaussian ommatidial
filtering is applied at generation time in ``stimulus_generators._filter_pano``.

The generators in ``stimulus_generators.py`` read these caches when training
and analysis scripts generate their stimuli.

Usage::

    python -m drosophila_model.data_handling.process_raw_data            # all
    python -m drosophila_model.data_handling.process_raw_data random_pano
"""
import argparse
import io
from pathlib import Path

import numpy as np
import torch

from drosophila_model.config import get_data_folder, load_config

RAW_TYPES = ('random_pano',)
# data_001-100.rar from Meyer et al. 2014 (doi:10.4119/unibi/2689637), renamed.
PANORAMA_ARCHIVE = 'panorama_001-100.rar'


def loadmat_v73(source):
    """Read a MATLAB v7.3 ``.mat`` file (path or file object) into a dict of numpy arrays."""
    import h5py
    out = {}
    with h5py.File(source, "r") as f:
        for k in f.keys():
            out[k] = np.array(f[k])
    return out


def _extract_pano_image(mat_file, name):
    """Extract the raw panorama image array from a single ``.mat`` file.

    Mirrors the historical loader: take the 2D arrays in the file (dropping the
    first, which is MATLAB metadata), pick the largest, and transpose it.
    ``mat_file`` is a path or file object; ``name`` is used in errors.
    """
    pano_data = loadmat_v73(mat_file)
    pano_candidates = [np.asarray(v) for v in pano_data.values() if np.ndim(v) == 2][1:]
    if not pano_candidates:
        raise ValueError(
            f"Panoramic .mat file {name} did not contain a 2D array. "
            "Keys: " + ", ".join(pano_data.keys())
        )
    return max(pano_candidates, key=lambda a: a.size).T


def process_random_pano(data_folder):
    """Consolidate the raw panorama ``.mat`` images into a single ``.pt``.

    The ``.mat`` files are read in memory from :data:`PANORAMA_ARCHIVE`, in
    file-name order. Returns the path of the written ``all_panoramic.pt`` file.
    """
    from unrar.cffi import rarfile

    data_folder = Path(data_folder)
    archive = data_folder / 'stimulus' / 'raw' / PANORAMA_ARCHIVE
    baked_dir = data_folder / 'stimulus' / 'baked'
    if not archive.exists():
        raise FileNotFoundError(
            f"Panorama archive not found: {archive}. It is data_001-100.rar from "
            "Meyer et al. 2014 (doi:10.4119/unibi/2689637), renamed."
        )
    rar = rarfile.RarFile(str(archive))
    mat_names = sorted(n for n in rar.namelist() if n.endswith('.mat'))
    print(f"[random_pano] Found {len(mat_names)} panorama .mat files in {archive.name}.")

    panos = []
    for name in mat_names:
        print(f"[random_pano] Loading {name}...")
        panos.append(_extract_pano_image(io.BytesIO(rar.read(name)), name))

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

    if raw_type == 'random_pano':
        return process_random_pano(data_folder)
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
