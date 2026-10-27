"""Download the raw data from Zenodo and bake everything the code reads.

Fills ``[data].data_folder`` (``config.get_data_folder``) in three steps,
skipping any whose outputs already exist:

1. Raw inputs: when a bake step below needs them and they are missing, the
   data bundle ``drosophila_model_data_v1.zip`` is downloaded from Zenodo
   record :data:`ZENODO_RECORD` (or taken from ``--zip``), checked against the
   MD5 Zenodo lists, and unpacked. It holds only raw files: the FlyWire Codex
   downloads and the Reiser lab eye map in ``connectome_data/raw/``, and the
   panorama archive in ``stimulus/raw/``. A downloaded zip is deleted after
   unpacking; files already in the data folder are never overwritten.
2. Baked files: ``format_connectome_data`` (connectome tables, CT1 split,
   column geometry) and ``process_raw_data`` (panorama cache).
3. Training sets: ``create_training_data`` for every type in
   ``[stimulus].stimulus_type``.

``setup.sh`` runs this after installing the dependencies.

Usage:
    python -m drosophila_model.data_handling.prepare_data
    python -m drosophila_model.data_handling.prepare_data --zip path/to/drosophila_model_data_v1.zip
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from drosophila_model.config import get_data_folder, load_config


ZENODO_RECORD = ''  # TODO: the record number, once the Zenodo record is reserved
BUNDLE_NAME = 'drosophila_model_data_v1.zip'
# The raw inputs in the bundle, relative to the data folder.
RAW_FILES = (
    'connectome_data/raw/consolidated_cell_types.csv.gz',
    'connectome_data/raw/connections_princeton_no_threshold.csv.gz',
    'connectome_data/raw/column_assignment.csv.gz',
    'connectome_data/raw/fafb_v783_princeton_synapse_table.csv.gz',
    'connectome_data/raw/cell_stats.csv.gz',
    'connectome_data/raw/classification.csv.gz',
    'connectome_data/raw/eyemap_T4/data/eyemap.RData',
    'stimulus/raw/panorama_001-100.rar',
)
CONNECTOME_FILES = tuple(
    f'connectome_data/baked/{name}' for name in (
        'cell_info.parquet', 'synapses.parquet',
        'optic_lobe_column_index.parquet', 'optic_lobe_columns.parquet',
    )
)
PANORAMA_FILE = 'stimulus/baked/all_panoramic.pt'


def _zenodo_bundle():
    """Return ``(url, md5, size)`` of the bundle in :data:`ZENODO_RECORD`."""
    if not ZENODO_RECORD:
        raise RuntimeError(
            "The Zenodo record for the data bundle is not set yet "
            "(ZENODO_RECORD in drosophila_model/data_handling/prepare_data.py). "
            f"Pass a local copy of {BUNDLE_NAME} with --zip."
        )
    with urllib.request.urlopen(f'https://zenodo.org/api/records/{ZENODO_RECORD}') as response:
        files = json.load(response)['files']
    entry = next((f for f in files if f['key'] == BUNDLE_NAME), None)
    if entry is None:
        raise RuntimeError(f"Zenodo record {ZENODO_RECORD} has no file {BUNDLE_NAME}.")
    return entry['links']['self'], entry['checksum'].removeprefix('md5:'), entry['size']


def _download(url, out_path, md5, size):
    """Stream ``url`` to ``out_path``, checking its MD5."""
    part_path = out_path.with_name(out_path.name + '.part')
    digest = hashlib.md5()
    done, next_report = 0, 0
    with urllib.request.urlopen(url) as response, open(part_path, 'wb') as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
            digest.update(chunk)
            done += len(chunk)
            if done >= next_report:
                print(f"  {done / 1e9:.2f} / {size / 1e9:.2f} GB", flush=True)
                next_report += size // 20
    if digest.hexdigest() != md5:
        part_path.unlink()
        raise RuntimeError(f"MD5 mismatch for {url}; the download is corrupt, re-run to retry.")
    part_path.replace(out_path)


def _unpack(zip_path, data_folder):
    """Unpack the bundle into ``data_folder``, keeping files already there.

    The zip has one top-level folder (``drosophila_model_data/``), which is
    dropped so its contents land directly in ``data_folder``.
    """
    with zipfile.ZipFile(zip_path) as bundle:
        for info in bundle.infolist():
            parts = Path(info.filename).parts[1:]
            if info.is_dir() or not parts:
                continue
            if '..' in parts:
                raise ValueError(f"Unsafe path in {zip_path}: {info.filename}")
            target = data_folder.joinpath(*parts)
            if target.exists():
                continue
            print(f"  {Path(*parts).as_posix()}")
            target.parent.mkdir(parents=True, exist_ok=True)
            part_path = target.with_name(target.name + '.part')
            with bundle.open(info) as src, open(part_path, 'wb') as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            part_path.replace(target)


def get_raw_data(data_folder, zip_path=None):
    """Make sure every file in :data:`RAW_FILES` is in ``data_folder``."""
    missing = [f for f in RAW_FILES if not (data_folder / f).exists()]
    if not missing:
        print("Raw data: present")
        return
    print(f"Raw data: {len(missing)} file(s) missing, e.g. {missing[0]}")
    if zip_path is None:
        url, md5, size = _zenodo_bundle()
        downloaded = data_folder / BUNDLE_NAME
        print(f"Downloading {BUNDLE_NAME} ({size / 1e9:.1f} GB) from Zenodo record {ZENODO_RECORD} ...")
        _download(url, downloaded, md5, size)
        print(f"Unpacking {BUNDLE_NAME} into {data_folder} ...")
        _unpack(downloaded, data_folder)
        downloaded.unlink()
    else:
        print(f"Unpacking {zip_path} into {data_folder} ...")
        _unpack(Path(zip_path), data_folder)
    still_missing = [f for f in RAW_FILES if not (data_folder / f).exists()]
    if still_missing:
        raise RuntimeError(f"The data bundle did not contain {still_missing}.")


def _run(module, *args):
    """Run ``python -m drosophila_model.data_handling.<module> <args>``."""
    sys.stdout.flush()  # keep our messages ahead of the child's output
    subprocess.run(
        [sys.executable, '-m', f'drosophila_model.data_handling.{module}', *args],
        check=True,
    )


def prepare_data(zip_path=None):
    """Download the raw data if needed, then bake whatever is missing."""
    config = load_config()
    data_folder = get_data_folder(config)
    print(f"Data folder: {data_folder}")

    need_connectome = not all((data_folder / f).exists() for f in CONNECTOME_FILES)
    need_panoramas = not (data_folder / PANORAMA_FILE).exists()
    if need_connectome or need_panoramas:
        get_raw_data(data_folder, zip_path)

    if need_connectome:
        print("Connectome: baking ...")
        _run('format_connectome_data')
    else:
        print("Connectome: baked files present")
    if need_panoramas:
        print("Panoramas: baking ...")
        _run('process_raw_data', 'random_pano')
    else:
        print("Panoramas: baked file present")

    stimulus_type = config['stimulus']['stimulus_type']
    training_dir = data_folder / 'stimulus' / 'training'
    for name in [stimulus_type] if isinstance(stimulus_type, str) else stimulus_type:
        if (training_dir / f'{name}_train.pt').exists():
            print(f"Training set {name}: present")
        else:
            print(f"Training set {name}: baking ...")
            _run('create_training_data', name)


def main():
    parser = argparse.ArgumentParser(
        description='Download the raw data from Zenodo and bake everything the code '
                    'reads into [data].data_folder, skipping finished steps.'
    )
    parser.add_argument(
        '--zip', type=str, default=None,
        help=f'Use a local copy of {BUNDLE_NAME} instead of downloading it.'
    )
    args = parser.parse_args()
    prepare_data(zip_path=args.zip)


if __name__ == '__main__':
    main()
