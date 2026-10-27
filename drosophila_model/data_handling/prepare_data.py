"""Download the data from Zenodo and bake everything the code reads.

Fills ``[data].data_folder`` (``config.get_data_folder``) in two steps,
skipping any whose outputs already exist:

1. Baked files: the connectome tables in ``connectome_data/baked/`` and the
   panorama cache ``stimulus/baked/all_panoramic.pt``. By default they are
   downloaded ready-made, as ``drosophila_model_baked_v1.zip`` (about
   250 MB) from Zenodo record :data:`ZENODO_RECORD`. With ``--raw``, or when
   the raw inputs are already in the data folder, they are baked from the raw
   inputs instead, by ``format_connectome_data`` (connectome tables, CT1
   split, column geometry) and ``process_raw_data`` (panorama cache). Missing
   raw inputs are downloaded as ``drosophila_model_raw_v1.zip`` (about
   3.2 GB): the FlyWire Codex downloads and the Reiser lab eye map in
   ``connectome_data/raw/``, and the panorama archive in ``stimulus/raw/``.
2. Training sets: ``create_training_data`` for every type in
   ``[stimulus].stimulus_type``.

A downloaded bundle is checked against the MD5 Zenodo lists, unpacked and
deleted. ``--zip`` unpacks a local copy of either bundle first instead. Files
already in the data folder are never overwritten.

``setup.sh`` runs this after installing the dependencies.

Usage:
    python -m drosophila_model.data_handling.prepare_data
    python -m drosophila_model.data_handling.prepare_data --raw
    python -m drosophila_model.data_handling.prepare_data --zip path/to/drosophila_model_baked_v1.zip
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
BAKED_BUNDLE = 'drosophila_model_baked_v1.zip'
RAW_BUNDLE = 'drosophila_model_raw_v1.zip'
# The raw inputs in the raw bundle, relative to the data folder.
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
BAKED_FILES = (*CONNECTOME_FILES, PANORAMA_FILE)


def _missing(data_folder, files):
    """The entries of ``files`` not in ``data_folder``."""
    return [f for f in files if not (data_folder / f).exists()]


def _zenodo_file(name):
    """Return ``(url, md5, size)`` of file ``name`` in :data:`ZENODO_RECORD`."""
    if not ZENODO_RECORD:
        raise RuntimeError(
            "The Zenodo record for the data is not set yet "
            "(ZENODO_RECORD in drosophila_model/data_handling/prepare_data.py). "
            f"Pass a local copy of {name} with --zip."
        )
    with urllib.request.urlopen(f'https://zenodo.org/api/records/{ZENODO_RECORD}') as response:
        files = json.load(response)['files']
    entry = next((f for f in files if f['key'] == name), None)
    if entry is None:
        raise RuntimeError(f"Zenodo record {ZENODO_RECORD} has no file {name}.")
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
                print(f"  {done / 1e6:,.0f} / {size / 1e6:,.0f} MB", flush=True)
                next_report += size // 20
    if digest.hexdigest() != md5:
        part_path.unlink()
        raise RuntimeError(f"MD5 mismatch for {url}; the download is corrupt, re-run to retry.")
    part_path.replace(out_path)


def _unpack(zip_path, data_folder):
    """Unpack a bundle into ``data_folder``, keeping files already there.

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


def get_bundle(name, data_folder, expected):
    """Download bundle ``name`` from Zenodo and unpack it into ``data_folder``.

    Raises unless every file in ``expected`` is in ``data_folder`` afterwards.
    """
    url, md5, size = _zenodo_file(name)
    downloaded = data_folder / name
    print(f"Downloading {name} ({size / 1e6:,.0f} MB) from Zenodo record {ZENODO_RECORD} ...")
    _download(url, downloaded, md5, size)
    print(f"Unpacking {name} into {data_folder} ...")
    _unpack(downloaded, data_folder)
    downloaded.unlink()
    if missing := _missing(data_folder, expected):
        raise RuntimeError(f"{name} did not contain {missing}.")


def _run(module, *args):
    """Run ``python -m drosophila_model.data_handling.<module> <args>``."""
    sys.stdout.flush()  # keep our messages ahead of the child's output
    subprocess.run(
        [sys.executable, '-m', f'drosophila_model.data_handling.{module}', *args],
        check=True,
    )


def bake_from_raw(data_folder):
    """Bake the missing :data:`BAKED_FILES` from the raw inputs, downloading those if needed."""
    if missing := _missing(data_folder, RAW_FILES):
        print(f"Raw data: {len(missing)} file(s) missing, e.g. {missing[0]}")
        get_bundle(RAW_BUNDLE, data_folder, RAW_FILES)
    if _missing(data_folder, CONNECTOME_FILES):
        print("Connectome: baking from the raw data ...")
        _run('format_connectome_data')
    if _missing(data_folder, [PANORAMA_FILE]):
        print("Panoramas: baking from the raw data ...")
        _run('process_raw_data', 'random_pano')


def prepare_data(raw=False, zip_path=None):
    """Get the baked files (downloaded, or baked from raw), then bake the training sets."""
    config = load_config()
    data_folder = get_data_folder(config)
    print(f"Data folder: {data_folder}")
    if zip_path is not None:
        print(f"Unpacking {zip_path} into {data_folder} ...")
        _unpack(Path(zip_path), data_folder)

    if not _missing(data_folder, BAKED_FILES):
        print("Baked files: present")
    elif raw or not _missing(data_folder, RAW_FILES):
        bake_from_raw(data_folder)
    else:
        get_bundle(BAKED_BUNDLE, data_folder, BAKED_FILES)

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
        description='Download the data from Zenodo and bake everything the code '
                    'reads into [data].data_folder, skipping finished steps.'
    )
    parser.add_argument(
        '--raw', action='store_true',
        help=f'Bake the connectome tables and panorama cache from the raw data '
             f'({RAW_BUNDLE}, about 3.2 GB) instead of downloading them ready-made '
             f'({BAKED_BUNDLE}, about 250 MB).'
    )
    parser.add_argument(
        '--zip', type=str, default=None,
        help=f'Unpack a local copy of {BAKED_BUNDLE} or {RAW_BUNDLE} first, '
             f'instead of downloading it.'
    )
    args = parser.parse_args()
    prepare_data(raw=args.raw, zip_path=args.zip)


if __name__ == '__main__':
    main()
