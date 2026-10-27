"""Download the data from the GitHub data release and bake everything the code reads.

Fills ``[data].data_folder`` (``config.get_data_folder``) in two steps,
skipping any whose outputs already exist:

1. Baked files: the connectome tables in ``connectome_data/baked/`` and the
   panorama cache ``stimulus/baked/all_panoramic.pt``. By default they are
   downloaded ready-made, as ``drosophila_model_baked_v1.zip`` (about
   250 MB) from the release :data:`DATA_RELEASE` of :data:`DATA_REPO`. With
   ``--raw``, or when the raw inputs are already in the data folder, they are
   baked from the raw inputs instead, by ``format_connectome_data``
   (connectome tables, CT1 split, column geometry) and ``process_raw_data``
   (panorama cache). Missing raw inputs are downloaded as
   ``drosophila_model_raw_v1.zip`` (about 3.2 GB, in parts under GitHub's
   2 GiB file limit): the FlyWire Codex downloads and the Reiser lab eye map
   in ``connectome_data/raw/``, and the panorama archive in ``stimulus/raw/``.
2. Training sets: ``create_training_data`` for every type in
   ``[stimulus].stimulus_type``.

A downloaded bundle is checked against the SHA-256 GitHub lists, unpacked and
deleted. If the repository is private, the download uses the ``GH_TOKEN`` or
``GITHUB_TOKEN`` environment variable, else the GitHub CLI's login
(``gh auth login``). ``--zip`` unpacks a local copy of either bundle first
instead. Files already in the data folder are never overwritten.

``setup.sh`` runs this after installing the dependencies. The bundles are
built and uploaded by ``build_data_release``.

Usage:
    python -m drosophila_model.data_handling.prepare_data
    python -m drosophila_model.data_handling.prepare_data --raw
    python -m drosophila_model.data_handling.prepare_data --zip path/to/drosophila_model_baked_v1.zip
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from drosophila_model.config import get_data_folder, load_config


DATA_REPO = 'Nondairy-Creamer/drosophila_model'
# The data the code downloads: the release tagged data-<DATA_VERSION>. Bump it
# (and publish that release with build_data_release) when the data change.
DATA_VERSION = 'v1'
DATA_RELEASE = f'data-{DATA_VERSION}'
BAKED_BUNDLE = f'drosophila_model_baked_{DATA_VERSION}.zip'
RAW_BUNDLE = f'drosophila_model_raw_{DATA_VERSION}.zip'
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


def _github_token():
    """A GitHub token from ``GH_TOKEN`` / ``GITHUB_TOKEN`` or the GitHub CLI's login, else None."""
    for variable in ('GH_TOKEN', 'GITHUB_TOKEN'):
        if os.environ.get(variable):
            return os.environ[variable]
    try:
        result = subprocess.run(['gh', 'auth', 'token'], capture_output=True, text=True)
    except OSError:  # no GitHub CLI
        return None
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None


def _open(url, token=None, accept='application/vnd.github+json'):
    """Open ``url``, authenticated with ``token`` if given."""
    request = urllib.request.Request(url, headers={'Accept': accept, 'User-Agent': 'drosophila_model'})
    if token:
        # Unredirected: not passed on to the storage host GitHub redirects downloads to.
        request.add_unredirected_header('Authorization', f'Bearer {token}')
    return urllib.request.urlopen(request)


def _release_assets():
    """Return ``(assets, token)`` for :data:`DATA_RELEASE`.

    Tries without a login first; ``token`` is the one needed to download the
    assets (None for a public repository).
    """
    url = f'https://api.github.com/repos/{DATA_REPO}/releases/tags/{DATA_RELEASE}'
    try:
        with _open(url) as response:
            return json.load(response)['assets'], None
    except urllib.error.HTTPError as error:
        if error.code != 404:  # a private repository looks missing without a login
            raise
    where = f"release {DATA_RELEASE} of github.com/{DATA_REPO}"
    token = _github_token()
    if token is None:
        raise RuntimeError(
            f"Could not find {where}. If the repository is private, log in with "
            f"`gh auth login` or set GITHUB_TOKEN to a token that can read it, "
            f"then re-run. Or pass a local copy of the bundle with --zip."
        )
    try:
        with _open(url, token) as response:
            return json.load(response)['assets'], token
    except urllib.error.HTTPError as error:
        if error.code in (401, 403, 404):
            raise RuntimeError(
                f"Could not read {where} with your GitHub login ({error}). Check that "
                f"the release exists and that the login can read the repository "
                f"(`gh auth status`), or that GH_TOKEN / GITHUB_TOKEN is a valid "
                f"token for it."
            ) from None
        raise


def _bundle_assets(name, assets):
    """The release assets that make up bundle ``name``, in order.

    That is the asset ``name`` itself or, for a bundle over GitHub's 2 GiB
    file limit, its parts ``name.001``, ``name.002``, ...
    """
    if whole := [a for a in assets if a['name'] == name]:
        return whole
    pattern = re.escape(name) + r'\.\d{3}'
    if parts := sorted((a for a in assets if re.fullmatch(pattern, a['name'])), key=lambda a: a['name']):
        return parts
    raise RuntimeError(f"Release {DATA_RELEASE} of github.com/{DATA_REPO} has no {name}.")


def _download(assets, out_path, token):
    """Stream ``assets`` to ``out_path``, joined in order, checking each one's SHA-256."""
    part_path = out_path.with_name(out_path.name + '.part')
    size = sum(a['size'] for a in assets)
    done, next_report = 0, 0
    with open(part_path, 'wb') as out:
        for asset in assets:
            # The API URL takes a login (for a private repository); the browser one does not.
            url = asset['url'] if token else asset['browser_download_url']
            digest = hashlib.sha256()
            with _open(url, token, accept='application/octet-stream') as response:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
                    digest.update(chunk)
                    done += len(chunk)
                    if done >= next_report:
                        print(f"  {done / 1e6:,.0f} / {size / 1e6:,.0f} MB", flush=True)
                        next_report += size // 20
            if asset.get('digest') not in (None, f'sha256:{digest.hexdigest()}'):
                out.close()
                part_path.unlink()
                raise RuntimeError(f"SHA-256 mismatch for {asset['name']}; the download is corrupt, re-run to retry.")
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
    """Download bundle ``name`` from :data:`DATA_RELEASE` and unpack it into ``data_folder``.

    Raises unless every file in ``expected`` is in ``data_folder`` afterwards.
    """
    assets, token = _release_assets()
    assets = _bundle_assets(name, assets)
    size = sum(a['size'] for a in assets)
    downloaded = data_folder / name
    print(f"Downloading {name} ({size / 1e6:,.0f} MB) from release {DATA_RELEASE} of github.com/{DATA_REPO} ...")
    _download(assets, downloaded, token)
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
        description='Download the data from the GitHub data release and bake '
                    'everything the code reads into [data].data_folder, '
                    'skipping finished steps.'
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
