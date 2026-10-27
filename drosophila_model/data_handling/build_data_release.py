"""Build the data release that ``prepare_data`` downloads, and upload it.

Zips the baked files and the raw inputs in ``[data].data_folder``
(``config.get_data_folder``) into ``drosophila_model_baked_<version>.zip`` and
``drosophila_model_raw_<version>.zip``, with ``data_release_README.md`` inside
each as ``README.md``. <version> is ``prepare_data.DATA_VERSION``. GitHub caps
a release file at 2 GiB, so a larger zip is split into equal parts
``<zip>.001``, ``<zip>.002``, ..., which ``prepare_data`` joins again.

The zips and the README are written to a temporary folder (about 3.5 GB in
the system's temp folder), uploaded to the release
``prepare_data.DATA_RELEASE`` with the GitHub CLI (``gh``, logged in with write
access to the repository), creating it if needed, and the folder is removed.
Files of the same name are replaced, so code already pointing at that release
gets the new data; bump ``DATA_VERSION`` first to publish a new release
instead. ``--out-dir`` writes the files to a folder of your choice and stops
there, without uploading.

Usage:
    python -m drosophila_model.data_handling.build_data_release
    python -m drosophila_model.data_handling.build_data_release --only baked
    python -m drosophila_model.data_handling.build_data_release --out-dir path/to/folder
"""
import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

from drosophila_model.config import get_data_folder, load_config
from drosophila_model.data_handling.prepare_data import (
    BAKED_BUNDLE, DATA_RELEASE, DATA_REPO, DATA_VERSION, RAW_BUNDLE, RAW_FILES,
)


README = Path(__file__).with_name('data_release_README.md')
EYEMAP_LICENSE = 'connectome_data/raw/eyemap_T4/LICENSE'
# (path in the zip under drosophila_model_data/, source relative to the data folder)
BUNDLE_FILES = {
    BAKED_BUNDLE: [
        *((f'connectome_data/baked/{name}',) * 2 for name in (
            'cell_info.parquet', 'synapses.parquet', 'optic_lobe_column_index.parquet',
            'optic_lobe_columns.parquet', 'ct1_pieces.parquet', 'ct1_connections.parquet',
        )),
        ('stimulus/baked/all_panoramic.pt',) * 2,
        ('eyemap_T4_LICENSE', EYEMAP_LICENSE),
    ],
    RAW_BUNDLE: [(path, path) for path in sorted({*RAW_FILES, EYEMAP_LICENSE})],
}
# Already-compressed files are stored; everything else is deflated.
STORED = ('.gz', '.rar', '.RData')
MAX_ASSET = 2_000_000_000  # bytes; GitHub's limit is 2 GiB per release file


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def build_bundle(name, data_folder, out_dir, readme):
    """Zip bundle ``name`` into ``out_dir``, split into parts if too large; return the files."""
    out = out_dir / name
    tmp = out.with_name(name + '.tmp')
    print(f"Building {name} ...", flush=True)
    with zipfile.ZipFile(tmp, 'w') as bundle:
        bundle.writestr('drosophila_model_data/README.md', readme, zipfile.ZIP_DEFLATED)
        for arcname, source in BUNDLE_FILES[name]:
            method = zipfile.ZIP_STORED if source.endswith(STORED) else zipfile.ZIP_DEFLATED
            bundle.write(data_folder / source, f'drosophila_model_data/{arcname}', method, compresslevel=9)
            print(f"  {arcname}", flush=True)
    with zipfile.ZipFile(tmp) as bundle:
        if (bad := bundle.testzip()) is not None:
            raise RuntimeError(f"{name}: {bad} is corrupt")
    for old in out_dir.glob(f'{name}.[0-9][0-9][0-9]'):
        old.unlink()
    size = tmp.stat().st_size
    if size <= MAX_ASSET:
        tmp.replace(out)
        return [out]
    n_parts = -(-size // MAX_ASSET)
    part_size = -(-size // n_parts)
    parts = [out_dir / f'{name}.{i:03d}' for i in range(1, n_parts + 1)]
    with open(tmp, 'rb') as src:
        for part in parts:
            with open(part, 'wb') as dst:
                remaining = part_size
                while remaining and (chunk := src.read(min(1 << 24, remaining))):
                    dst.write(chunk)
                    remaining -= len(chunk)
    tmp.unlink()
    out.unlink(missing_ok=True)
    return parts


def _gh(*args, **kwargs):
    return subprocess.run(['gh', *args, '--repo', DATA_REPO], check=True, **kwargs)


def upload(files, sha256s, bundles, readme_path):
    """Upload ``files`` to :data:`DATA_RELEASE`, creating it if needed, and check their ``sha256s`` on GitHub.

    Earlier assets of ``bundles`` (a whole zip or its parts) not among
    ``files`` are deleted, so a bundle split differently than before is not
    joined with stale parts.
    """
    exists = subprocess.run(
        ['gh', 'release', 'view', DATA_RELEASE, '--repo', DATA_REPO], capture_output=True,
    ).returncode == 0
    if exists:
        print(f"Uploading to release {DATA_RELEASE} of github.com/{DATA_REPO} ...", flush=True)
        _gh('release', 'upload', DATA_RELEASE, *files, '--clobber')
        _gh('release', 'edit', DATA_RELEASE, '--notes-file', readme_path)
    else:
        print(f"Creating release {DATA_RELEASE} of github.com/{DATA_REPO} ...", flush=True)
        _gh('release', 'create', DATA_RELEASE, *files,
            '--title', f'Data {DATA_VERSION}', '--notes-file', readme_path)

    release = json.loads(subprocess.run(
        ['gh', 'api', f'repos/{DATA_REPO}/releases/tags/{DATA_RELEASE}'],
        check=True, capture_output=True, text=True,
    ).stdout)
    names = {f.name for f in files}
    pattern = '|'.join(re.escape(b) for b in bundles)
    for asset in release['assets']:
        if asset['name'] not in names and re.fullmatch(rf'({pattern})(\.\d{{3}})?', asset['name']):
            print(f"  deleting stale {asset['name']}")
            _gh('release', 'delete-asset', DATA_RELEASE, asset['name'], '--yes')
    digests = {a['name']: a.get('digest') for a in release['assets']}
    for f, sha256 in zip(files, sha256s):
        if digests.get(f.name) != f'sha256:{sha256}':
            raise RuntimeError(f"GitHub's SHA-256 of {f.name} ({digests.get(f.name)}) does not match the local file.")
    print("Uploaded; GitHub's SHA-256 of every file matches.")


def build(out_dir, only=None):
    """Write the README and the bundles (all, or just ``only``) to ``out_dir``.

    Returns ``(files, sha256s, bundles)``; the README is ``files[0]``.
    """
    data_folder = get_data_folder(load_config())
    out_dir.mkdir(parents=True, exist_ok=True)
    readme = README.read_text(encoding='utf-8').replace('{version}', DATA_VERSION)
    readme_path = out_dir / 'README.md'
    readme_path.write_text(readme, encoding='utf-8')

    bundles = {'baked': [BAKED_BUNDLE], 'raw': [RAW_BUNDLE], None: [BAKED_BUNDLE, RAW_BUNDLE]}[only]
    files = [readme_path]
    for name in bundles:
        files += build_bundle(name, data_folder, out_dir, readme)
    sha256s = [_sha256(f) for f in files]
    for f, sha256 in zip(files, sha256s):
        print(f"{f.name}: {f.stat().st_size / 1e6:,.1f} MB, sha256 {sha256}", flush=True)
    return files, sha256s, bundles


def main():
    parser = argparse.ArgumentParser(
        description=f'Build the data release and upload it to release {DATA_RELEASE} with the GitHub CLI.'
    )
    parser.add_argument('--only', choices=['baked', 'raw'], default=None, help='Build only one bundle.')
    parser.add_argument('--out-dir', type=Path, default=None,
                        help='Write the files to this folder and stop, without uploading.')
    args = parser.parse_args()

    if args.out_dir is not None:
        build(args.out_dir, args.only)
        return
    with tempfile.TemporaryDirectory(prefix='drosophila_model_release_') as tmp:
        files, sha256s, bundles = build(Path(tmp), args.only)
        upload(files, sha256s, bundles, files[0])


if __name__ == '__main__':
    main()
