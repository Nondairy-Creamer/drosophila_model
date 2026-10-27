#!/bin/bash
# Set up the code base from a fresh clone: install the dependencies, download
# the raw data from Zenodo and bake it into the files the code reads. Run from
# the repo root, inside the Python environment you want to use (on Windows,
# from Git Bash):
#
#     bash setup.sh [DATA_FOLDER [DATA_ZIP]]
#
# DATA_FOLDER (or the $DATA_FOLDER environment variable) is where the data
# go. Without it, the [data].data_folder of an existing config.toml is used,
# else ../drosophila_model_data, the packaged default. config.toml is created
# or updated so [data].data_folder points there.
#
# DATA_ZIP (or $DATA_ZIP) is a local copy of drosophila_model_data_v1.zip to
# use instead of downloading it.
#
# Re-running is safe: data already in place are not downloaded or baked again.
set -e

DATA_FOLDER="${1:-${DATA_FOLDER:-}}"
DATA_ZIP="${2:-${DATA_ZIP:-}}"

# We intentionally do NOT create a venv -- on RunPod, the system Python ships
# with a torch build matched to the pod's GPU (e.g. Blackwell / sm_120 needs a
# cu128 nightly newer than what the stable cu128 index serves). Running
# against the active Python lets us inherit that working torch.
#
# pip is always run as `python -m pip`: on Windows (Git Bash) pip.exe refuses
# to upgrade itself, and this also guarantees pip matches `python` below.
python -m pip install --upgrade pip

# Third-party deps. torch is listed unpinned in requirements.txt so pip will
# leave any existing install alone (RunPod) and install the default CPU build
# on fresh machines without torch.
python -m pip install -r requirements.txt

# Register this repo as the `drosophila_model` package in editable mode.
# --no-deps avoids pip re-resolving torch off pyproject.toml's version pin
# and accidentally replacing the pod's GPU torch with a CPU build from PyPI.
python -m pip install -e . --no-deps

# Point config.toml's [data].data_folder at the data folder (creating the
# file, and the folder, if needed). Only the data_folder line is rewritten, so
# the rest of an existing config.toml and its comments are kept. The folder is
# passed as an argument, which Git Bash converts to a Windows path.
python - "$DATA_FOLDER" <<'PYEOF'
import re
import sys
import tomllib
from pathlib import Path

import tomli_w

path = Path('config.toml')
original = path.read_text(encoding='utf-8') if path.exists() else ''
current = tomllib.loads(original).get('data', {}).get('data_folder')
folder = sys.argv[1] or current or '../drosophila_model_data'
folder = Path(folder).expanduser().resolve()
folder.mkdir(parents=True, exist_ok=True)
folder = folder.as_posix()
if current == folder:
    print(f"config.toml: [data].data_folder = {folder}")
    sys.exit()

line = tomli_w.dumps({'data_folder': folder}).strip()
text, n = re.subn(r'(?m)^data_folder\s*=.*$', lambda m: line, original)
try:
    ok = n == 1 and tomllib.loads(text).get('data', {}).get('data_folder') == folder
except tomllib.TOMLDecodeError:
    ok = False
if not ok:
    config = tomllib.loads(original)
    config.setdefault('data', {})['data_folder'] = folder
    text = tomli_w.dumps(config)
path.write_text(text, encoding='utf-8')
print(f"config.toml: set [data].data_folder = {folder}")
PYEOF

# Download the raw data and bake the connectome tables, the panorama cache and
# the training sets for [stimulus].stimulus_type.
if [ -n "$DATA_ZIP" ]; then
    python -m drosophila_model.data_handling.prepare_data --zip "$DATA_ZIP"
else
    python -m drosophila_model.data_handling.prepare_data
fi

echo "Setup complete. Train with: python -m drosophila_model.training.run_model"
