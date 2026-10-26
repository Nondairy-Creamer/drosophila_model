#!/bin/bash
# Set up a training server (e.g. RunPod) from a fresh clone. Run from the repo
# root:
#
#     bash setup_server.sh [DATA_FOLDER]
#
# DATA_FOLDER (or the $DATA_FOLDER environment variable) is where the data
# bundle is downloaded; it defaults to ../drosophila_model_data, the same
# default as [data].data_folder. The installed config.toml gets
# [data].data_folder set to DATA_FOLDER.
set -e

DATA_FOLDER="${1:-${DATA_FOLDER:-../drosophila_model_data}}"
mkdir -p "$DATA_FOLDER"
DATA_FOLDER="$(cd "$DATA_FOLDER" && pwd)"
echo "Data folder: $DATA_FOLDER"

# We intentionally do NOT create a venv -- on RunPod, the system Python ships
# with a torch build matched to the pod's GPU (e.g. Blackwell / sm_120 needs a
# cu128 nightly newer than what the stable cu128 index serves). Running
# against the system Python lets us inherit that working torch.
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

mkdir -p "$DATA_FOLDER/connectome_data/baked"
mkdir -p "$DATA_FOLDER/stimulus/baked"
mkdir -p "$DATA_FOLDER/stimulus/training"

# Minimal dataset: the baked connectome, the stimulus/baked caches written by
# process_raw_data.py (all_panoramic.pt, flyview.pt) so create_training_data
# can bake training sets without the large raw source files, whatever baked
# training sets were uploaded to
# stimulus/training, and the per-machine config.toml / config_data.toml.
# Training sets that are still missing are baked at the end of this script.
#
# gdown can fail on individual files when Google Drive hits its per-file daily
# download cap. Don't let that abort the whole setup -- just warn and continue
# so the config install below still runs. Re-run setup_server.sh later to retry
# (existing ./config.toml / ./config_data.toml are left untouched).
#
# Drive IDs (from the source folder listing):
#   connectome_data/baked/  folder  1lkj3q3awKuCOqNnGticnufXqDHMV5f67
#   stimulus/baked/         folder  1VnciFS4X9BXU0NAgxj-obnpQyPxixkwc
#   stimulus/training/      folder  12W17H3OQojSquBDgdfnUv-09TTbZYj0w
#   config.toml                     17JhWo8jaq7zA41peCR7CHXRDkOXFYS6k
#   config_data.toml                1y_zl_ZNtHIQeIW-3gvHdaEiNYBRYRi9k
set +e
gdown --folder "https://drive.google.com/drive/folders/1lkj3q3awKuCOqNnGticnufXqDHMV5f67" -O "$DATA_FOLDER/connectome_data/baked"
gdown --folder "https://drive.google.com/drive/folders/1VnciFS4X9BXU0NAgxj-obnpQyPxixkwc" -O "$DATA_FOLDER/stimulus/baked"
gdown --folder "https://drive.google.com/drive/folders/12W17H3OQojSquBDgdfnUv-09TTbZYj0w" -O "$DATA_FOLDER/stimulus/training"
gdown "17JhWo8jaq7zA41peCR7CHXRDkOXFYS6k" -O "$DATA_FOLDER/config.toml"
gdown "1y_zl_ZNtHIQeIW-3gvHdaEiNYBRYRi9k" -O "$DATA_FOLDER/config_data.toml"
set -e

# Install the downloaded per-machine overlays at the repo root, where
# load_config() looks for them (CWD-relative); config_default.toml and
# config_default_data.toml ship inside the package. Existing files are kept.
# The installed config.toml gets [data].data_folder = $DATA_FOLDER.
if [ -f ./config.toml ]; then
    echo "./config.toml already exists; leaving it (check its [data].data_folder)."
elif [ -f "$DATA_FOLDER/config.toml" ]; then
    python - "$DATA_FOLDER/config.toml" "$DATA_FOLDER" <<'PYEOF'
import re
import sys
import tomllib

import tomli_w

src, folder = sys.argv[1], sys.argv[2]
with open(src, encoding='utf-8') as f:
    original = f.read()
# Rewrite only the data_folder line so the file's comments survive; when that
# line is missing (or the edit does not parse), re-serialise the whole file.
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
with open('config.toml', 'w', encoding='utf-8') as f:
    f.write(text)
PYEOF
    echo "Installed ./config.toml with [data].data_folder = $DATA_FOLDER"
else
    echo "warning: $DATA_FOLDER/config.toml not found; create ./config.toml with [data].data_folder = \"$DATA_FOLDER\"" >&2
fi
if [ -f ./config_data.toml ]; then
    echo "./config_data.toml already exists; leaving it."
elif [ -f "$DATA_FOLDER/config_data.toml" ]; then
    cp "$DATA_FOLDER/config_data.toml" ./config_data.toml
else
    echo "warning: $DATA_FOLDER/config_data.toml not found; skipping copy" >&2
fi

# Training reads pre-baked datasets only (stimulus/training/<type>_train.pt).
# Bake any that are missing for the configured [stimulus].stimulus_type.
python - <<'PYEOF'
import subprocess
import sys

from drosophila_model.config import get_data_folder, load_config

config = load_config()
training_dir = get_data_folder(config) / 'stimulus' / 'training'
stimulus_type = config['stimulus']['stimulus_type']
for name in [stimulus_type] if isinstance(stimulus_type, str) else stimulus_type:
    if (training_dir / f'{name}_train.pt').exists():
        print(f'{name}: training data present')
        continue
    print(f'{name}: baking training data ...')
    subprocess.run([sys.executable, '-m', 'drosophila_model.data_handling.create_training_data', name],
                   check=True)
PYEOF

echo "Setup complete. Train with: python -m drosophila_model.training.run_model"
