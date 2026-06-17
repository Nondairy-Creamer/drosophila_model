#!/bin/bash
set -e

# We intentionally do NOT create a venv -- on RunPod, the system Python ships
# with a torch build matched to the pod's GPU (e.g. Blackwell / sm_120 needs a
# cu128 nightly newer than what the stable cu128 index serves). Running
# against the system Python lets us inherit that working torch.
pip install --upgrade pip

# Third-party deps. torch is listed unpinned in requirements.txt so pip will
# leave any existing install alone (RunPod) and install the default CPU build
# on fresh machines without torch.
pip install -r requirements.txt

# Register this repo as the `drosophila_model` package in editable mode.
# --no-deps avoids pip re-resolving torch off pyproject.toml's version pin
# and accidentally replacing the pod's GPU torch with a CPU build from PyPI.
pip install -e . --no-deps

mkdir -p ../drosophila_model_data/stimulus/baked
mkdir -p ../drosophila_model_data/stimulus/training

# Minimal dataset: only what's needed to train the default optic_lobe model.
# Stimuli are no longer generated on the fly -- training loads pre-baked
# datasets from stimulus/training (see data_handling/create_training_data.py).
# We also pull stimulus/baked (the process_raw_data.py caches: small_drone.pt,
# rpi_drone.pt, all_panoramic.pt) so the on-the-fly generators / create_training_data
# can be re-run without re-downloading the large raw source files.
#
# gdown can fail on individual files when Google Drive hits its per-file daily
# download cap. Don't let that abort the whole setup -- just warn and continue
# so the config copy below still runs. Re-run setup_server.sh later to retry.
#
# Drive IDs (from the source folder listing):
#   connectome_data/baked/  folder  1lkj3q3awKuCOqNnGticnufXqDHMV5f67
#   stimulus/baked/         folder  1VnciFS4X9BXU0NAgxj-obnpQyPxixkwc
#   stimulus/training/      folder  12W17H3OQojSquBDgdfnUv-09TTbZYj0w
#   config.toml                     17JhWo8jaq7zA41peCR7CHXRDkOXFYS6k
#   config_data.toml                1y_zl_ZNtHIQeIW-3gvHdaEiNYBRYRi9k
set +e
gdown --folder "https://drive.google.com/drive/folders/1lkj3q3awKuCOqNnGticnufXqDHMV5f67" -O ../drosophila_model_data/connectome_data/baked
gdown --folder "https://drive.google.com/drive/folders/1VnciFS4X9BXU0NAgxj-obnpQyPxixkwc" -O ../drosophila_model_data/stimulus/baked
gdown --folder "https://drive.google.com/drive/folders/12W17H3OQojSquBDgdfnUv-09TTbZYj0w" -O ../drosophila_model_data/stimulus/training
gdown "17JhWo8jaq7zA41peCR7CHXRDkOXFYS6k" -O ../drosophila_model_data/config.toml
gdown "1y_zl_ZNtHIQeIW-3gvHdaEiNYBRYRi9k" -O ../drosophila_model_data/config_data.toml
set -e

# Place the downloaded user configs at the repo root where load_config() /
# get_data_config() look for them (CWD-relative). config_default.toml and
# config_default_data.toml are shipped inside the package; these are the
# per-machine overlays (gitignored).
if [ -f ../drosophila_model_data/config.toml ]; then
    cp ../drosophila_model_data/config.toml ./config.toml
else
    echo "warning: ../drosophila_model_data/config.toml not found; skipping copy" >&2
fi
if [ -f ../drosophila_model_data/config_data.toml ]; then
    cp ../drosophila_model_data/config_data.toml ./config_data.toml
else
    echo "warning: ../drosophila_model_data/config_data.toml not found; skipping copy" >&2
fi