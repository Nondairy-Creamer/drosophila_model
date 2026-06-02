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

mkdir -p ../drosophila_model_data/stimulus/

# Minimal dataset: only what's needed to train the default optic_lobe model.
# gdown can fail on individual files when Google Drive hits its per-file daily
# download cap. Don't let that abort the whole setup -- just warn and continue
# so the config copy below still runs. Re-run setup_server.sh later to retry.
#
# Drive IDs (from the source folder listing):
#   baked/         folder  1lkj3q3awKuCOqNnGticnufXqDHMV5f67
#   all_panoramic.pt       1-y4Lt0Ps7xRCl2JmT_gJWoMVHJH5emMa
#   drone_data.pt          1oh-srfSyASfcpbMQVLpMgvu0q7M1Q0O7
#   config.toml            17JhWo8jaq7zA41peCR7CHXRDkOXFYS6k
set +e
gdown --folder "https://drive.google.com/drive/folders/1lkj3q3awKuCOqNnGticnufXqDHMV5f67" -O ../drosophila_model_data/connectome_data/baked
gdown "1-y4Lt0Ps7xRCl2JmT_gJWoMVHJH5emMa" -O ../drosophila_model_data/stimulus/all_panoramic.pt
gdown "1oh-srfSyASfcpbMQVLpMgvu0q7M1Q0O7" -O ../drosophila_model_data/stimulus/drone_data.pt
gdown "17JhWo8jaq7zA41peCR7CHXRDkOXFYS6k" -O ../drosophila_model_data/config.toml
set -e

# Place the downloaded user config at the repo root where load_config() looks
# for it (CWD-relative). config_default.toml is shipped inside the package;
# this is the per-machine overlay (gitignored).
if [ -f ../drosophila_model_data/config.toml ]; then
    cp ../drosophila_model_data/config.toml ./config.toml
else
    echo "warning: ../drosophila_model_data/config.toml not found; skipping copy" >&2
fi