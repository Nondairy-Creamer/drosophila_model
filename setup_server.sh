#!/bin/bash
set -e

# python3 -m venv .venv
# source .venv/bin/activate
pip install --upgrade pip

# Third-party deps. requirements.txt pins the CUDA 12.8 torch wheel index so
# we get GPU-enabled torch/torchvision/torchaudio (PyPI's default torch is
# CPU-only).
pip install -r requirements.txt

# Register this repo as the `drosophila_model` package in editable mode.
# --no-deps skips re-resolving the deps from pyproject.toml so the cu128 torch
# we just installed isn't clobbered by a CPU build from PyPI.
pip install -e . --no-deps

mkdir -p ../connectome_data/

gdown --folder "https://drive.google.com/drive/folders/1I5c9Q48eqas6mbjhoTXLPlRk7S4iTR1p" -O ../connectome_data


cp ../connectome_data/config.toml .
