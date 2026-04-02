#!/bin/bash
set -e

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --force-reinstall -r requirements.txt

mkdir -p ../connectome_data/

gdown --folder "https://drive.google.com/drive/folders/1I5c9Q48eqas6mbjhoTXLPlRk7S4iTR1p" -O ../connectome_data


cp ../connectome_data/config.toml .
