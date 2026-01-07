#!/bin/bash
set -e

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

mkdir -p ../connectome_data/

gdown --folder "https://drive.google.com/drive/folders/1I5c9Q48eqas6mbjhoTXLPlRk7S4iTR1p" -O ../connectome_data

# gdown 11fa8BvQWHKF3dUVaXBpz2-Lm41flELGZ -O ../connectome_data/column_assignment.csv
# gdown 15GdWFPG94Sm0EIrShthk6myyU3DniWzR -O ../connectome_data/config.toml
# gdown 1ZzVnAXyxHnQboFE2ZBbITCJF3vNhtqVS -O ../connectome_data/connections_princeton.csv
# gdown 1wVViFXeBseY_SFDC2JE6AnWUgO7j4ffN -O ../connectome_data/fly_view3.npy
# gdown 1Q3FByY1GIph4vC7P50dLpSBNRcc9kAz2 -O ../connectome_data/optic_lobes_cell_info.parquet
# gdown 1N-gnap7OpOVYSDDH9DGCWqQByUQU8RsQ -O ../connectome_data/optic_lobes_column_info.parquet
# gdown 12eDWqar-cw919Ct1eEghc_vh5lQOUf24 -O ../connectome_data/optic_lobes_synapse_counts.parquet
# gdown 1UmkWWGjauyT8KFhWEq5GPXZkZCAluRmN -O ../connectome_data/visual_neuron_types.csv

cp ../connectome_data/config.toml .
