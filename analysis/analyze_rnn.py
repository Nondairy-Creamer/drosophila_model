import torch
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # adds /workspace
import neural_models as nm
import data_utilities as du

model_folder = Path('models/2025/10/17/20_21_18')

config = du.load_config(model_folder / 'config.toml')
model = nm.NeuralRnn.load(model_folder / 'model.pt')
cell_inputs = torch.load(model_folder / 'cell_inputs.pt')

print('hi')
