import model_utilities as mu
import torch
import numpy as np
from matplotlib import pyplot as plt

device = 'cpu'
dtype = torch.float32
dt = 0.01  # in seconds
max_time = 1 # in seconds
time_points = torch.arange(0, max_time, step=dt)
num_time = len(time_points)
synapse_strength_init_log = np.log(0.01)
nonlinearity = torch.relu

# --- define the parameters and get the sparse connectivity matrix ---
synapse_counts = mu.load_sparse_matrix(device=device, dtype=dtype)
diagonal = torch.full((synapse_counts.shape[0],), 0.9, requires_grad=True, dtype=dtype, device=device)
synapse_strength = torch.tensor(synapse_strength_init_log, requires_grad=True, dtype=dtype, device=device)
dynamics_matrix = torch.exp(synapse_strength) * synapse_counts + torch.diag(diagonal)
num_neurons = dynamics_matrix.shape[0]

# --- get a sign wave stimulus ---
sample_x = torch.arange(0, 360, step=5) / 360 * 2 * np.pi
sign_waves = mu.create_sign_wave(sample_x, dt=dt)


# --- run the model forward ---
x = torch.zeros((num_neurons, num_time), dtype=dtype, device=device) + 0.1
for ti in range(num_time):
    x[:, ti] = dynamics_matrix @ nonlinearity(x[:, ti-1])
    print(ti, '/', num_time)

a=1
