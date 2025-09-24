from pathlib import Path
import pandas as pd
import torch
import numpy as np

def load_sparse_matrix(device=torch.device('cpu'), dtype=torch.float32):
    # --- load ---
    connectome_folder = Path('/Users/mcreamer/Documents/drosophila_connectome')
    cell_types_df = pd.read_parquet(connectome_folder / 'optic_lobes_cell_types.parquet')
    syn_df = pd.read_parquet(connectome_folder / 'optic_lobes_synapse_counts.parquet')
    num_neurons = len(cell_types_df.index)
    index_dtype = torch.int64

    # --- choose which axis is row/col ---
    # values → torch tensor on the right device and dtype
    values = torch.tensor(syn_df['synapse_counts'].values, dtype=dtype, device=device)

    # indices must be int64
    rows = torch.tensor(syn_df['pre_matrix_ind'].values, dtype=index_dtype, device=device)
    cols = torch.tensor(syn_df['post_matrix_ind'].values, dtype=index_dtype, device=device)

    # shape the indices as (2, nnz)
    indices = torch.stack([rows, cols], dim=0)

    # --- build sparse matrix ---
    # first construct with coo, then coalesce, then convert to CSR for speed
    coo = torch.sparse_coo_tensor(indices, values, size=(num_neurons, num_neurons), device=device)
    coo = coo.coalesce()  # coalesce: sums duplicates and sorts indices (required for reliable CSR conversion)
    csr = coo.to_sparse_csr()  # dtype is preserved

    return csr

def create_sign_wave(sample_x, contrast=1, wavelength=np.pi/6, temporal_frequency=torch.arange(0, 10, step=2), dt=0.01, max_time=1):
    t = torch.arange(0, max_time, step=dt)
    t_mesh, x_mesh = torch.meshgrid(t, sample_x)

    sign_waves = []
    for tf in temporal_frequency:
        sign_waves.append(contrast * torch.sin(2 * np.pi * (x_mesh / wavelength + tf * t_mesh)))

    return sign_waves