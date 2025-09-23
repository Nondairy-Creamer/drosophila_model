from pathlib import Path
import numpy as np
import pandas as pd
import torch

# --- load ---
connectome_folder = Path('/Users/mcreamer/Documents/drosophila_connectome')
cell_types_df = pd.read_parquet(connectome_folder / 'optic_lobes_cell_types.parquet')
syn_df = pd.read_parquet(connectome_folder / 'optic_lobes_synapse_counts.parquet')

num_neurons = len(cell_types_df.index)
device = torch.device("cpu")  # or "cuda"

# --- choose which axis is row/col ---
# values → torch tensor on the right device and dtype
values = torch.tensor(syn_df['synapse_counts'].values, device=device)

# indices must be int64
rows = torch.tensor(syn_df['pre_matrix_ind'].values, dtype=torch.int64, device=device)
cols = torch.tensor(syn_df['post_matrix_ind'].values, dtype=torch.int64, device=device)

# shape the indices as (2, nnz)
indices = torch.stack([rows, cols], dim=0)

# --- build sparse matrix ---
# first construct with coo, then coalesce, then convert to CSR for speed
coo = torch.sparse_coo_tensor(indices, values, size=(num_neurons, num_neurons), device=device)
coo = coo.coalesce()  # coalesce: sums duplicates and sorts indices (required for reliable CSR conversion)
csr = coo.to_sparse_csr()  # dtype is preserved

# print out the expected first entry
example_row = indices[0, 0]
example_col = indices[1, 0]
example_val = values[0]

print(csr[example_row, example_col])
print('expected', example_val)

a=1