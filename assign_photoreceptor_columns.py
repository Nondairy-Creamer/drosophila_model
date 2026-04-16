import pandas as pd
import numpy as np

cell_info = pd.read_parquet('../connectome_data/optic_lobes_cell_info.parquet')
cell_connections = pd.read_parquet('../connectome_data/optic_lobes_synapse_counts.parquet')

def assign_photoreceptor_columns(cell_type, cell_info, cell_connections):
    """ 
    Modifies cell_info in place to assign columns to the photoreceptors of the given cell type. 
    Finds the column of the downstream cell with the most synapses, and applies that column to the photoreceptor.
    """
    root_ids = cell_info[cell_info['type']==cell_type]['root_id'].values
    outputs = cell_connections[cell_connections['pre_root_id'].isin(root_ids)]
    output_columns = {id: {} for id in root_ids} 
    for row in outputs.itertuples():
        column_index = cell_info[cell_info['root_id'] == row.post_root_id]['column_index'].values[0]
        if np.isnan(column_index):
            continue
        column_index =int(column_index)
        if column_index not in output_columns[row.pre_root_id]:
            output_columns[row.pre_root_id][column_index] = 0
        output_columns[row.pre_root_id][column_index] += row.synapse_counts
    for root_id, columns in output_columns.items():
        max_synapse_counts = 0
        max_column_index = None
        for column_index, synapse_counts in columns.items():
            if synapse_counts > max_synapse_counts:
                max_synapse_counts = synapse_counts
                max_column_index = column_index
        try:
            if max_column_index is not None:
                cell_info.loc[cell_info['root_id'] == root_id, 'column_index'] = max_column_index
                cell_info.loc[cell_info['root_id'] == root_id, 'column_id'] = max_column_index + 1
            else:
                cell_info.loc[cell_info['root_id'] == root_id, 'column_index'] = np.nan
                cell_info.loc[cell_info['root_id'] == root_id, 'column_id'] = np.nan
        except:
            print(type(max_column_index))

assign_photoreceptor_columns('R1-6', cell_info, cell_connections)
assign_photoreceptor_columns('R7', cell_info, cell_connections)
assign_photoreceptor_columns('R8', cell_info, cell_connections)

cell_info.to_parquet('../connectome_data/optic_lobes_cell_info.parquet', index=False)