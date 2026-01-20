import numpy as np
import pandas as pd
import torch


def compute_average_connections(model):
    """
    Compute the average synapse count between each pair of cell types.
    
    Returns:
        ave_con: numpy array of shape (num_cell_types, num_cell_types)
                 where ave_con[post_type, pre_type] is the average synapse count
                 from pre_type to post_type
    """
    indices = model.synapse_counts.indices().cpu().numpy()  # shape (2, nnz)
    values = model.synapse_counts.values().cpu().numpy()    # shape (nnz,)

    # Get cell type index for each pre and post neuron in the synapse list
    pre_types = model.neuron_to_type_index[indices[0]].cpu().numpy()
    post_types = model.neuron_to_type_index[indices[1]].cpu().numpy()

    # Use pandas groupby to compute mean for each (pre_type, post_type) pair
    synapse_df = pd.DataFrame({
        'pre_type': pre_types,
        'post_type': post_types,
        'count': values
    })
    ave_con_df = synapse_df.groupby(['post_type', 'pre_type'])['count'].mean()

    # Fill the ave_con matrix
    ave_con = np.zeros((model.num_cell_types, model.num_cell_types))
    for (post_type, pre_type), mean_val in ave_con_df.items():
        ave_con[int(post_type), int(pre_type)] = mean_val
    
    return ave_con


def path_from_a_to_b(model, start_names, end_names, top_n=20, max_iterations=20, n_extra_steps=3, verbose=True):
    """
    Find the neurons needed to connect start neurons to end neurons by iteratively
    following the top N outgoing connections.
    
    Args:
        model: CurrentModel model
        start_names: list of starting cell type names
        end_names: list of ending cell type names
        top_n: number of top connections to follow from each neuron
        max_iterations: maximum number of iterations before giving up
        n_extra_steps: number of extra iterations after finding all end neurons
                       (to capture major outputs of destination neurons)
        verbose: if True, print progress
    
    Returns:
        path_neurons: list of all unique neuron names in the path
    """
    # Compute average connections matrix
    ave_con = compute_average_connections(model)
    
    # Get input strengths for all cell types (for weighting outgoing connections)
    all_input_strengths = model.input_strength_log.exp().detach().cpu().numpy()
    
    # Create name to index mapping
    name_to_idx = {name: i for i, name in enumerate(model.unique_cell_types)}
    
    # Helper function to do one iteration of expansion
    def expand_frontier(current_neurons, all_neurons):
        new_neurons = set()
        for neuron_name in current_neurons:
            if neuron_name not in name_to_idx:
                continue
            cell_index = name_to_idx[neuron_name]
            
            # Get outgoing connections weighted by input strength of receivers
            # ave_con[:, cell_index] gives connections FROM cell_index TO each post_type
            outgoing_weighted = ave_con[:, cell_index] * all_input_strengths
            
            # Get top N connections (by absolute value, in case of negative weights)
            top_indices = np.argsort(np.abs(outgoing_weighted))[::-1][:top_n]
            
            for idx in top_indices:
                if outgoing_weighted[idx] > 0:  # only include if there's a connection
                    target_name = model.unique_cell_types[idx]
                    if target_name not in all_neurons:
                        new_neurons.add(target_name)
        return new_neurons
    
    # Initialize with start neurons
    current_neurons = set(start_names)  # frontier: neurons to explore this iteration
    all_neurons = set(start_names)      # all discovered neurons
    end_names_set = set(end_names)
    
    iteration = 0
    found_all = False
    
    # Phase 1: expand until we find all end neurons
    while not end_names_set.issubset(all_neurons) and iteration < max_iterations:
        iteration += 1
        new_neurons = expand_frontier(current_neurons, all_neurons)
        all_neurons.update(new_neurons)
        
        if verbose:
            found_ends = end_names_set.intersection(all_neurons)
            print(f"Iteration {iteration}: {len(new_neurons)} new neurons, "
                  f"{len(all_neurons)} total, {len(found_ends)}/{len(end_names)} targets found")
        
        current_neurons = new_neurons
        
        if len(new_neurons) == 0:
            if verbose:
                print("No new neurons found, stopping.")
            break
    
    found_all = end_names_set.issubset(all_neurons)
    
    if verbose:
        if found_all:
            print(f"\nFound all {len(end_names)} target neurons in {iteration} iterations")
        else:
            missing = end_names_set - all_neurons
            print(f"\nCould not find {len(missing)} target neurons: {missing}")
    
    # Phase 2: do extra steps to capture outputs of destination neurons
    if n_extra_steps > 0 and verbose:
        print(f"\nDoing {n_extra_steps} extra steps to capture outputs...")
    
    for extra in range(n_extra_steps):
        new_neurons = expand_frontier(current_neurons, all_neurons)
        all_neurons.update(new_neurons)
        
        if verbose:
            print(f"Extra step {extra + 1}: {len(new_neurons)} new neurons, {len(all_neurons)} total")
        
        current_neurons = new_neurons
        
        if len(new_neurons) == 0:
            if verbose:
                print("No new neurons found, stopping extra steps.")
            break
    
    if verbose:
        print(f"Total neurons in path: {len(all_neurons)}")
    
    return sorted(all_neurons)


def rotate_2d(xy, angle):
    # data shape should be (2, n) with n data points
    # first dimension should be x, y
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return rotation_matrix @ xy

def rotate_3d(xyz, angle, axis):
    # data shape should be (3, n) with n data points
    # first dimension should be x, y, z
    assert axis in ('x', 'y', 'z'), f"axis must be 'x', 'y', or 'z', got {axis}"

    rotation_matrix = np.eye(3)
    rotation_matrix[1, 1] = np.cos(angle)
    rotation_matrix[1, 2] = -np.sin(angle)
    rotation_matrix[2, 1] = np.sin(angle)
    rotation_matrix[2, 2] = np.cos(angle)

    if axis == 'x':
        permute_order = (0, 1, 2)
    elif axis == 'y':
        permute_order = (2, 0, 1)
    else:  # axis must be z
        permute_order = (1, 2, 0)

    rotation_matrix = rotation_matrix[permute_order, :][:, permute_order]

    return rotation_matrix @ xyz

def xyz_to_rtp(xyz):
    r = np.sqrt(xyz[0, :]**2 + xyz[1, :]**2 + xyz[2, :]**2)
    theta = np.arccos(xyz[2, :] / r)
    phi = np.arctan2(xyz[1, :], xyz[0, :])

    return np.vstack((r, theta, phi))

def rtp_to_xyz(rtp):
    x = rtp[0, :] * np.sin(rtp[1, :]) * np.cos(rtp[2, :])
    y = rtp[0, :] * np.sin(rtp[1, :]) * np.sin(rtp[2, :])
    z = rtp[0, :] * np.cos(rtp[1, :])

    return np.vstack((x, y, z))
