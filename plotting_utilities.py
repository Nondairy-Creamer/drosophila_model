import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import torch

def plot_readout_target(readout, target, save_folder=Path('figures'), eps=1e-8):
    save_folder = convert_and_create_save_folder(save_folder)

    for readout_neuron in readout:
        plt.figure()
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        direction = ['progressive', 'regressive', 'up', 'down']
        num_direction = len(direction)
        readout_plot = torch.nanmean(readout[readout_neuron], dim=(1, 2)).detach().cpu().numpy()
        readout_plot = readout_plot.reshape((num_direction, readout_plot.shape[0] // num_direction)).T
        target_plot = target[readout_neuron].mean(dim=(1, 2)).detach().cpu().numpy()
        target_plot = target_plot.reshape((num_direction, target_plot.shape[0] // num_direction)).T
        
        # scale target to match prediction for visualization
        # since scale doesnt matter for our loss
        target_plot /= np.sqrt(np.sum(target_plot**2))
        target_plot *= np.sqrt(np.sum(readout_plot**2))

        # plot with 0 in frame
        max_plot = max(np.max(readout_plot), np.max(target_plot))
        plot_fudge = max_plot * 0.1
        for j in range(readout_plot.shape[1]):
            plt.plot(target_plot[:, j], color=colors[j], label=direction[j])
            plt.plot(readout_plot[:, j], color=colors[j], ls='--', label=None)
        plt.title(readout_neuron)
        plt.ylabel('model response')
        plt.xlabel('stimuli')
        plt.ylim(-plot_fudge, max_plot + plot_fudge)
        plt.legend()
        plt.savefig(save_folder / ('pred_vs_target_' + readout_neuron + '.png'))
        plt.close()

def plot_loss(component_losses, save_folder=Path('figures')):
    """Plot total loss and per-component losses side by side.

    component_losses: dict mapping component name -> list of per-epoch values.
    """
    save_folder = convert_and_create_save_folder(save_folder)

    components = {k: np.asarray(v, dtype=float) for k, v in component_losses.items() if v}
    if not components:
        return

    total = np.mean(list(components.values()), axis=0)

    has_multiple = len(components) > 1
    if has_multiple:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        ax2 = None

    ax1.plot(total)
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epochs')
    ax1.set_title('Total loss')
    finite_mask = np.isfinite(total)
    if finite_mask.any():
        ax1.set_ylim((0.9 * np.nanmin(total[finite_mask]), 1.1 * np.nanpercentile(total[finite_mask], 95)))

    if ax2 is not None:
        for name, arr in components.items():
            ax2.plot(arr, label=name)
        ax2.set_ylabel('loss')
        ax2.set_xlabel('epochs')
        ax2.set_title('Loss components')
        ax2.legend()
        all_vals = np.concatenate(list(components.values()))
        finite = np.isfinite(all_vals)
        if finite.any():
            ax2.set_ylim((0.9 * np.nanmin(all_vals[finite]), 1.1 * np.nanpercentile(all_vals[finite], 95)))

    fig.tight_layout()
    fig.savefig(save_folder / 'loss.png')
    plt.close(fig)

def plot_response_over_space(response, x, z, save_folder=Path('figures')):
    save_folder = convert_and_create_save_folder(save_folder)

    for si, s in enumerate(response):
        # just plot the first timepoint
        plt.figure()
        sc = plt.scatter(x, z, c=s)
        plt.colorbar(sc)
        plt.xlabel('location in x, posterior <-> anterior')
        plt.ylabel('location in z, ventral <-> dorsal')
        plt.savefig(save_folder / ('resp_' + str(si) + '.png'))
        plt.close()

def convert_and_create_save_folder(save_folder):
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    return save_folder

def scatter_3d_with_origin_axes(xyz, save_folder=Path('figures')):
    save_folder = convert_and_create_save_folder(save_folder)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = xyz[0, :], xyz[1, :], xyz[2, :]

    # Scatter plot
    sc = ax.scatter(x, y, z, c=z, cmap='viridis', s=20, alpha=0.8)

    # Equal aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Axis arrows starting at origin (0,0,0)
    axis_length = max_range * 1.2
    ax.quiver(0, 0, 0, axis_length, 0, 0, color='k', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, axis_length, 0, color='k', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, axis_length, color='k', arrow_length_ratio=0.1)

    # Labels at tips
    ax.text(axis_length, 0, 0, 'X', color='k', fontsize=12)
    ax.text(0, axis_length, 0, 'Y', color='k', fontsize=12)
    ax.text(0, 0, axis_length, 'Z', color='k', fontsize=12)

    # Labels + colorbar
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Scatter with Origin Axes")

    fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10, label="Z value (color)")
    plt.savefig(save_folder / '3d_column_position.png')
    plt.close()

def xt_plot(stim_time_space, x, optic_flow=None, stimulus=None, save_folder=Path('figures'), run_tag=None, plot_every=4):
    save_folder = convert_and_create_save_folder(save_folder)
    
    valid_cells = ~np.any(np.isnan(stim_time_space), axis=(0, 1))
    valid_stim = stim_time_space[:, :, valid_cells]
    valid_x = x[valid_cells]
    sort_inds = np.argsort(valid_x)
    x_sorted = valid_x[sort_inds]
    stim_sorted = valid_stim[:, :, sort_inds]

    if optic_flow is not None:
        valid_optic_flow = optic_flow[:, :, valid_cells, :]
        optic_flow_sorted = valid_optic_flow[:, :, sort_inds, :]

    if stimulus is not None:
        valid_stimulus = stimulus[:, :, valid_cells]
        stimulus_sorted = valid_stimulus[:, :, sort_inds]

    # Only plot every plot_every stimuli
    plot_indices = np.arange(0, stim_sorted.shape[0], plot_every)
    extent = [x_sorted.min(), x_sorted.max(), 0, valid_stim.shape[1]]

    for pi in plot_indices:
        # Determine subplot layout based on what's provided
        if optic_flow is not None and stimulus is not None:
            fig, axes = plt.subplots(1, 4, figsize=(16, 4), gridspec_kw={'width_ratios': [2, 2, 1, 1]})
            ax_stim, ax_pred, ax_of_x, ax_of_y = axes
        elif optic_flow is not None:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={'width_ratios': [2, 1, 1]})
            ax_pred, ax_of_x, ax_of_y = axes
        elif stimulus is not None:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            ax_stim, ax_pred = axes
        else:
            fig, ax_pred = plt.subplots()
        
        # Plot stimulus if provided
        if stimulus is not None:
            im_stim = ax_stim.imshow(stimulus_sorted[pi, :, :], interpolation='nearest', aspect='auto', extent=extent)
            ax_stim.set_xlabel('phi')
            ax_stim.set_ylabel('time')
            ax_stim.set_title('stimulus')
            fig.colorbar(im_stim, ax=ax_stim)
        
        # No transpose: rows=time (y-axis), cols=phi_sorted high-to-low (x-axis)
        im_pred = ax_pred.imshow(stim_sorted[pi, :, :], interpolation='nearest', aspect='auto', extent=extent)
        if np.all(stim_sorted[pi, :, :] >= 0):
            im_pred.set_clim(vmin=0)
        ax_pred.set_xlabel('phi')
        ax_pred.set_ylabel('time')
        ax_pred.set_title('voltage')
        fig.colorbar(im_pred, ax=ax_pred)

        if optic_flow is not None:
            optic_flow_x = optic_flow_sorted[pi, :, :, 0]
            optic_flow_y = optic_flow_sorted[pi, :, :, 1]

            # horizontal flow
            im_of_x = ax_of_x.imshow(optic_flow_x, interpolation='nearest', aspect='auto', cmap='coolwarm', extent=extent)
            max_abs_x = np.nanmax(np.abs(optic_flow_x))
            if max_abs_x > 1e-10:
                im_of_x.set_clim(vmin=-max_abs_x, vmax=max_abs_x)
            else:
                im_of_x.set_clim(vmin=-1, vmax=1)
            ax_of_x.set_xlabel('phi')
            ax_of_x.set_ylabel('time')
            ax_of_x.set_title('left/right flow')
            fig.colorbar(im_of_x, ax=ax_of_x)

            # vertical flow
            im_of_y = ax_of_y.imshow(optic_flow_y, interpolation='nearest', aspect='auto', cmap='coolwarm', extent=extent)
            max_abs_y = np.nanmax(np.abs(optic_flow_y))
            if max_abs_y > 1e-10:
                im_of_y.set_clim(vmin=-max_abs_y, vmax=max_abs_y)
            else:
                im_of_y.set_clim(vmin=-1, vmax=1)
            ax_of_y.set_xlabel('phi')
            ax_of_y.set_ylabel('time')
            ax_of_y.set_title('up/down flow')
            fig.colorbar(im_of_y, ax=ax_of_y)
        
        plt.tight_layout()
        
        # Build filename with optional run_tag prefix
        filename = 'xt_plot_' + str(pi) + '.png'
        if run_tag is not None:
            filename = str(run_tag) + '_' + filename
        
        plt.savefig(save_folder / filename)
        plt.close()

def xt_plot_flow_prediction(left_eye_input, right_eye_input, left_eye_flow, right_eye_flow, phi, save_folder=Path('figures'), save_tag=None):
    """
    Create a 2x2 plot showing left/right eye stimuli and their predicted optic flow.
    
    Args:
        left_eye_input: (stimuli, time, columns) tensor or numpy array for left eye
        right_eye_input: (stimuli, time, columns) tensor or numpy array for right eye
        left_eye_flow: (stimuli, time, columns, 2) tensor or numpy array for left eye optic flow
        right_eye_flow: (stimuli, time, columns, 2) tensor or numpy array for right eye optic flow
        phi: (columns,) array of phi angles (azimuthal angles) for right eye
        save_folder: Path to save the plot
        save_tag: Optional tag appended to filename (before extension)
    """
    save_folder = convert_and_create_save_folder(save_folder)
    
    # Convert to numpy if tensors
    if torch.is_tensor(left_eye_input):
        left_eye_input_np = left_eye_input.detach().cpu().numpy()
    else:
        left_eye_input_np = left_eye_input
    if torch.is_tensor(right_eye_input):
        right_eye_input_np = right_eye_input.detach().cpu().numpy()
    else:
        right_eye_input_np = right_eye_input
    
    # Convert flow to numpy if tensors
    if torch.is_tensor(left_eye_flow):
        optic_flow_left = left_eye_flow.detach().cpu().numpy()
    else:
        optic_flow_left = left_eye_flow
    if torch.is_tensor(right_eye_flow):
        optic_flow_right = right_eye_flow.detach().cpu().numpy()
    else:
        optic_flow_right = right_eye_flow
    
    # Process each stimulus
    num_stimuli = left_eye_input_np.shape[0]
    
    for si in range(num_stimuli):
        # Get data for this stimulus
        left_stim = left_eye_input_np[si, :, :]  # (time, columns)
        right_stim = right_eye_input_np[si, :, :]  # (time, columns)
        left_flow = optic_flow_left[si, :, :, :]  # (time, columns, 2)
        right_flow = optic_flow_right[si, :, :, :]  # (time, columns, 2)
        
        # Sort phi from positive to negative (reverse order) so phi decreases along x-axis
        # For right eye: use phi directly
        # For left eye: use -phi (mirrored) for x-axis labels
        sort_inds_right = np.argsort(phi)[::-1]  # Reverse sort: high to low
        phi_sorted_right = phi[sort_inds_right]
        
        # For left eye, phi values are mirrored (-phi), so we sort by -phi
        sort_inds_left = np.argsort(-phi)[::-1]  # Reverse sort: high to low
        phi_sorted_left = -phi[sort_inds_left]
        
        # Sort stimuli and flow by phi
        left_stim_sorted = left_stim[:, sort_inds_left]
        right_stim_sorted = right_stim[:, sort_inds_right]
        left_flow_sorted = left_flow[:, sort_inds_left, :]
        right_flow_sorted = right_flow[:, sort_inds_right, :]
        
        # Create 3x2 subplot
        fig, axes = plt.subplots(3, 2, figsize=(12, 15))
        ax_top_left, ax_top_right, ax_mid_left, ax_mid_right, ax_bottom_left, ax_bottom_right = axes.flatten()
        
        # Set extent for plots: [xmin, xmax, bottom, top]
        # For phi: decreasing from positive to negative (high to low)
        # For time: increasing from top to bottom (0 at top, max at bottom)
        # Note: With origin='upper', first row (time 0) is at top, last row (time num_time-1) is at bottom
        # Extent format: [left, right, bottom, top] in data coordinates
        num_time = left_stim.shape[0]
        # Time 0 should be at top, time num_time-1 should be at bottom
        extent_left = [phi_sorted_left.max(), phi_sorted_left.min(), num_time-1, 0]
        extent_right = [phi_sorted_right.max(), phi_sorted_right.min(), num_time-1, 0]
        
        # Top left: Left eye stimulus
        im_left_stim = ax_top_left.imshow(left_stim_sorted, interpolation='nearest', aspect='auto', 
                                         extent=extent_left, origin='upper')
        ax_top_left.set_xlabel('phi')
        ax_top_left.set_ylabel('time')
        ax_top_left.set_title('Left Eye Stimulus')
        plt.colorbar(im_left_stim, ax=ax_top_left)
        
        # Top right: Right eye stimulus
        im_right_stim = ax_top_right.imshow(right_stim_sorted, interpolation='nearest', aspect='auto', 
                                            extent=extent_right, origin='upper')
        ax_top_right.set_xlabel('phi')
        ax_top_right.set_ylabel('time')
        ax_top_right.set_title('Right Eye Stimulus')
        plt.colorbar(im_right_stim, ax=ax_top_right)
        
        # Middle left: Left eye optic flow (horizontal/LR component)
        left_flow_horizontal = left_flow_sorted[:, :, 0]
        max_abs_left_h = np.nanmax(np.abs(left_flow_horizontal))
        im_left_flow_h = ax_mid_left.imshow(left_flow_horizontal, interpolation='nearest', aspect='auto', 
                                             cmap='coolwarm', extent=extent_left, origin='upper')
        if max_abs_left_h > 1e-10:
            im_left_flow_h.set_clim(vmin=-max_abs_left_h, vmax=max_abs_left_h)
        else:
            im_left_flow_h.set_clim(vmin=-1, vmax=1)
        ax_mid_left.set_xlabel('phi')
        ax_mid_left.set_ylabel('time')
        ax_mid_left.set_title('Left Eye Optic Flow (LR)')
        plt.colorbar(im_left_flow_h, ax=ax_mid_left)
        
        # Middle right: Right eye optic flow (horizontal/LR component)
        right_flow_horizontal = right_flow_sorted[:, :, 0]
        max_abs_right_h = np.nanmax(np.abs(right_flow_horizontal))
        im_right_flow_h = ax_mid_right.imshow(right_flow_horizontal, interpolation='nearest', aspect='auto',
                                               cmap='coolwarm', extent=extent_right, origin='upper')
        if max_abs_right_h > 1e-10:
            im_right_flow_h.set_clim(vmin=-max_abs_right_h, vmax=max_abs_right_h)
        else:
            im_right_flow_h.set_clim(vmin=-1, vmax=1)
        ax_mid_right.set_xlabel('phi')
        ax_mid_right.set_ylabel('time')
        ax_mid_right.set_title('Right Eye Optic Flow (LR)')
        plt.colorbar(im_right_flow_h, ax=ax_mid_right)
        
        # Bottom left: Left eye optic flow (vertical/UD component)
        left_flow_vertical = left_flow_sorted[:, :, 1]
        max_abs_left_v = np.nanmax(np.abs(left_flow_vertical))
        im_left_flow_v = ax_bottom_left.imshow(left_flow_vertical, interpolation='nearest', aspect='auto', 
                                             cmap='coolwarm', extent=extent_left, origin='upper')
        if max_abs_left_v > 1e-10:
            im_left_flow_v.set_clim(vmin=-max_abs_left_v, vmax=max_abs_left_v)
        else:
            im_left_flow_v.set_clim(vmin=-1, vmax=1)
        ax_bottom_left.set_xlabel('phi')
        ax_bottom_left.set_ylabel('time')
        ax_bottom_left.set_title('Left Eye Optic Flow (UD)')
        plt.colorbar(im_left_flow_v, ax=ax_bottom_left)
        
        # Bottom right: Right eye optic flow (vertical/UD component)
        right_flow_vertical = right_flow_sorted[:, :, 1]
        max_abs_right_v = np.nanmax(np.abs(right_flow_vertical))
        im_right_flow_v = ax_bottom_right.imshow(right_flow_vertical, interpolation='nearest', aspect='auto',
                                               cmap='coolwarm', extent=extent_right, origin='upper')
        if max_abs_right_v > 1e-10:
            im_right_flow_v.set_clim(vmin=-max_abs_right_v, vmax=max_abs_right_v)
        else:
            im_right_flow_v.set_clim(vmin=-1, vmax=1)
        ax_bottom_right.set_xlabel('phi')
        ax_bottom_right.set_ylabel('time')
        ax_bottom_right.set_title('Right Eye Optic Flow (UD)')
        plt.colorbar(im_right_flow_v, ax=ax_bottom_right)
        
        plt.tight_layout()
        
        # Build filename
        filename = 'xt_plot_flow_prediction_' + str(si)

        filename = filename + '.png'
        if save_tag is not None:
            filename = str(save_tag) + '_' + filename
        
        plt.savefig(save_folder / filename)
        plt.close()