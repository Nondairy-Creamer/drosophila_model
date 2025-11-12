import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import imageio.v3 as iio


def fig_to_rgb(fig):
    fig.canvas.draw()
    buf, (w, h) = fig.canvas.print_to_buffer()
    return np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[..., :3].copy()


def make_scatter_plot_factory(pred_plot, cell_x, cell_z):
    """
    Returns a factory callable that, given a direction index, initializes a scatter plot and
    provides an update function for per-frame rendering.
    """
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    def init_plot_for_direction(d, di):
        fig, ax = plt.subplots(figsize=(6,4), dpi=120)
        chosen_activity = pred_plot[di, :, :]

        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        sc = ax.scatter(cell_x, cell_z, c=chosen_activity[0, :], cmap="viridis", norm=norm)
        cbar = fig.colorbar(sc, ax=ax)
        ax.set_xlabel('location in x, posterior <-> anterior')
        ax.set_ylabel('location in z, ventral <-> dorsal')
        ax.set_title(d)
        num_frames = chosen_activity.shape[0]

        def update(frame_t):
            sc.set_array(chosen_activity[frame_t, :])
            cbar.update_normal(sc)

        return fig, update, num_frames

    return init_plot_for_direction

def make_scatter_diamond_sequence_factory(pair_resp, cell_x, cell_z, offset=2.6, label=False, scale=1.0, size_scale=1.0, order=('right','left','up','down')):
    """
    New-format only, single input dict (same format as make_scatter_diamond_factory).
    Returns a factory callable that, for any direction index, initializes a single-axes
    diamond layout and plays the four direction blocks back-to-back in the specified order,
    inserting 25 zero frames between them AND after the last block.
      - pair_resp: dict with keys ['right','left','up','down'], each with:
          - 'stimulus': [time, columns]
          - 'responses': dict of four entries (insertion order), each [time, columns]
      - order: tuple specifying playback order of direction keys
    The returned init callable conforms to make_movie: init(d, di) -> (fig, update, num_frames).
    """
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    positions = {
        'up': (0.0, +offset),
        'right': (+offset, 0.0),
        'left': (-offset, 0.0),
        'down': (0.0, -offset),
    }

    def init_plot_for_direction(d, di):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
        try:
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax.set_position([0, 0, 1, 1])
        except Exception:
            pass

        # Compute tight limits once (independent of which dict is playing)
        coords = []
        for dir_name in ['up', 'right', 'left', 'down']:
            dx, dz = positions[dir_name]
            coords.append((scale * cell_x + dx, scale * cell_z + dz))
        # include stimulus footprint
        coords.append(((0.5 * scale) * cell_x, (0.5 * scale) * cell_z))
        x_all = np.concatenate([xy[0] for xy in coords])
        y_all = np.concatenate([xy[1] for xy in coords])
        x_min, x_max = np.min(x_all), np.max(x_all)
        y_min, y_max = np.min(y_all), np.max(y_all)
        x_span = max(1e-6, x_max - x_min)
        y_span = max(1e-6, y_max - y_min)
        span = max(x_span, y_span)
        margin_frac = 0.02
        half = 0.5 * span * (1.0 + margin_frac)
        x_center = 0.5 * (x_min + x_max)
        y_center = 0.5 * (y_min + y_max)
        ax.set_xlim(x_center - half, x_center + half)
        ax.set_ylim(y_center - half, y_center + half)
        ax.set_aspect('equal', 'box')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_axis_off()

        # Marker sizes
        base_ms = plt.rcParams.get("lines.markersize", 6.0)
        base_s = float(base_ms) ** 2
        s_dir = max(1.0, base_s * size_scale)
        s_stim = max(1.0, s_dir * 0.2)  # keep consistent with diamond factory default

        # Create artists for four directions
        dir_order = ['up', 'right', 'left', 'down']
        dir_scatters = {}
        for dir_name in dir_order:
            dx, dz = positions[dir_name]
            sc = ax.scatter(
                scale * cell_x + dx,
                scale * cell_z + dz,
                c=np.zeros_like(cell_x),
                cmap="viridis",
                norm=norm,
                s=s_dir
            )
            if label:
                ax.text(dx, dz + 0.9 * scale, dir_name, ha='center', va='bottom', fontsize=10, color='black')
            dir_scatters[dir_name] = sc

        # Stimulus artist (always present but may be zeroed)
        stim_scatter = ax.scatter(
            (0.5 * scale) * cell_x,
            (0.5 * scale) * cell_z,
            c=np.zeros_like(cell_x),
            cmap="viridis",
            norm=norm,
            s=s_stim
        )
        if label:
            ax.text(0.0, 0.9 * 0.5 * scale, 'stimulus', ha='center', va='bottom', fontsize=10, color='black')

        num_cols = cell_x.shape[0]

        # Helper to extract arrays for a single direction block
        def extract_block_arrays(block):
            values_in_order = list(block.get('responses', {}).values()) if isinstance(block.get('responses', {}), dict) else []
            mapping_dirs = ['right', 'left', 'up', 'down']
            # Determine frame count for this block
            L_candidates = []
            stim_arr = block.get('stimulus', None)
            if isinstance(stim_arr, np.ndarray):
                L_candidates.append(stim_arr.shape[0])
            for arr in values_in_order:
                if isinstance(arr, np.ndarray):
                    L_candidates.append(arr.shape[0])
            L = min(L_candidates) if L_candidates else 0
            # Build directional arrays, clipped to L
            dir_arrays = {}
            for pos, dir_name in enumerate(mapping_dirs):
                if pos < len(values_in_order) and isinstance(values_in_order[pos], np.ndarray):
                    dir_arrays[dir_name] = values_in_order[pos][:L, :]
                else:
                    dir_arrays[dir_name] = np.zeros((L, num_cols), dtype=float)
            if isinstance(stim_arr, np.ndarray):
                stim_array = stim_arr[:L, :]
            else:
                stim_array = np.zeros((L, num_cols), dtype=float)
            return L, dir_arrays, stim_array

        # Build segment schedule from the single input dict ordered by `order`
        segments = []
        segment_data = []
        total_frames = 0
        blanks = 25
        for i, dir_key in enumerate(order):
            block = pair_resp.get(dir_key, {})
            L, dir_arrays, stim_array = extract_block_arrays(block)
            segments.append(('data', i, L))
            segment_data.append((dir_arrays, stim_array))
            total_frames += L
            # Always add a blank segment after each block, including the last
            segments.append(('blank', None, blanks))
            total_frames += blanks

        def update(frame_t):
            # Determine which segment and local frame
            t = frame_t
            for seg_type, idx, length in segments:
                if t < length:
                    if seg_type == 'blank':
                        zero_vec = np.zeros(num_cols, dtype=float)
                        for sc in dir_scatters.values():
                            sc.set_array(zero_vec)
                        stim_scatter.set_array(zero_vec)
                    else:  # data
                        dir_arrays, stim_array = segment_data[idx]
                        for dir_name, sc in dir_scatters.items():
                            sc.set_array(dir_arrays[dir_name][t, :])
                        stim_scatter.set_array(stim_array[t, :])
                    return
                t -= length
            # Safety: if out of range, show blank
            zero_vec = np.zeros(num_cols, dtype=float)
            for sc in dir_scatters.values():
                sc.set_array(zero_vec)
            stim_scatter.set_array(zero_vec)

        return fig, update, total_frames

    return init_plot_for_direction


def make_quiver_plot_factory(x_flow, z_flow, cell_x, cell_z):
    """
    Returns a factory callable that, given a direction index, initializes a quiver plot (+X from pred_plot_0, +Y from pred_plot_1)
    and provides an update function for per-frame rendering. Arrow color encodes magnitude.
    """
    magnitude = np.hypot(x_flow, z_flow)
    max_mag = np.percentile(magnitude, 99) if np.any(magnitude) else 1.0
    norm = mcolors.Normalize(vmin=0.0, vmax=max_mag)

    def init_plot_for_direction(d, di):
        fig, ax = plt.subplots(figsize=(6,4), dpi=120)
        chosen_u = -x_flow[di, :, :]
        chosen_v = z_flow[di, :, :]

        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        q = ax.quiver(
            cell_x, cell_z,
            chosen_u[0, :], chosen_v[0, :],
            np.hypot(chosen_u[0, :], chosen_v[0, :]),
            cmap="viridis",
            norm=norm,
            angles='xy',
            scale_units='xy',
            scale=1.0,
        )
        cbar = fig.colorbar(q, ax=ax)
        ax.set_xlabel('location in x, posterior <-> anterior')
        ax.set_ylabel('location in z, ventral <-> dorsal')
        ax.set_title(d)
        num_frames = chosen_u.shape[0]

        def update(frame_t):
            q.set_UVC(
                chosen_u[frame_t, :],
                chosen_v[frame_t, :],
                np.hypot(chosen_u[frame_t, :], chosen_v[frame_t, :])
            )
            cbar.update_normal(q)

        return fig, update, num_frames

    return init_plot_for_direction


def make_movie(init_plot_for_direction, directions, fig_folder, frame_rate, save_name=''):
    """
    Generic movie maker. Accepts an initializer factory for the chosen figure type.
      - init_plot_for_direction(d, di) -> (fig, update(frame_t), num_frames)
    """
    for di, d in enumerate(directions):
        frames = []
        fig, update, num_frames = init_plot_for_direction(d, di)

        for t in range(num_frames):
            update(t)
            frames.append(fig_to_rgb(fig))

        plt.close(fig)

        stack = np.stack(frames, axis=0)
        iio.imwrite(
            fig_folder / f"{save_name}_{d}.mp4",
            stack,
            fps=frame_rate,
            codec="libx264",
            pixelformat="yuv420p",
        )

 
def make_scatter_grid_factory(pair_resp, cell_x, cell_z):
    """
    Returns a factory callable that, given a direction index, initializes a 2x2 grid
    of scatter plots for the four pair responses present in `pair_resp`.
      - Each subplot shows one pair with its title set to the pair key (pkey).
      - Colors are normalized to [0, 1] similar to make_scatter_plot_factory.
    Expected `pair_resp` keys: 'right_cells', 'left_cells', 'up_cells', 'down_cells'
    and values shaped as [num_directions, num_time, num_columns].
    """
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    preferred_order = ['right_cells', 'left_cells', 'up_cells', 'down_cells']
    keys = [k for k in preferred_order if k in pair_resp]
    # Append any extra keys deterministically to avoid surprises
    keys += [k for k in pair_resp.keys() if k not in keys]

    def init_plot_for_direction(d, di):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=120)
        axes = axes.ravel()

        scatters = []
        colorbars = []
        activities = []
        num_frames = None

        for ax, key in zip(axes, keys):
            chosen_activity = pair_resp[key][di, :, :]
            if num_frames is None:
                num_frames = chosen_activity.shape[0]

            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
            sc = ax.scatter(cell_x, cell_z, c=chosen_activity[0, :], cmap="viridis", norm=norm)
            cbar = fig.colorbar(sc, ax=ax)
            ax.set_xlabel('location in x, posterior <-> anterior')
            ax.set_ylabel('location in z, ventral <-> dorsal')
            ax.set_title(key)

            scatters.append(sc)
            colorbars.append(cbar)
            activities.append(chosen_activity)

        # If fewer than 4 keys, hide remaining axes
        for ax in axes[len(keys):]:
            ax.axis('off')

        def update(frame_t):
            for sc, cbar, chosen_activity in zip(scatters, colorbars, activities):
                sc.set_array(chosen_activity[frame_t, :])
                cbar.update_normal(sc)

        return fig, update, num_frames

    return init_plot_for_direction

def make_scatter_grid_3x3_factory(pair_resp, cell_x, cell_z):
    """
    Returns a factory callable that, given a direction index, initializes a 3x3 grid
    of scatter plots where only four positions are populated:
      - 'up_cells' at (0, 1)
      - 'right_cells' at (1, 2)
      - 'left_cells' at (1, 0)
      - 'down_cells' at (2, 1)
    All other subplots are empty. Axes, ticks, labels, and colorbars are removed; only
    the scatter data and a title (the dict key) are shown.
    """
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    position_by_key = {
        'up_cells': (0, 1),
        'right_cells': (1, 2),
        'left_cells': (1, 0),
        'down_cells': (2, 1),
    }
    preferred_order = ['up_cells', 'right_cells', 'left_cells', 'down_cells']
    keys = [k for k in preferred_order if k in pair_resp]

    def init_plot_for_direction(d, di):
        # Use manual axis placement so we can enlarge each by ~30% relative to a 3x3 cell
        fig = plt.figure(figsize=(6, 6), dpi=120)

        scatters = []
        activities = []
        num_frames = None

        # Compute centers for a 3x3 grid in figure coordinates
        cell_centers = {
            0: 1.0/6.0,   # ~0.1667
            1: 3.0/6.0,   # 0.5
            2: 5.0/6.0,   # ~0.8333
        }
        # Default cell size in figure coords is 1/3; enlarge by 30%
        base_size = 1.0 / 3.0
        side = min(0.45, base_size * 1.3)  # clamp to avoid going out of bounds

        for key in keys:
            r, c = position_by_key[key]
            cx = cell_centers[c]
            cy = 1.0 - cell_centers[r]  # invert y for figure coordinates (origin at bottom)
            left = max(0.0, cx - side / 2.0)
            bottom = max(0.0, cy - side / 2.0)
            # Ensure the box stays within [0,1]
            if left + side > 1.0:
                left = 1.0 - side
            if bottom + side > 1.0:
                bottom = 1.0 - side
            ax = fig.add_axes([left, bottom, side, side])

            chosen_activity = pair_resp[key][di, :, :]
            if num_frames is None:
                num_frames = chosen_activity.shape[0]

            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
            sc = ax.scatter(cell_x, cell_z, c=chosen_activity[0, :], cmap="viridis", norm=norm)
            # Remove decorations and keep title
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_aspect('equal', 'box')
            ax.set_title(key)

            scatters.append(sc)
            activities.append(chosen_activity)

        num_frames = num_frames if num_frames is not None else 0

        def update(frame_t):
            for sc, chosen_activity in zip(scatters, activities):
                sc.set_array(chosen_activity[frame_t, :])

        return fig, update, num_frames

    return init_plot_for_direction

def make_scatter_diamond_factory(pair_resp, cell_x, cell_z, offset=2.6, label=False, scale=1.0, size_scale=1.0):
    """
    New-format only.
    Returns a factory callable that, given a direction index, initializes a single-axes
    scatter where the four datasets are arranged in a diamond configuration:
      - Up at (0, +offset), Right at (+offset, 0), Left at (-offset, 0), Down at (0, -offset)
    Input structure:
      pair_resp: dict with keys ['right','left','up','down'].
        For each direction key k: pair_resp[k] is a dict with:
          - 'stimulus': array [time, columns], plotted at the center (half spatial scale, smaller markers)
          - 'responses': dict with four entries, each array [time, columns].
            Entries are used in insertion order: first->right, second->left, third->up, fourth->down.
    Axes, ticks, labels, and colorbars are removed; optional text labels display the element name.
    The `scale` parameter uniformly scales each scatter cloud about its own center.
    """
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    positions = {
        'up': (0.0, +offset),
        'right': (+offset, 0.0),
        'left': (-offset, 0.0),
        'down': (0.0, -offset),
    }

    def init_plot_for_direction(d, di):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
        # Remove outer figure padding; make axes fill the figure
        try:
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax.set_position([0, 0, 1, 1])
        except Exception:
            pass
        scatters = []
        activities = []
        num_frames = None
        # Precompute scaled/shifted coordinates per direction to set tight limits (independent of data values)
        coords = []
        for dx, dz in positions.values():
            coords.append((scale * cell_x + dx, scale * cell_z + dz))
        coords.append(((0.5 * scale) * cell_x, (0.5 * scale) * cell_z))

        if coords:
            x_all = np.concatenate([xy[0] for xy in coords])
            y_all = np.concatenate([xy[1] for xy in coords])
            x_min, x_max = np.min(x_all), np.max(x_all)
            y_min, y_max = np.min(y_all), np.max(y_all)
            x_span = max(1e-6, x_max - x_min)
            y_span = max(1e-6, y_max - y_min)
            span = max(x_span, y_span)
            margin_frac = 0.02  # very small whitespace
            half = 0.5 * span * (1.0 + margin_frac)
            x_center = 0.5 * (x_min + x_max)
            y_center = 0.5 * (y_min + y_max)
            ax.set_xlim(x_center - half, x_center + half)
            ax.set_ylim(y_center - half, y_center + half)
        else:
            # Fallback limits if nothing present
            xy_margin = 0.2
            max_extent = offset + (scale * 1.0) + xy_margin
            ax.set_xlim(-max_extent, +max_extent)
            ax.set_ylim(-max_extent, +max_extent)
        ax.set_aspect('equal', 'box')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_axis_off()

        # Determine marker size relative to Matplotlib defaults
        base_ms = plt.rcParams.get("lines.markersize", 6.0)
        base_s = float(base_ms) ** 2  # scatter uses area in points^2
        s_val = max(1.0, base_s * size_scale)

        # New format: pair_resp[d] contains 'stimulus' [T,C] and 'responses' dict of 4 entries [T,C]
        block = pair_resp.get(d, {})
        # Build directional arrays in requested order: right, left, up, down
        resp_arrays = []
        if 'responses' in block and isinstance(block['responses'], dict):
            values_in_order = list(block['responses'].values())
            mapping_dirs = ['right', 'left', 'up', 'down']
            for idx in range(4):
                arr = values_in_order[idx] if idx < len(values_in_order) else None
                resp_arrays.append((mapping_dirs[idx], arr))
        else:
            resp_arrays = [('right', None), ('left', None), ('up', None), ('down', None)]

        # Determine number of frames
        lengths = []
        for _, arr in resp_arrays:
            if arr is not None:
                lengths.append(arr.shape[0])
        if 'stimulus' in block:
            lengths.append(block['stimulus'].shape[0])
        num_frames = min(lengths) if lengths else 0

        # Create direction scatters
        for dir_name, arr in resp_arrays:
            dx, dz = positions[dir_name]
            initial = arr[0, :] if (arr is not None and num_frames > 0) else np.zeros_like(cell_x)
            sc = ax.scatter(
                scale * cell_x + dx,
                scale * cell_z + dz,
                c=initial,
                cmap="viridis",
                norm=norm,
                s=s_val
            )
            if label:
                ax.text(dx, dz + 0.9 * scale, dir_name, ha='center', va='bottom', fontsize=10, color='black')
            scatters.append(sc)
            activities.append(arr if arr is not None else np.zeros((max(1, num_frames), cell_x.shape[0])))

        # Center stimulus
        stim_arr = block.get('stimulus', None)
        stim_initial = stim_arr[0, :] if (stim_arr is not None and num_frames > 0) else np.zeros_like(cell_x)
        sc = ax.scatter(
            (0.5 * scale) * cell_x,
            (0.5 * scale) * cell_z,
            c=stim_initial,
            cmap="viridis",
            norm=norm,
            s=max(1.0, s_val * 0.2)
        )
        if label:
            ax.text(0.0, 0.9 * 0.5 * scale, 'stimulus', ha='center', va='bottom', fontsize=10, color='black')
        scatters.append(sc)
        activities.append(stim_arr if stim_arr is not None else np.zeros((max(1, num_frames), cell_x.shape[0])))

        num_frames = num_frames if num_frames is not None else 0

        def update(frame_t):
            for sc, chosen_activity in zip(scatters, activities):
                # Safeguard for zero-length arrays
                if chosen_activity.shape[0] == 0:
                    sc.set_array(np.zeros(cell_x.shape[0], dtype=float))
                else:
                    t = min(frame_t, chosen_activity.shape[0] - 1)
                    sc.set_array(chosen_activity[t, :])

        return fig, update, num_frames

    return init_plot_for_direction

