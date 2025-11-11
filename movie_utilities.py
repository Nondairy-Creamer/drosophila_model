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

 
