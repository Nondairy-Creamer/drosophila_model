import matplotlib
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
from matplotlib.colors import Normalize, ListedColormap
import matplotlib.pyplot as plt
import numpy as np

from .utils import hex_vertices


def _use_bundled_ffmpeg_if_needed():
    """Point matplotlib's ffmpeg writer at imageio-ffmpeg's bundled binary
    when no system ffmpeg is on PATH (a system ffmpeg is still preferred)."""
    if not animation.writers.is_available("ffmpeg"):
        import imageio_ffmpeg
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()


def color_hexagons(x, y, values, plot=True, filename=None, colormap=None, fps=20,
                   titles=None, vmin=None, vmax=None, colorbar=True):
    """
    Creates animated visualization of ommatidial responses as colored hexagons.

    Parameters:
        x (np.ndarray): Hexagon center x-coordinates (N,), e.g. from
            ``utils.column_hex_centers``.
        y (np.ndarray): Hexagon center y-coordinates (N,).
        values (np.ndarray): Response values per ommatidium, shape (T, N) where T is number of frames.
            Values should be integer indices into *colormap* (0-255).
        plot (bool): Display plot interactively (default: True)
        filename (str, optional): Output video filename (e.g., "output.mp4")
        colormap (np.ndarray, optional): Matplotlib colormap array (256, 4). If None, uses plasma.
        fps (int): Frames per second for output video (default: 20)
        titles (list[str], optional): Per-frame titles.
        vmin (float, optional): Data value corresponding to colormap index 0.
        vmax (float, optional): Data value corresponding to colormap index 255.
        colorbar (bool): Show a colorbar alongside the hexagons (default: True).

    Returns:
        None: Displays animation and/or saves it to file; the figure is closed afterwards.
    """
    if not plot and filename is None:
        return  # nothing to show or save

    verts = hex_vertices(x, y)
    if colormap is None:
        colormap = cm.plasma(np.linspace(0.0, 1, 256))
    # setup animation
    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.set_aspect('equal')
    polys = []
    for vert in verts:
        closed = np.concatenate((vert, vert[:1, :]), axis=0)
        p_poly = ax.fill(closed[:, 0], closed[:, 1], color='gray')[0]
        polys.append(p_poly)

    if colorbar and vmin is not None and vmax is not None:
        listed_cmap = ListedColormap(colormap[:, :3])
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = cm.ScalarMappable(cmap=listed_cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

    # Lay out with a title in place so update()'s per-frame titles fit.
    if titles is not None:
        ax.set_title(titles[0])
    plt.tight_layout()
    # Animation update function
    def update(frame):
        curr_values = values[frame]
        for it, p_poly in enumerate(polys):
            color = colormap[int(curr_values[it]),:3]
            p_poly.set_facecolor(color)
        if titles is not None:
            ax.set_title(titles[frame])
        return polys
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(values), interval=1/fps*1000, blit=False)
    if filename is not None:
        _use_bundled_ffmpeg_if_needed()
        anim.save(
            filename,
            writer="ffmpeg",
            fps=fps,
            codec="libx264",
            extra_args=["-pix_fmt", "yuv420p"]
        )
    if plot:
        plt.show()
    plt.close(fig)
