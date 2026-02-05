"""
Create a movie of the natural movie stimulus with true optic flow.
1x2 layout:
  - Left: Stimulus intensity (scatter plot)
  - Right: True optic flow (quiver plot)

No model required — loads the stimulus directly via su.get_stimulus.
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import imageio as iio_v2

import data_utilities as du
import stimulus_utilities as su


def fig_to_rgb(fig):
    """Convert matplotlib figure to RGB array."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[..., :3].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make a movie showing stimulus and true optic flow (no model needed)."
    )
    parser.add_argument(
        "--config", type=str, default="config.toml", help="Path to config file"
    )
    parser.add_argument(
        "--stim-index", type=int, nargs='+', default=[5],
        help="Stimulus indices to visualize (space-separated list)"
    )
    parser.add_argument(
        "--fps", type=int, default=50, help="Output movie frame rate"
    )
    parser.add_argument(
        "--output-dir", type=str, default="figures/natural_movie",
        help="Output directory for the movie"
    )
    args = parser.parse_args()

    config = du.load_config(args.config)
    connectome_folder = Path(config['data']['connectome_folder'])
    output_folder = Path(args.output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)

    stimulus_type = config['stimulus']['stimulus_type']
    frame_rate = args.fps

    # --- Load stimulus and optic flow ---
    print("Loading stimulus and optic flow...")
    stimulus, optic_flow = su.get_stimulus(stimulus_type, config)
    stimulus = stimulus.numpy()
    optic_flow = optic_flow.numpy()

    print(f"Stimulus shape: {stimulus.shape}")
    print(f"Optic flow shape: {optic_flow.shape}")

    # --- Load column positions for scatter/quiver plots ---
    import pandas as pd
    column_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_column_info.parquet')
    cell_x = column_info_df.x.to_numpy()
    cell_z = column_info_df.z.to_numpy()

    # Axes limits (shared across all movies)
    x_margin = 0.1 * (cell_x.max() - cell_x.min())
    z_margin = 0.1 * (cell_z.max() - cell_z.min())

    for stim_index in args.stim_index:
        # --- Extract single stimulus ---
        stim_single = stimulus[stim_index, :, :]          # (time, columns)
        true_flow_single = optic_flow[stim_index, :, :, :]  # (time, columns, 2)
        num_time, num_cols = stim_single.shape

        print(f"\nCreating movie for stimulus {stim_index} with {num_time} frames...")

        # --- Normalizations ---
        stim_min, stim_max = np.nanmin(stim_single), np.nanmax(stim_single)
        stim_norm = mcolors.Normalize(vmin=stim_min, vmax=stim_max)

        true_flow_u = true_flow_single[:, :, 0]  # horizontal (LR)
        true_flow_v = true_flow_single[:, :, 1]  # vertical (UD)

        true_magnitude = np.hypot(true_flow_u, true_flow_v)
        max_mag = np.nanpercentile(true_magnitude, 99) if np.any(true_magnitude) else 1.0
        flow_norm = mcolors.Normalize(vmin=0.0, vmax=max_mag)

        # --- Create figure (1x2 layout) ---
        fig, (ax_stim, ax_true) = plt.subplots(1, 2, figsize=(12.8, 5.12), dpi=100)

        for ax in [ax_stim, ax_true]:
            ax.set_xlim(cell_x.max() + x_margin, cell_x.min() - x_margin)
            ax.set_ylim(cell_z.min() - z_margin, cell_z.max() + z_margin)
            ax.set_aspect('equal')
            ax.set_xlabel('x (anterior ← → posterior)')
            ax.set_ylabel('z (ventral ↔ dorsal)')

        # Initialize stimulus scatter
        sc_stim = ax_stim.scatter(
            cell_x, cell_z, c=stim_single[0, :], cmap='gray',
            norm=stim_norm, s=90, edgecolors='none'
        )
        ax_stim.set_title('Stimulus')
        fig.colorbar(sc_stim, ax=ax_stim, label='intensity')

        # Arrow scale
        arrow_scale = 0.2 / max_mag if max_mag > 0 else 1.0

        # Initialize true optic flow quiver (negate horizontal for inverted x-axis)
        q_true = ax_true.quiver(
            cell_x, cell_z,
            -true_flow_u[0, :], true_flow_v[0, :],
            np.hypot(true_flow_u[0, :], true_flow_v[0, :]),
            cmap='viridis', norm=flow_norm,
            angles='xy', scale_units='xy', scale=1 / arrow_scale,
            width=0.003
        )
        ax_true.set_title('True Optic Flow')
        fig.colorbar(q_true, ax=ax_true, label='magnitude')

        plt.tight_layout()

        # --- Create movie ---
        out_path = output_folder / f'stimulus_flow_stim_{stim_index}.mp4'
        print(f"Writing movie to {out_path}...")

        with iio_v2.get_writer(str(out_path), codec="libx264", fps=frame_rate) as writer:
            for t in range(num_time):
                # Update stimulus
                sc_stim.set_array(stim_single[t, :])

                # Update true optic flow (negate horizontal for inverted x-axis)
                u_true_t = true_flow_u[t, :]
                v_true_t = true_flow_v[t, :]
                mag_true_t = np.hypot(u_true_t, v_true_t)
                q_true.set_UVC(-u_true_t, v_true_t, mag_true_t)

                # Update title with frame number
                ax_stim.set_title(f'Stimulus (frame {t+1}/{num_time})')

                # Capture frame
                writer.append_data(fig_to_rgb(fig))

                if (t + 1) % 20 == 0:
                    print(f"  Frame {t+1}/{num_time}")

        plt.close(fig)
        print(f"Done! Movie saved to {out_path}")


if __name__ == "__main__":
    main()
