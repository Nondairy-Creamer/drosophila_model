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

import pandas as pd

import data_utilities as du
import stimulus_utilities as su

from analysis_utilities import make_stimulus_flow_movie


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
        "--fps", type=int, default=100, help="Output movie frame rate"
    )
    parser.add_argument(
        "--stim-type", type=str, default='flyview',
        help="Stimulus type (default: use value from config)"
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

    stimulus_type = args.stim_type if args.stim_type else config['stimulus']['stimulus_type']
    frame_rate = args.fps

    # --- Load column positions for scatter/quiver plots ---
    column_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_column_info.parquet')
    column_phi_theta = column_info_df[['phi', 'theta']].to_numpy()

    # --- Load stimulus and optic flow ---
    print("Loading stimulus and optic flow...")
    stimulus, optic_flow, _ = su.get_stimulus_one_eye(stimulus_type, config, column_phi_theta)
    stimulus = stimulus.numpy()
    optic_flow = optic_flow.numpy()

    print(f"Stimulus shape: {stimulus.shape}")
    print(f"Optic flow shape: {optic_flow.shape}")
    import numpy as np
    phi = column_phi_theta[:, 0]
    theta = column_phi_theta[:, 1]
    cell_x = np.sin(theta) * np.cos(phi)
    cell_y = np.sin(theta) * np.sin(phi)
    cell_z = np.cos(theta)

    for stim_index in args.stim_index:
        print(f"\nCreating movie for stimulus {stim_index} with {stimulus.shape[1]} frames...")
        out_path = output_folder / f'stimulus_flow_stim_{stim_index}.mp4'

        make_stimulus_flow_movie(
            stimulus[stim_index],
            optic_flow[stim_index],
            cell_x,
            cell_z,
            out_path,
            frame_rate,
            cell_y=cell_y,
        )


if __name__ == "__main__":
    main()
