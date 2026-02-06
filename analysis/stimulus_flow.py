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
        "--fps", type=int, default=50, help="Output movie frame rate"
    )
    parser.add_argument(
        "--stim-type", type=str, default=None,
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

    # --- Load stimulus and optic flow ---
    print("Loading stimulus and optic flow...")
    stimulus, optic_flow = su.get_stimulus(stimulus_type, config)
    stimulus = stimulus.numpy()
    optic_flow = optic_flow.numpy()

    print(f"Stimulus shape: {stimulus.shape}")
    print(f"Optic flow shape: {optic_flow.shape}")

    # --- Load column positions for scatter/quiver plots ---
    column_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_column_info.parquet')
    cell_x = column_info_df.x.to_numpy()
    cell_z = column_info_df.z.to_numpy()

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
        )


if __name__ == "__main__":
    main()
