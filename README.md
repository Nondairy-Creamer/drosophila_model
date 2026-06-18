# Drosophila Connectome-Constrained Neural Models

Training and analysis of connectome-constrained models of the *Drosophila* optic lobe. Models use synapse-level connectivity from the FlyWire connectome to define network architecture, then learn synaptic weights and dynamics that reproduce motion vision computations (optic flow / ego-motion estimation).

## Quick Start

Train a single model:

```bash
python run_model.py --seed 42 --device cuda:0
```

Launch multiple runs across all available GPUs:

```bash
python submit_runs.py --runs-per-gpu 4
```

Both scripts read parameters from `config.toml` at the repo root, which overlays `drosophila_model/config_default.toml` (shipped inside the package). `config.toml` is gitignored — create your own from the defaults. Connectome data lives in the sibling `drosophila_model_data/` directory (path set by `[data].data_folder` in the config).

## Repo Structure

### Training Pipeline

| File | Role |
|---|---|
| `config.toml` | All hyperparameters: model architecture, training schedule, stimulus, data paths |
| `run_model.py` | Entry point — loads config, builds model, generates stimulus, runs training loop |
| `submit_runs.py` | Multi-GPU launcher — distributes `run_model.py` jobs across GPUs with different seeds |
| `continue_training.py` | Resume training from a saved checkpoint |
| `neural_models.py` | Model definitions (`BaseRnn`, `CurrentModel`, etc.) — connectome-constrained RNNs with decoders (see [model_architecture.md](model_architecture.md)) |
| `training_utilities.py` | Training loop, gradient updates, checkpointing, baked-stimulus loading (in `training/`) |
| `loss_functions.py` | Loss functions (MSE, correlation, cosine, combined) |
| `config.py` | Config loading/saving (`load_config`, `save_config`); data config is merged under `config['data']` |
| `connectome_loading.py` | `ConnectomeData` + `load_connectome` / `get_connectome_subset` (in `model_definition/`) |
| `plotting_utilities.py` | Plotting helpers used during training (in `training/`) |
| `download_models.py` | Rsync model artifacts to/from a remote server (in `training/`) |

### Data Handling

Stimulus and connectome data prep lives in `data_handling/`.

| File | Role |
|---|---|
| `stimulus_generators.py` | On-the-fly stimulus generators — random panoramas, sine gratings, edges, drone footage, optic flow |
| `process_raw_data.py` | Bake the large raw inputs (drone videos, panorama `.mat`) from `stimulus/raw` into cached `.pt` files under `stimulus/baked` (`small_drone.pt`, `rpi_drone.pt`, `all_panoramic.pt`) |
| `create_training_data.py` | Bake a fixed, training-ready dataset (`stimulus/training/<stimulus_type>.pt`) that training loads |
| `drone_utilities.py` | Raw drone-rig processing — video loading, pose loading, dome calibration, ego-motion solving, per-dataset processors |
| `estimate_pose.py` | `small_drone` AprilTag pose-estimation engine (dual-camera bundle adjustment) |
| `drone_pose.py` | `small_drone` batch driver (CLI) — writes `pose_<id>.npz` via `estimate_pose` |
| `extract_tags.py` | `rpi_drone` dome tag-bearing / board-pose extractor (CLI) |
| `sampling.py` | Shared spatial/temporal sampling primitives (ommatidium sampling matrices, time interpolation) |
| `format_connectome_data.py` | Bake the canonical connectome bundle from the raw CSVs |

### Analysis

Analysis scripts are assorted and live in the `analysis/` folder. Some highlights:

- `find_best_model.py` — rank trained models by score
- `analyze_connectome.py` / `visualize_connectome.py` — inspect connectivity
- `make_flow_movie.py` / `make_flow_movie_two_eyes.py` — optic flow visualizations
- `sine_wave_sweep.py` — tuning curve analysis
- `plot_odometry.py` — ego-motion estimation
- `benchmark_model.py` — runtime benchmarking
- `analyze_decoder.py` / `analyze_conv.py` — inspect learned decoder and input conv weights
- `analyze_signal_prop.py` — signal propagation analysis
- `analysis_utilities.py` — shared analysis helpers

### Tests

Tests live in `tests/`. Run them with:

```bash
python tests/run_tests.py
```

## Key Concepts

- **Connectome data** is stored in the sibling `drosophila_model_data/` directory (parquet files for cell info, synapse counts, column positions).
- **Input neurons**: L1, L2, L3 (lamina cells that receive photoreceptor input).
- **Output neurons**: T4a-d, T5a-d (direction-selective motion detectors for the four cardinal directions).
- **Coordinate conventions** are documented in `coordinate_conventions.md`.
- Trained models are saved under `models/<timestamp>/` with the config, initial model, stimulus, and target tensors for full reproducibility.
