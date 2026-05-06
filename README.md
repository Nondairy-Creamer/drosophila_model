# Drosophila Connectome-Constrained Neural Models

Training and analysis of connectome-constrained models of the *Drosophila* optic lobe. Models use synapse-level connectivity from the FlyWire connectome to define network architecture, then learn synaptic weights and dynamics that reproduce motion vision computations (optic flow / ego-motion estimation).

For real-time streaming on a Raspberry Pi 5 (CSI cameras, 30 Hz, UDP-publishing ego-motion), see [README-rpi.md](README-rpi.md).

## Quick Start

Train a single model:

```bash
python run_model.py --seed 42 --device cuda:0
```

Launch multiple runs across all available GPUs:

```bash
python submit_runs.py --runs-per-gpu 4
```

Both scripts read parameters from `config.toml` (lives in the `connectome_data` sibling directory by default).

## Repo Structure

### Training Pipeline

| File | Role |
|---|---|
| `config.toml` | All hyperparameters: model architecture, training schedule, stimulus, data paths |
| `run_model.py` | Entry point — loads config, builds model, generates stimulus, runs training loop |
| `submit_runs.py` | Multi-GPU launcher — distributes `run_model.py` jobs across GPUs with different seeds |
| `continue_training.py` | Resume training from a saved checkpoint |
| `neural_models.py` | Model definitions (`NeuralModel`, `CurrentModel`, etc.) — connectome-constrained RNNs with decoders (see [model_architecture.md](model_architecture.md)) |
| `training_utilities.py` | Training loop, gradient updates, checkpointing |
| `loss_functions.py` | Loss functions (MSE, correlation, cosine, combined) |
| `stimulus_utilities.py` | Stimulus generation — random panoramas, sine gratings, drone footage, optic flow computation |
| `data_utilities.py` | Config loading/saving, connectome data helpers |
| `model_utilities.py` | Git provenance, misc model helpers |
| `plotting_utilities.py` | Plotting helpers used during training |
| `movie_utilities.py` | Video/animation utilities |

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

- **Connectome data** is stored in the sibling `connectome_data/` directory (parquet files for cell info, synapse counts, column positions).
- **Input neurons**: L1, L2, L3 (lamina cells that receive photoreceptor input).
- **Output neurons**: T4a-d, T5a-d (direction-selective motion detectors for the four cardinal directions).
- **Coordinate conventions** are documented in `coordinate_conventions.md`.
- Trained models are saved under `models/<timestamp>/` with the config, initial model, stimulus, and target tensors for full reproducibility.
