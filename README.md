# Drosophila connectome models

Sparse, efficient recurrent models of the *Drosophila* nervous system whose
architecture comes from the FlyWire connectome. Each model is a
`ConnectomeModel`: neurons and signed synapses are taken from the connectome
(thresholded by synapse count, optionally restricted to a subset of cell
types), and only per-cell-type parameters — gains, time constants, reversal
potentials, input filters and readouts — are learned. The main use is the
right optic lobe trained to report optic flow and ego-motion (rotation and
translation) from natural and synthetic visual input.

- Model internals: [model_architecture.md](model_architecture.md)
- Coordinate and flow sign conventions: [coordinate_conventions.md](coordinate_conventions.md)
- Derivations (LyX): [math/](math/)

## Setup

Requires Python 3.11 or newer.

```bash
pip install -r requirements.txt
```

```bash
pip install -e . --no-deps
```

On a fresh GPU server, `setup_server.sh [DATA_FOLDER]` does both. It also
downloads the data bundle and installs the per-machine `config.toml` with
`[data].data_folder` set. Finally it bakes any missing training sets.

All commands below are run **from the repo root**, as
`python -m drosophila_model.<subpackage>.<module>`. Relative paths such as
`config.toml`, `models/` and `figures/` resolve against the working directory.

### Configuration

Settings come from two packaged defaults:

- [`drosophila_model/config_default.toml`](drosophila_model/config_default.toml):
  model, training, which datasets to train on.
- [`drosophila_model/config_default_data.toml`](drosophila_model/config_default_data.toml):
  stimulus-generation parameters, merged under `config['data']`.

Each is overlaid by an optional, gitignored file at the repo root:
`config.toml` and `config_data.toml`. Overlay keys must already exist in the
defaults, so typos fail loudly. The data bundle is read from
`[data].data_folder`, which defaults to `../drosophila_model_data` (a folder
next to the repo); set it in `config.toml` if your data lives elsewhere. A
minimal `config.toml`:

```toml
[model]
device = "cuda:0"          # falls back to CPU automatically when CUDA is unavailable
dynamics_type = "ConductanceDynamics"

[training]
loss_components = ["flow_loss", "odometry_loss"]

[stimulus]
stimulus_type = ["full_field_step", "random_pano", "counterphase_sine"]

[data]
data_folder = "/path/to/drosophila_model_data"
```

In code, use `config.load_config()` for the live config,
`config.load_run_config(model_folder)` for a trained run's saved config, and
`config.get_data_folder(config)` for the data location.

## Data

`[data].data_folder` holds:

| Folder | Contents | Produced by |
|---|---|---|
| `connectome_data/baked/` | `cell_info.parquet`, `synapses.parquet` (one row per pre, post, neuropil), `optic_lobe_column_index.parquet`, `optic_lobe_columns.parquet` | `data_handling.format_connectome_data`, from the raw FlyWire CSVs in `connectome_data/raw/` (`optic_lobe_columns.parquet`, the column viewing directions, ships with the data bundle) |
| `stimulus/raw/` | panorama `.mat` files, the rendered fly-view `.pkl` | recorded / rendered elsewhere |
| `stimulus/baked/` | `all_panoramic.pt`, `flyview.pt` | `data_handling.process_raw_data [random_pano\|flyview]` (omit the type to process both) |
| `stimulus/training/` | `<type>_train.pt` and `<type>_test.pt` | `data_handling.create_training_data <type>` |

Training never generates stimuli on the fly; it loads `<type>_train.pt`. Bake
one dataset per stimulus type you train on:

```bash
python -m drosophila_model.data_handling.create_training_data random_pano
```

Bakeable types are `static_sine`, `random_sine`, `counterphase_sine`,
`random_pano`, `full_field_step`, `full_field_flicker` and `flyview`. All of
them bake two-eye data (`[right | left]` on the column axis).

The split is a seeded shuffle of whole source groups, so correlated examples
never straddle train/test. Examples share a group when they are segments of
one clip, repeats of one panorama, or phase/contrast variants of one grating.
The first `train_fraction` of the groups go to `_train.pt` and the rest to
`_test.pt`. `full_field_step` and `full_field_flicker` are train-only (no
`_test.pt`).

`edge`, `expanding_dot` and `directional_sine` are analysis-only probes and
are refused by `create_training_data`. Re-bake after changing `[model].dt` or
the generation parameters; loading checks the baked `dt`.

## Training

```bash
python -m drosophila_model.training.run_model --seed 42 --device cuda:0
```

```bash
python -m drosophila_model.training.submit_runs --runs-per-gpu 4
```

| Module (`drosophila_model.training.…`) | What it does |
|---|---|
| `run_model` | Train one model from `config.toml` (`--seed`, `--device`, `--run-id`, `--timestamp`, `--config-path`; `--model-folder` warm-starts from an existing run with fresh noise, optimizer and folder). |
| `submit_runs` | Launch `runs_per_gpu` × GPUs runs with seeds `base_seed + i` as `models/<tag>_gpuN_runM/`. Detaches by default; launcher log in `logs/submit_runs_<tag>.log`, per-run logs in `logs/`. |
| `continue_training` | Resume a run in place from its best checkpoint, optimizer state and loss history (`--model-folder`, `--num-steps`); `--new-session` copies it to `<run>_cont` and restarts the optimizer. |
| `find_best_model` | Rank runs under `models/` by best training loss and plot loss curves per group (`figures/find_best_model/`). |
| `download_models` | rsync runs to/from a remote server (`<server>`, `--push`, `--model-folder`, `--remote-dir`). |
| `kill_submit_runs` | Stop all launcher and training processes (POSIX). |

A run folder `models/<timestamp>[_<run_id>]/` contains:

- `config.toml`: the full merged config, plus `[run_info]` with seed, device,
  run id, eye count and the rig's camera offset.
- `model_init.pt` and `model.pt` (the best checkpoint so far).
- `model_opt.pt`: optimizer state and loss history.
- `cell_inputs.pt` and `target.pt`: the exact training data.
- `figures/training/loss.png`.

How a run is shaped:

- **Eyes:** the eye count comes from the dataset. Two-eye data is trained
  through the optic lobe's two-eye ego-motion solver, using the camera offset
  registered for that stimulus type. `odometry_loss` needs two-eye data.
- **Loss components** (`[training].loss_components`): `flow_loss`,
  `odometry_loss` and `motion_mag`. Regularizers listed among the readouts
  add their penalty to the loss.
- **Dynamics** (`[model].dynamics_type`): `CurrentDynamics`,
  `CurrentDivisiveDynamics`, `ConductanceDynamics`, `CellAnnDynamics`
  (per-neuron ANN on the connectome; depth `[model].cell_ann_depth`) or
  `ArbitraryDynamics` (connectome-free all-to-all cell-type baseline).
- **Readouts** (`[model.regions.optic_lobe].readouts`): `FlowConv2Readout`
  (default), `FlowConv3Readout`, `FlowLinearReadout`, `FlowNeuronReadout`,
  `MotionMagReadout`, `MotionMagConv2Readout`. Regularizers: `L1Reg`, `L2Reg`,
  `SigmaReg`.
- **Sparsity / size:**
  - `[model].synapse_count_threshold` drops weak connections: neuron pairs
    whose total synapse count (summed over neuropils) is below it.
  - `[model].use_connectome_subset` keeps only the listed cell types.
  - `analysis.model_analysis.model_trimming` removes the least important cell
    types from a trained model while its training loss stays within 10%.
- **Region:** `run_model` trains optic-lobe models (right eye). Antenna-lobe
  models are built by the olfactory analysis scripts.

## Analysis

Analysis scripts take a model folder (or its `model.pt`). Figures follow one
convention:

- Model-specific figures go to `<model_folder>/figures/<analysis_name>/`.
- General figures go to `figures/<analysis_name>/`.
- Non-figure data (STA arrays, dumps) goes to
  `<model_folder>/analysis_data/<analysis_name>/`, and caches go to `cache/`.

Scripts that replay a model's own `cell_inputs.pt` evaluate on its training
split. `performance_sweep` evaluates on the held-out `_test.pt` files.

### `drosophila_model.analysis.model_analysis`

| Script | What it does |
|---|---|
| `benchmark_model` | Standard battery on one model: edge sweep, pano odometry, white-noise and sine silencing, performance sweep (each skippable). |
| `performance_sweep` | Flow and ego-motion cosine similarity on baked `random_sine`, `random_pano`, `flyview` (`--split test` by default). |
| `pano_sweep` | Odometry (rotation and translation) on random-velocity panoramas. |
| `plot_odometry` | Predicted vs. true odometry on the run's training stimuli. |
| `edge_sweep` / `sine_wave_sweep` / `natural_pano_velocity_sweep` | Velocity / temporal-frequency tuning of readouts and cells. |
| `white_noise` | Spike-triggered-average spatiotemporal filters of cell types. |
| `silencing` / `sine_wave_silencing` / `white_noise_silencing` | Effects of silencing cell-type combinations on loss, tuning and filters. |
| `model_trimming` | Remove cell types by importance (binary search) while loss stays within 10%; saves `<model>_trimmed/`. |
| `model_timing` | Inference time and peak memory. |
| `cell_input_traces` | Synaptic input decomposition onto T4a/T5a for edge or sine stimuli. |
| `signal_propagation` | L1 → Tm1 → T5 responses to a sine grating in an untrained model. |
| `analyze_connectome` / `visualize_connectome` | Spatial layout of T4a inputs / cell-type connectivity matrix. |
| `analyze_decoder` / `plot_input_conv_filters` | Learned readout weights / learned input filters. |
| `find_steady_state` | Recompute and save a model's steady state (overwrites `model.pt`). |
| `view_edge_stimulus`, `view_sine_stimulus`, `view_impulse_stimulus`, `view_natural_scene`, `view_stimulus_preprocessing` | Per-stimulus response plots and hex-mosaic movies of a trained run (`--model`, required: the run folder or its `model.pt`). |
| `run_stimulus_battery` | Run all `view_*` scripts and tile their hex movies. |
| `view_spatial_frequency` | Spatial power spectrum of the natural panoramas. |
| `verify_flyview_odometry` | Check the fly-view dataset's odometry reconstruction. |
| `antenna_lobe_decoding` / `olf_decoding_mimic` | Odor-identity decoding from ORN vs. PN responses of antenna-lobe models. |

### `drosophila_model.analysis.movie_scripts`

| Script | What it does |
|---|---|
| `run_model_on_stimulus` | Model flow (and `--ego-motion` odometry) on a freshly generated stimulus. |
| `run_model_on_stimulus_cells` | Diamond-layout movie of four cell types around the stimulus. |
| `run_model_on_cell_inputs` | Replay the run's training inputs and show predicted vs. true flow. |
| `natural_pano_t4t5_movies` | T4/T5 diamond movies for rotating natural panoramas. |
| `stimulus_flow` | Stimulus with its true optic flow (no model). |

## Code layout

```
drosophila_model/
  config.py                 config loading/saving, data-folder and device resolution
  model_definition/         ConnectomeModel (neural_models), dynamics mixins (dynamics),
                            regions (optic / antenna lobe), readouts, model_layers,
                            ego_motion (flow <-> 6-DOF), connectome_loading
  training/                 run_model, submit_runs, continue_training, training_utilities
                            (load_processed_stimulus, model_outputs, compute_loss_components,
                            train_model), loss_functions, model management tools
  data_handling/            raw-data and training-data baking, stimulus_generators, sampling,
                            format_connectome_data
  analysis/model_analysis/  analysis scripts + analysis_utilities
  analysis/movie_scripts/   movie scripts + movie_utilities
  fly_eye/                  hex-mosaic rendering of per-column activity
tests/                      pytest suite
```

## Tests

```bash
python -m pytest -p no:cacheprovider
```

The model-building tests use `[data].data_folder` (the repo-root
`config.toml`, or the default) and are skipped when the connectome isn't
there.
