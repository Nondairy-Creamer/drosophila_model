# Reference

Details beyond the [README](../README.md): the data pipeline, every training
tool and option, every analysis script, and the code layout.

## Setup on a server

On a server whose system Python already has a working torch (e.g. RunPod), skip
the venv and use that Python. If pip reports that the environment is externally
managed, run `PIP_BREAK_SYSTEM_PACKAGES=1 bash setup.sh`.

## Data

`[data].data_folder` holds the raw inputs and the files baked from them. The
`data-v1` [release](https://github.com/Nondairy-Creamer/drosophila_model/releases/tag/data-v1)
has both, as two zips (the raw one split into parts under GitHub's 2 GiB file
limit). `setup.sh` (through `data_handling.prepare_data`) downloads the baked
`connectome_data/baked/` and `stimulus/baked/`
(`drosophila_model_baked_v1.zip`) and bakes the training sets;
`setup.sh --raw` downloads the raw inputs instead
(`drosophila_model_raw_v1.zip`) and runs every step in the "Produced by"
column. If the repository is private, the download uses your GitHub CLI login
(`gh auth login`) or a token in `GITHUB_TOKEN`.

To publish changed baked files (e.g. after splitting another cell type like
CT1), bump `DATA_VERSION` in `data_handling/prepare_data.py` and run
`python -m drosophila_model.data_handling.build_data_release`, which zips the
data folder (in a temporary folder, removed afterwards) and uploads it as a
new `data-<version>` release with the GitHub CLI. Code at older commits keeps
downloading the release it names.

| Folder | Contents | Produced by |
|---|---|---|
| `connectome_data/raw/` | FlyWire Codex downloads (FAFB v783), as `.csv.gz`: `consolidated_cell_types`, `connections_princeton_no_threshold`, `column_assignment`, `fafb_v783_princeton_synapse_table`, and optionally `cell_stats` / `classification`; the Reiser lab eye map, `eyemap_T4/data/eyemap.RData` | the raw bundle (files as downloaded from [Codex](https://codex.flywire.ai) and [eyemap_T4](https://github.com/reiserlab/eyemap_T4)) |
| `connectome_data/baked/` | `cell_info.parquet`, `synapses.parquet` (one row per pre, post, neuropil), `optic_lobe_column_index.parquet`, `optic_lobe_columns.parquet`, and the CT1 split (`ct1_pieces.parquet`, `ct1_connections.parquet`) | `data_handling.format_connectome_data`, which first runs `data_handling.split_ct1` (each CT1 split into per-column `CT1_M` / `CT1_L` pieces by synapse position) and `data_handling.build_column_geometry` (column viewing directions from the Reiser lab eye map, Zhao et al. 2025, in `connectome_data/raw/eyemap_T4/`) if their outputs are missing |
| `stimulus/raw/` | `panorama_001-100.rar` (panorama `.mat` files, read from the archive without extracting it) | the raw bundle (Meyer et al. 2014's `data_001-100.rar`, renamed) |
| `stimulus/baked/` | `all_panoramic.pt` | `data_handling.process_raw_data random_pano` |
| `stimulus/training/` | `<type>_train.pt` and `<type>_test.pt` | `data_handling.create_training_data <type>` |

Training never generates stimuli on the fly; it loads `<type>_train.pt`.
`setup.sh` bakes the types in `[stimulus].stimulus_type`; bake any other type
you train on with:

```bash
python -m drosophila_model.data_handling.create_training_data random_pano
```

Bakeable types are `static_sine`, `random_sine`, `counterphase_sine`,
`random_pano`, `full_field_step` and `full_field_flicker`, all computed on the
fly (`random_pano` from the raw panorama images). All of them bake two-eye
data (`[right | left]` on the column axis).

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
python -m drosophila_model.training.train --seed 42 --device cuda:0
```

```bash
python -m drosophila_model.training.launch_runs --runs-per-gpu 4
```

| Module (`drosophila_model.training.…`) | What it does |
|---|---|
| `train` | Train one model from `config.toml` (`--seed`, `--device`, `--run-id`, `--timestamp`, `--config-path`; `--model-folder` warm-starts from an existing run with fresh noise, optimizer and folder). |
| `launch_runs` | Launch `runs_per_gpu` × GPUs runs with seeds `base_seed + i` as `models/<tag>_gpuN_runM/`. Detaches by default; launcher log in `logs/launch_runs_<tag>.log`, per-run logs in `logs/`. |
| `resume_runs` | Resume a run in place from its best checkpoint, optimizer state and loss history (`--model-folder`, `--num-epochs`); `--new-session` copies it to `<run>_cont` and restarts the optimizer. |
| `rank_runs` | Rank runs under `models/` by best training loss and plot loss curves per group (`figures/rank_runs/`). |
| `sync_models` | rsync runs to/from a remote server (`<server>`, `--push`, `--model-folder`, `--remote-dir`). |
| `stop_runs` | Stop all launcher and training processes (POSIX). |

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
- **Readouts** (`[model.regions.optic_lobe].readouts`):
`FlowConv2Readout` (default), `FlowConv2DenomReadout`, `FlowConv3Readout`,
`FlowLinearReadout`, `FlowNeuronReadout`, `MotionMagReadout`,
`MotionMagConv2Readout`. Regularizers: `L1Reg`, `L2Reg`, `SigmaReg`.
- **Sparsity / size:**
  - `[model].synapse_count_threshold` drops weak connections: neuron pairs
  whose total synapse count (summed over neuropils) is below it.
  - `[model].use_connectome_subset` keeps only the listed cell types.
  - `analysis.model_analysis.model_trimming` removes the least important cell
  types from a trained model while its training loss stays within 10%.
- **Region:** `train` builds optic-lobe models only (right eye). Antenna-lobe
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
| `performance_sweep` | Flow and ego-motion cosine similarity on baked `random_sine`, `random_pano` (`--split test` by default). |
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
| `antenna_lobe_decoding` / `olf_decoding_mimic` | Odor-identity decoding from ORN vs. PN responses of antenna-lobe models. |

### `drosophila_model.analysis.movie_scripts`

| Script | What it does |
|---|---|
| `run_model_on_stimulus` | Model flow (and `--ego-motion` odometry, or `--input-conv` for the image after the input conv) on a freshly generated stimulus. |
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
  training/                 train, launch_runs, resume_runs, training_utilities
                            (load_processed_stimulus, model_outputs, compute_loss_components,
                            fit), loss_functions, model management tools
  data_handling/            raw-data and training-data baking, stimulus_generators, sampling,
                            prepare_data, build_data_release, format_connectome_data,
                            split_ct1, build_column_geometry
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
