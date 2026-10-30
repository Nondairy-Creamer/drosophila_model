# Model Architecture

Overview of the neural model classes (all in `drosophila_model/model_definition/`).

## Module Layout

| Module | Contents |
|---|---|
| `neural_models.py` | `SerializableModel`, `BaseRnn`, `ConnectomeModel`, `compose_model_class` |
| `dynamics.py` | Dynamics mixins: `CurrentDynamics`, `CurrentDivisiveDynamics`, `ConductanceDynamics`, `CellAnnDynamics`, `ArbitraryDynamics` |
| `regions.py` | `RegionView` base + per-area subclasses (`OpticLobeRegion`, `AntennaLobeRegion`), the area-name → region-class `REGION_REGISTRY`, and `build_region_specs` / `build_regions` |
| `readouts.py` | Optic-lobe readouts (`FlowConv2Readout`, …) and training-only regularizers (`L1Reg`, …) |
| `model_layers.py` | Reusable `nn.Module` building blocks: `HexGridConv`, `TemporalLRU`, `CellTypeLayerNorm` |
| `ego_motion.py` | `EyeFlow` + flow -> 6-DOF solver (`estimate_from_flow`), known-depth solver (`estimate_ego_motion_known_depth`), rotation math, velocity integration, synthetic flow (`generate_optic_flow`) |
| `connectome_loading.py` | `ConnectomeData` dataclass + `load_connectome(region_specs, count_threshold=...)` |

## Class Hierarchy

```
torch.nn.Module
├── BaseRnn (ABC)             Connectome-agnostic RNN scaffolding (nonlinearity,
│   │                         rng, dt, save/load via SerializableModel,
│   │                         set_sigma_max)
│   └── ConnectomeModel       Loads any combination of areas; owns self.regions,
│                             predict and find_steady_state
│
│  Concrete models are composed at runtime by
│      compose_model_class(dynamics_type)
│  e.g. CurrentDynamics + ConnectomeModel → ConnectomeModel_CurrentDynamics
│  (dynamics_type: CurrentDynamics | CurrentDivisiveDynamics |
│                  ConductanceDynamics | CellAnnDynamics |
│                  ArbitraryDynamics)
│
│  Anatomical scope is data-driven, not class-driven:
│      ConnectomeData built with
│          region_specs={'optic_lobe': {'neuropils': ['ME','LO','LOP','LA'],
│                                       'side': 'right'}}
│      → model.regions = {'optic_lobe': OpticLobeRegion(side='right', ...)}
│      → optic-lobe-specific methods live on that region object.
│      Region membership is union-by-neuropil: a neuron belongs to the region
│      if it has an above-threshold connection (pair total) with synapses
│      in any of the region's resolved neuropils (with intrinsic = true: only
│      if all its above-threshold connections lie inside them, ignoring
│      unassigned UNASGD synapses).
│
├── Regions (regions.py, all nn.Module so submodules land in state_dict)
│   ├── RegionView            Base: name + cell_indices + side; weakref to model;
│   │                            predict-time hooks (input mapping, recording,
│   │                            readouts, input pre-filter)
│   ├── OpticLobeRegion       Owns ALL vision-specific state and methods:
│   │                            column_phi_theta / column_p_q / column_indices /
│   │                            column_indices_per_neuron / num_columns /
│   │                            column_phi_theta_two_eyes,
│   │                            input_pre_conv / input_spatial_conv / temporal_lru
│   │                            (built when region_opts['use_input_conv']=true),
│   │                            plus its own config (use_input_conv, conv_size_x,
│   │                            input_pre_conv_size, num_temporal_modes),
│   │                            its readouts (_readout_modules),
│   │                            apply_input_conv, get_shared_columns, get_column_xyz,
│   │                            estimate_ego_motion_from_flow, predict_two_eyes,
│   │                            estimate_ego_motion_from_image, mirror_camera_offset.
│   │                          side='right' only: the left eye is modelled as a
│   │                          mirrored right eye (predict_two_eyes). num_eyes is
│   │                          an alias for num_sides (always 1).
│   └── AntennaLobeRegion     Olfactory area: one input channel per ORN type,
│                             per-type recording of PN types. Built by the
│                             olfactory analysis scripts, not by `train`.
│
├── Readouts (readouts.py, built per region from [model.regions.<area>].readouts)
│   ├── HexGridReadout        Base: stacked HexGridConv layers over per-column
│   │   │                        features of the readout_neurons
│   │   ├── FlowLinearReadout, FlowConv2NoDenomReadout, FlowConv2Readout, FlowConv3Readout   (2-D flow)
│   │   └── MotionMagReadout, MotionMagConv2Readout                 (|flow|)
│   ├── FlowNeuronReadout     2-D flow from weighted T4/T5 firing rates
│   └── Regularizer           training_only: L1Reg, L2Reg, SigmaReg,
│                                DecoderWeightL1Reg
│
└── Layers (model_layers.py)
    ├── HexGridConv           Hex-grid spatial convolution
    ├── TemporalLRU           Complex diagonal linear recurrence (temporal filter)
    └── CellTypeLayerNorm     Layer norm with per-cell-type weight/bias
```

## Configuration

A `ConnectomeModel` is parameterised by the `[model]` table (dynamics mixin,
nonlinearity, synapse threshold, …) plus the set of `[model.regions.<area>]`
sub-tables (one per loaded area). Each region owns its own `side` filter;
other keys in the sub-table are region-private (input-conv sizes, readouts,
etc.):

```toml
[model]
model_type     = 'ConnectomeModel'       # only valid value
dynamics_type  = 'ConductanceDynamics'   # a mixin class name in dynamics.py
nonlinearity   = 'ReLU'                  # any torch.nn class name, or 'identity'
use_layer_norm = false
synapse_count_threshold = 2

# Per-region option sub-table (the name must be in regions.REGION_REGISTRY).
# The membership keys (side, neuropils, intrinsic) drive the connectome
# loader; neurons_to_exclude, input_neurons, output_neurons, sign_overrides,
# readouts, readout_neurons and regularizer_parameters are consumed by the
# model / RegionView.__init__; everything else is forwarded verbatim to
# OpticLobeRegion._load_extras as `opts` (unknown keys raise).
[model.regions.optic_lobe]
side                = 'right'            # the only supported value
use_input_conv      = true
conv_size_x         = 7
input_pre_conv_size = 5
num_temporal_modes  = 4
readouts            = ['FlowConv2NoDenomReadout']
readout_neurons     = [                  # cell types the hex-grid readouts decode
    'T1', 'T2', 'T2a', 'T3',
    'T4a', 'T4b', 'T4c', 'T4d',
    'T5a', 'T5b', 'T5c', 'T5d',
    'Tm1', 'Tm2', 'Tm3', 'Tm4',
    'Tm9', 'Tm16', 'Tm20', 'Tm28', 'Tm30',
]

[model.regions.optic_lobe.regularizer_parameters]
input_layer_l1 = 0.0                     # > 0 adds a DecoderWeightL1Reg per hex-grid readout
```

The loader (`connectome_loading.load_connectome(region_specs, count_threshold=...)`)
takes a `{region_name: {'neuropils': [...], 'side': 'right'|'left'|'both',
'intrinsic': bool}}` mapping. `count_threshold` applies to the total synapse
count of a neuron pair, summed over neuropils; a pair that reaches it is an
above-threshold connection in every neuropil where it has synapses. With
`intrinsic = false` (default) a neuron is claimed if it has at least one
above-threshold connection (as pre or post) in any of the region's resolved
neuropils; with `intrinsic = true` only if all of its above-threshold
connections lie inside them. Synapses FlyWire assigns to no neuropil
(`UNASGD`, mostly in the gaps between neighbouring neuropils) don't count as
outside: otherwise a single stray one drops a cell, which removed 15-35% of
the L1, Mi1 and T1 cells.
The model's neurons are the union over regions, and every synapse between two
kept neurons is kept, whichever neuropil it sits in. Each region carries its
own side, so different regions can come from different hemispheres in the
same model (the optic lobe supports only `'right'`). Bilateral neuropil base
names (e.g. `'ME'`) are expanded to `'ME_R'` / `'ME_L'` per `side`; names that
already end in `_L` / `_R` pass through unchanged, a bare name that the bundle
has unsided (central, like `'EB'`) is kept as is, and names the bundle does
not have are dropped.

`region_specs` is built at the call site from `config['model']['regions']` via
`regions.build_region_specs(...)`: that pulls the region class's default
`neuropils` list, `default_side` (`'right'` for the optic lobe, `'both'`
otherwise) and `intrinsic` flag, each overridable per-instance via
`config['model']['regions'][<name>]`. `train.build_connectome(config)`
also applies `[model].use_connectome_subset` (a cell-type allow-list, via
`connectome_loading.get_connectome_subset` +
`load_connectome_from_dataframes`) when it is non-empty.

`config['model']['regions']` is also passed to `ConnectomeModel.__init__` as
`region_kwargs={...}` and stored on the model so it round-trips through
save/load. `config_default.toml` only ships defaults for `optic_lobe`; configs
that introduce new areas (e.g. antenna lobe) supply their own region sub-table.
`train` builds optic-lobe models only (it rejects other regions); the
antenna-lobe models are built by `antenna_lobe_decoding.py` and
`olf_decoding_mimic.py`.

## Data layout

The raw inputs come from the raw bundle of the repo's GitHub data release
(`data-v1`), and the connectome bundle is
baked once by `python -m drosophila_model.data_handling.format_connectome_data`.
`setup.sh`, through `data_handling.prepare_data`, downloads the baked
connectome and panorama cache ready-made from the release's baked bundle and
bakes the training sets (with `--raw`, it downloads the raw bundle and runs
every bake step). Everything lives,
with the stimulus data, under `config['data']['data_folder']` (read it with
`config.get_data_folder(config)`; default `../drosophila_model_data`):

```
<data_folder>/
├── connectome_data/
│   ├── raw/                          # FlyWire Codex downloads (FAFB v783), as downloaded
│   │   ├── consolidated_cell_types.csv.gz             # root_id, primary_type, additional_type(s)
│   │   ├── connections_princeton_no_threshold.csv.gz  # pre_root_id, post_root_id, neuropil, syn_count, nt_type
│   │   ├── column_assignment.csv.gz                   # optic-lobe per-neuron columns (column_id, p, q)
│   │   ├── fafb_v783_princeton_synapse_table.csv.gz   # per-synapse positions; read by split_ct1.py
│   │   │                                              # and build_column_geometry.py
│   │   ├── cell_stats.csv.gz, classification.csv.gz   # optional extras merged into cell_info
│   │   └── eyemap_T4/data/eyemap.RData  # Reiser lab eye map (Zhao et al. 2025; github.com/reiserlab/eyemap_T4)
│   └── baked/
│       ├── ct1_pieces.parquet, ct1_connections.parquet  # emitted by split_ct1.py:
│       │     # each CT1 split into per-column CT1_M / CT1_L pieces (synthetic
│       │     # root IDs); format_connectome_data.py swaps them in for CT1
│       ├── cell_info.parquet         # emitted by format_connectome_data.py
│       │     columns: root_id, primary_type, neurotransmitter, neuropils,
│       │     plus length_nm, area_nm, size_nm, flow, super_class, class,
│       │     sub_class, side when the extras are present
│       │     - neuropils: list[str] of every neuropil this neuron has at
│       │       least one synapse in (union; pre + post). Required by the
│       │       loader but not read: membership is recomputed from the
│       │       thresholded synapse table
│       │     - neurotransmitter: per-cell-type dominant pre-synaptic NT
│       │       (photoreceptors forced to 'HIS'; may be NaN for cell types
│       │       that never appear pre-synaptically)
│       ├── synapses.parquet          # emitted by format_connectome_data.py
│       │     columns: pre_root_id, post_root_id, synapse_counts, nt_type, neuropil
│       │     - one row per (pre, post, neuropil); the neuropil column drives
│       │       region membership
│       ├── optic_lobe_column_index.parquet  # emitted by format_connectome_data.py
│       │     columns: root_id, column_index (0-indexed)
│       │     # loaded by OpticLobeRegion._load_extras
│       └── optic_lobe_columns.parquet  # emitted by build_column_geometry.py from the
│             # eye map: each eye-map column matched to a FlyWire left-medulla Mi1,
│             # carried to the right-lobe column with the same (p, q); rim columns
│             # the eye map misses are interpolated over the hex grid
│             columns: column_index (0..C-1 in row order), phi, theta, p, q, x, y, z,
│             source ('eye_map' | 'interpolated')
│             # right-eye geometry, loaded by OpticLobeRegion._load_extras
└── stimulus/
    ├── raw/                          # raw sources, read once by process_raw_data.py
    │   └── panorama_001-100.rar      # raw panorama .mat source images (random_pano),
    │                                 # read from the archive without extracting it
    ├── baked/                        # cache from process_raw_data.py random_pano,
    │   │                             # read by the generators in stimulus_generators.py
    │   └── all_panoramic.pt          # consolidated raw panorama images (random_pano)
    └── training/                     # fixed datasets from create_training_data.py <stimulus_type>
        ├── <stimulus_type>_train.pt  # loaded for training by training_utilities.load_processed_stimulus
        └── <stimulus_type>_test.pt   # held-out split for evaluation (not written for
                                      # train-only types, e.g. full_field_step)
```

`create_training_data` splits each dataset by whole source group (correlated
examples, e.g. the segments of one clip or the phase/contrast variants of one
grating, share a group): the groups are shuffled with `[data].shared.seed` and
the first `train_fraction` of them go to `_train.pt`. `train_fraction` comes
from `[data].shared` unless the type's own `[data.<stimulus_type>]` table
sets one (in `config_default_data.toml`, `full_field_step` and
`full_field_flicker` use 1.0, so they have no `_test.pt`).

Universal fields land on `ConnectomeData`; per-region union-membership masks
land in `region_membership` (one bool array per region, aligned with
`root_ids`). Per-region geometry / per-neuron lookups (today: optic-lobe
column index and hex geometry) are loaded by the corresponding `RegionView`
subclass from `connectome_data/baked/`, so the model itself stays
anatomy-agnostic. All visual-specific state and operations
(`column_phi_theta`, `column_p_q`, `num_columns`, `column_indices_per_neuron`,
the input-conv stack — `input_pre_conv` / `input_spatial_conv` / `temporal_lru`
—, the readouts, `apply_input_conv`, `get_shared_columns`, `get_column_xyz`,
`estimate_ego_motion_from_flow`, `predict_two_eyes`,
`estimate_ego_motion_from_image`) live on `model.regions['optic_lobe']` and
are accessed through it.

To add an area (e.g. antenna lobe):
1. Subclass `RegionView` in `regions.py`, set its `neuropils = [...]` (bilateral
   base names; the loader appends `_L` / `_R` per `side`), override
   `_load_extras` to read any per-area files (and emit them from
   `format_connectome_data.py` if needed) and validate its own options,
   override the predict-time hooks it needs (`input_stim_size`,
   `build_input_mapping`, `initialize_recording`, `record_step`, …), and
   register it in `REGION_REGISTRY`. No change to the universal
   `cell_info.parquet` / `synapses.parquet` is required — region membership is
   computed at load time from the neuropil column.

---

## BaseRnn and ConnectomeModel

`BaseRnn` is the abstract, connectome-agnostic base: nonlinearity, seeded
`rng`, `device` / `dtype` / `dt`, `add_noise_to_parameters`, `set_sigma_max`,
and save/load (through `SerializableModel`). `ConnectomeModel` adds everything
connectome-specific: connectome ingestion, the regions (which own input
processing and readouts), simulation (`predict`), steady state, silencing
helpers and structural trimming. The dynamics mixins supply the update rule.

### Construction

`ConnectomeModel` takes a single `connectome: ConnectomeData` (built via
`connectome_loading.load_connectome(region_specs, count_threshold=...)`). The
dataclass holds:

| Field | Description |
|---|---|
| `synapse_counts_coo` | `(E, 3)` int64 array — `[pre_root_id, post_root_id, count]`, one row per neuron pair (its `synapses.parquet` rows summed over neuropils) |
| `root_ids` | `(N,)` int64 array of neuron root IDs |
| `cell_types` | `(N,)` string array of primary cell-type labels |
| `neurotransmitters` | `(N,)` string array of neurotransmitter labels per neuron |
| `region_membership` | dict `{region_name: (N,) bool}` — true where the neuron meets that region's membership rule against the thresholded synapse table (union; the same neuron can belong to multiple regions but only contributes one row to `root_ids`) |
| `region_specs` | input filter: `{region_name: {neuropils, side, intrinsic}}` actually loaded |
| `regions` | property: `list(region_specs.keys())` |
| `data_folder` | path used by `RegionView._load_extras` to find `connectome_data/baked/` |

All visual-column accessors and helpers (`column_phi_theta`, `column_p_q`,
`num_columns`, `column_indices_per_neuron`, `get_shared_columns`,
`get_column_xyz`) live on the optic-lobe region and are accessed as
`model.regions['optic_lobe'].<attr>`. The model itself owns only the
universal `region_membership` masks.

Key model-level constructor arguments (keyword-only):

| Argument | Default | Description |
|---|---|---|
| `excitatory_nt` | `['ACH']` | Neurotransmitter labels treated as excitatory |
| `inhibitory_nt` | `['GLUT', 'GABA']` | Neurotransmitter labels treated as inhibitory |
| `nonlinearity` | `'Softplus'` | Activation function (any `torch.nn` class name, or `'identity'`) |
| `dt` | `0.01` | Simulation timestep |
| `count_threshold` | `2` | Minimum total synapse count of a neuron pair to keep the connection |
| `region_kwargs` | `None` (→ `{}`) | Per-region option sub-dict: `{<area>: {...}}`. Wired up from `config['model']['regions']`. |
| `use_layer_norm` | `True` | Whether to apply `CellTypeLayerNorm` after each RNN step |
| `seed` | `None` | Seeds `self.rng`, which drives parameter init, training noise and batch order |
| `init_std` | `0.1` | Std of the Gaussian noise added to every parameter at init |
| `init_new_model` | `True` | (dynamics mixins) Run `set_sigma_max` at init; `load` passes `False` |

The `BaseRnn` class defaults (`nonlinearity='Softplus'`,
`use_layer_norm=True`) apply only to direct construction. Config-driven builds
(`train`) pass `[model].nonlinearity` and `[model].use_layer_norm`
explicitly — `config_default.toml` sets `'ReLU'` and `false` — along with
`count_threshold = [model].synapse_count_threshold`, `[model].dt`, the
`[model.regions]` table and the run's seed.

Region-owned configuration (sits in `region_kwargs['<area>']`, i.e.
`[model.regions.<area>]`):

| Region | Key | Default | Description |
|---|---|---|---|
| any | `side` | class `default_side` (`'right'` for `optic_lobe`, else `'both'`) | Hemisphere(s) of the bilateral neuropils to load |
| any | `neuropils` | class `neuropils` (`optic_lobe`: `ME`, `LO`, `LOP`, `LA`) | Neuropils that claim neurons for this region |
| any | `intrinsic` | `false` | Membership stringency (see above) |
| any | `neurons_to_exclude` | class `default_neurons_to_exclude` (`optic_lobe`: `R1-6`, `R7`, `R8`) | Cell types stripped from this region before preprocessing |
| any | `input_neurons` / `output_neurons` | class defaults (`optic_lobe`: `L1`/`L2`/`L3` in, `T4a-d`/`T5a-d` out) | `predict` defaults |
| any | `sign_overrides` | class `sign_overrides` (`optic_lobe`: `C2`/`C3` → `L1`/`L2` flipped) | `(pre, post, factor)` list applied after NT signing |
| any | `readouts` | class `default_readouts` (`optic_lobe`: `['FlowConv2NoDenomReadout']`) | Readout / regularizer class names from `readouts.py` |
| any | `readout_neurons` | class `default_readout_neurons` | Cell types the hex-grid readouts decode |
| any | `regularizer_parameters` | `{}` | Only `input_layer_l1` (weight of the auto-built `DecoderWeightL1Reg`) |
| `optic_lobe` | `use_input_conv` | `True` | Build + apply the input-conv stack (HexGridConv pre-filter + contrast norm + TemporalLRU) |
| `optic_lobe` | `conv_size_x` | `7` | Hex-grid diameter for the contrast-norm spatial conv |
| `optic_lobe` | `input_pre_conv_size` | `5` | Hex-grid diameter for the pre-filter conv |
| `optic_lobe` | `num_temporal_modes` | `4` | Complex modes in the temporal LRU |

### Connectome preprocessing (`_preprocess_connectome`)

At construction time the raw connectome is filtered in order:
1. Remove `synapse_counts_coo` rows below `count_threshold`. There is one row
   per neuron pair (summed over neuropils), so the threshold applies per pair.
2. Remove neurons whose neurotransmitter is not in `excitatory_nt` or `inhibitory_nt`.
3. Remove per-region cell-type exclusions. Each region exposes a
   `default_neurons_to_exclude` class attribute (e.g. `OpticLobeRegion` strips
   `['R1-6', 'R7', 'R8']` since photoreceptor terminals are the model's
   stimulus input, not connectome nodes). The exclusion is applied wherever
   the neuron sits inside that region's membership mask, so a cell type that
   happens to live in another region too keeps its copies there. Override via
   `region_kwargs[<name>]['neurons_to_exclude']`.
4. Remove connections involving removed neurons.
5. Remove neurons with no remaining connections.

Root IDs are then converted to dense matrix indices. The result is a sparse
`(N, N)` CSR synapse count matrix stored as `self.synapse_counts` (row =
postsynaptic, column = presynaptic). It is still unsigned: each dynamics mixin
signs or splits it in its own `__init__` (see below).

### Input processing pipeline

When `region_kwargs['optic_lobe']['use_input_conv'] = true`, raw columnar
stimulus passes through three stages before entering the RNN. The entire
pipeline is owned by `OpticLobeRegion` — the model invokes each region's
`apply_input_pre_filter(u_t, filter_state)` (the optic lobe's calls
`apply_input_conv`) inside its `predict` loop:

1. **Spatial pre-filter** (`input_pre_conv`, a `HexGridConv` with positive, normalized weights) — starts as a near-identity.
2. **Spatial contrast normalization** (`input_spatial_conv`, `HexGridConv` with positive, normalized weights) — computes a local weighted mean and variance over hex-grid neighbours (initially the column and its six neighbours), then normalizes: `y = -(x - mean) / sqrt(var + eps)`.
3. **Temporal filtering** (`TemporalLRU`) — a complex diagonal linear recurrence that applies learned bandpass filtering independently per column. Its state is the `filter_state` returned by `predict`.

Submodules live at `model.regions['optic_lobe'].input_pre_conv`,
`...input_spatial_conv`, and `...temporal_lru`; their parameters land in
`state_dict()` under the `regions.optic_lobe.*` prefix.

### Abstract methods

The dynamics mixins must implement:

- **`forward(v_prev, u_t, stim_to_cell_mapping=None)`** — one-step RNN update. Takes the previous voltage state `v_prev` `(B, N)` and external input `u_t`, returns the next voltage `v_next` `(B, N)`.
- **`firing_rate(v)`** — maps voltage to a firing rate.
- **`calc_sigma_max()`** / **`_apply_sigma_scale(scale_factor)`** — only needed by `set_sigma_max` (all three mixins implement them).

### Key methods

| Method | Description |
|---|---|
| `predict(inputs, input_neurons=None, output_neurons=None, init_state=None, init_filter_state=None)` | Run the full simulation loop. Returns per-region voltage/firing-rate traces and readout outputs, the final state, and the stim/readout mappings. |
| `find_steady_state(run_time=100, max_iter=25)` | Iterate the model with zero input until state converges; stores result in `self.steady_state`. Returns `False` (leaving it unchanged) if the dynamics blow up. |
| `save(path)` / `BaseRnn.load(path, device=None, dtype=None, freeze_params=True)` | Serialize/deserialize the model (constructor args — with the `ConnectomeData` stored as a plain dict — plus state dict, the sparse synapse matrices and the `model_type` / `dynamics_type` strings). `load` re-composes the concrete class via `compose_model_class`, defaults `device` to the local `config.toml`'s `[model].device`, points the connectome at the local `[data].data_folder`, and warns if the model was trained on another git commit. |
| `silence_neurons(names, invert_silencing=False, silence_inputs=False)` | Silence the outputs (and optionally inputs) of the specified cell types (lesion experiments). Defined by each dynamics mixin. |
| `silenced(names, ...)` | Context manager: `silence_neurons`, then restore the whole `state_dict` on exit (works for every dynamics). |
| `remove_cell_types(names)` | Return a new, smaller model rebuilt from the connectome without those cell types, with the trained parameters and steady state transferred. |
| `regions['optic_lobe'].get_shared_columns(neuron_list)` | Find columns that contain all specified cell types; returns neuron and column index arrays. |
| `regions['optic_lobe'].estimate_ego_motion_from_flow(phi, theta, d_right, d_up)` | Solve for 6-DOF ego-motion from one eye's optic flow (zero offset) via least-squares rotation estimation followed by SVD-based translation estimation; translation up to scale. Wraps `ego_motion.estimate_from_flow` (which takes a list of `EyeFlow`, each with a body-frame camera `offset`). |
| `regions['optic_lobe'].estimate_ego_motion_from_image(right_eye_inputs, left_eye_inputs, ..., eyes=('right', 'left'), camera_offset=None)` | Ego-motion pipeline: run the eye(s) through the model, decode optic flow, then solve rotation + translation. Returns `(rotations, translations, right_result, left_result)`. With both eyes, each eye carries its body-frame camera offset (`camera_offset` for the right eye, mirrored for the left; see [Two eyes and camera offsets](#two-eyes-and-camera-offsets)), so the rotation-induced inter-eye parallax recovers absolute translational scale, not just direction. |
| `regions['optic_lobe'].predict_two_eyes(inputs_right, inputs_left, ...)` | Per-eye `predict()` results via the mirror trick: both eyes are stacked on the batch axis, run once through the right optic lobe, split again, and left-eye readouts with `eye_asymmetric = True` are flipped back into the shared frame (`flip_for_left_eye`). |

### `predict` in detail

`predict` is the main simulation entry point. Given `inputs` of shape
`(batch, time, stim_size)` (for the optic lobe `stim_size = num_columns`, one
channel per hex column):

1. Builds `stim_to_cell_mapping` (`{'neuron_index', 'stimulus_index'}`) via the regions' `build_input_mapping` (optic lobe: the `input_neurons` in columns shared by all of them; default `L1`/`L2`/`L3`), or uses a pre-built mapping passed as `input_neurons`.
2. Resolves `output_neurons` per region and lets each region allocate its recording buffers (`initialize_recording`; optic lobe: per-cell-type `(batch, time, num_columns)` tensors with NaN padding, plus one buffer per readout).
3. Initializes voltage from `self.steady_state` (or a provided `init_state`).
4. For each timestep:
   - Applies the regions' input pre-filters (`apply_input_conv` when `use_input_conv`).
   - Calls `self.forward(v, u_t, stim_to_cell_mapping)` to get the next voltage.
   - Optionally applies `self.layer_norm`.
   - Computes firing rates via `self.firing_rate`.
   - Each region's `record_step` records voltage and firing rate for its output cell types and runs its readouts on the firing rates.
5. In training mode, evaluates the training-only regularizers once.
6. Returns `{'outputs': {region: {...}}, 'model_state': {'v', 'filter_state'}, 'mappings': {region: {...}}}`. `outputs['optic_lobe']` holds `voltage` / `firing_rate` dicts of `(batch, time, num_columns)` tensors per cell type plus one entry per readout (e.g. `outputs['optic_lobe']['FlowConv2Readout']`, `(batch, time, num_columns, 2)`). The per-region `input_mapping` / `output_mapping` can be passed back as `input_neurons` / `output_neurons` to skip rebuilding them.

### Training path

`training/train.py` builds a model from config and trains it on a baked
dataset:

1. `compose_model_class(dynamics_type, model_type)` gives the class;
   `build_connectome(config)` gives the `ConnectomeData`; the model is
   constructed with the config-driven arguments above.
2. `training_utilities.load_processed_stimulus(config)` loads
   `<data_folder>/stimulus/training/<type>_train.pt` for each
   `[stimulus].stimulus_type` and returns the stimulus, `optic_flow`,
   `train_mask`, `rotation` / `translation` targets, `num_eyes` and
   `eye_offset` of the dataset.
3. `training_utilities.fit` runs Adam over batches, with betas (0.9, 0.99)
   rather than PyTorch's (0.9, 0.999). Training starts on a flat stretch
   (flow_loss ~1, odometry_loss ~0.5) where almost no gradient reaches the
   network; a rare large gradient (a spiking batch) stays in Adam's
   second-moment estimate for ~1/(1 - beta2) steps and shrinks every step in
   that time, which with 0.999 could hold learning off for hundreds of steps.
   Each batch goes
   through `training_utilities.model_outputs(model, inputs, train_mask,
   num_eyes, eye_offset)`, which picks the path from the dataset's `num_eyes`:
   - `num_eyes == 2` (every baked dataset; columns `[right | left]`):
     `regions['optic_lobe'].estimate_ego_motion_from_image(...)` with
     `camera_offset=eye_offset`; both eyes' readouts are concatenated on the
     column axis and `pred_rotation` / `pred_translation` are added.
   - `num_eyes == 1`: plain `model.predict`, with every region's readouts
     flattened into one dict.

   `training_utilities.compute_loss_components(...)` then evaluates each name
   in `[training].loss_components` from `training/loss_functions.py`
   (`flow_loss`, `odometry_loss` (two-eye data only), `motion_mag`) and adds
   the training-only regularizer outputs; the training loss is their sum.
4. `train` records `[run_info]` (seed, device, run id, `num_eyes`,
   `eye_offset`) in the run's `config.toml`; `model.pt` is overwritten
   whenever the epoch loss improves.

Analysis scripts that score a model the way training does (e.g.
`performance_sweep`, on the `_test.pt` split) reuse `model_outputs`.

### Two eyes and camera offsets

Both eyes are modelled with one right optic lobe: the left eye's images (sampled
at `(-phi, theta)`) are run through it as a mirrored right eye
(`predict_two_eyes`; see `coordinate_conventions.md`). For the ego-motion solve
each eye can sit off the body centre (`ego_motion.EyeFlow.offset`). The offset
is a property of the rig that recorded the dataset, not of the model:

- a stimulus loader declares its right-eye offset with the
  `@stimulus_generators.camera_offset(x, y, z)` decorator (no current stimulus
  declares one), and `stimulus_generators.camera_offset_for_stimulus` looks it
  up (`(0, 0, 0)` when none is declared);
- `get_stimulus` returns it as `StimulusData.eye_offset`, `create_training_data`
  bakes it into the `.pt` files, `load_processed_stimulus` returns it, and
  `train` records it as `[run_info].eye_offset`
  (`analysis_utilities.run_eye_offset(run_config)` reads it back);
- `estimate_ego_motion_from_image(camera_offset=...)` uses it for the right eye
  and its sagittal mirror (`OpticLobeRegion.mirror_camera_offset`: `y` flips)
  for the left eye. `None` means `(0, 0, 0)`, a single viewpoint.

---

## Dynamics mixins (`dynamics.py`)

Each mixin owns its parameters and the methods that touch them. Unless noted,
all parameters are **per cell type** (not per neuron), expanded to per-neuron
via `neuron_to_type_index`; `T` = number of unique cell types.

## CurrentDynamics

Current-based synapse model. Synaptic input is modeled as injected current proportional to presynaptic firing rate and (signed) synapse count.

### Dynamics

```
z = D_in @ synapse_counts @ D_out @ f(v_prev) + b * u + offset
v_next = α * v_prev + (1 - α) * z
```

where:
- `f` is the nonlinearity (`[model].nonlinearity`)
- `α = sigmoid(persistence_logit)` — per-cell-type persistence (how much voltage carries over)
- `D_in`, `D_out` — diagonal scaling matrices (per-cell-type, learned in log space)
- `b` — per-cell-type stimulus strength (learned in log space); `u` drives only the input neurons in `stim_to_cell_mapping`
- `offset` — per-cell-type bias

Synapse counts are signed at construction (`_initialize_synapse_signs`):
excitatory neurotransmitters get positive sign, inhibitory get negative, and
then each region's `sign_overrides` are applied (optic lobe: the GABAergic
`C2` / `C3` → `L1` / `L2` contacts are flipped to excitatory).

### Learned parameters

| Parameter | Shape | Constraint | Role |
|---|---|---|---|
| `persistence_logit` | `(T,)` | sigmoid → (0,1) | Voltage decay rate |
| `input_strength_log` | `(T,)` | exp → positive | Post-synaptic gain |
| `output_strength_log` | `(T,)` | exp → positive | Pre-synaptic gain |
| `stimulus_strength_log` | `(T,)` | exp → positive | External input gain |
| `offset` | `(T,)` | unconstrained | Resting drive |

### Initialization

- `persistence` starts at 0.5, `stimulus_strength` at 10, `offset` at 0.
- `input_strength` and `output_strength` are initialized to `fan_in^(-1/2)` and `fan_out^(-1/2)` respectively (each cell type's mean per-neuron total of absolute incoming / outgoing synapse counts, clamped to ≥ 1), so that the variance of synaptic currents is approximately normalized.
- Gaussian noise of std `init_std` is added to every model parameter.
- For a new model (`init_new_model=True`), `set_sigma_max` then iteratively rescales `output_strength` until the spectral norm of the linearized dynamics matrix `diag(α) + diag(1 - α) @ D_in @ W @ D_out` is within 0.01 of `max_allowed_sigma` (0.95), keeping the network near the edge of stability.

---

## CurrentDivisiveDynamics

Current-based dynamics with divisive (Carandini-Heeger-style) normalization by
inhibition:

```
v_next = α * v_prev + (1 - α) * ((W+ @ g + b * u) / (W- @ g + σ) + offset)
```

where `g = D_out @ f(v_prev)`, both terms are scaled by `D_in`, and `W+` / `W-`
are the positive part and the magnitude of the negative part of the signed
synapse matrix (signed and overridden exactly as in `CurrentDynamics`). The
stimulus enters the numerator, so the feedforward drive is normalized by
inhibition too; `offset` is added outside the ratio.

Parameters and initialization are those of `CurrentDynamics` plus `sigma_log`
`(T,)` (exp → positive semi-saturation constant, initialized to 1).
`calc_sigma_max` linearizes at `v = 0` with slope 1, where the denominator is
just `σ`, so the spectral target bounds the worst-case excitatory gain.

---

## ConductanceDynamics

Conductance-based synapse model. Synaptic input is modeled as a conductance change that drives the membrane potential toward a reversal potential.

### Dynamics

Uses a frozen-conductance exponential integrator:

```
g_r = D_in_r @ synapse_counts_r @ D_out @ f(v_prev - threshold)    (per transmitter r)
g_tot = g_leak + Σ_r g_r
v_inf = (g_leak * E_leak + Σ_r g_r * E_r + b * u) / g_tot
α = exp(-dt * g_tot / C)
v_next = α * v_prev + (1 - α) * v_inf
```

Key differences from `CurrentDynamics`:
- Synapse counts are stored as **separate unsigned sparse matrices per presynaptic neurotransmitter**, since each transmitter drives toward its own reversal potential.
- Region `sign_overrides` are applied per edge: a sign-inverted edge moves into the conductance matrix of an opposite-sign transmitter (for the optic lobe, the GABAergic `C2` / `C3` → `L1` / `L2` contacts are carried by the ACH conductance), scaled by `|factor|`; `factor = 0` drops the edge. Other GABA inputs onto L1/L2 stay inhibitory.
- The effective persistence `α` is **state-dependent** — higher total conductance means faster dynamics.
- The firing rate includes a learned threshold: `f(v - threshold)`.

### Learned parameters

| Parameter | Shape | Constraint | Role |
|---|---|---|---|
| `rp_transmitter` | `(R, T)` | unconstrained | Reversal potential per transmitter per cell type |
| `rp_leak` | `(T,)` | unconstrained | Leak reversal potential |
| `g_leak_log` | `(T,)` | exp → positive | Leak conductance |
| `capacitance_inv_log` | `(T,)` | exp → positive | Inverse membrane capacitance |
| `threshold` | `(T,)` | unconstrained | Firing threshold |
| `stimulus_strength_log` | `(T,)` | exp → positive | External input gain |
| `input_strength_log` | `(R, T)` | exp → positive | Post-synaptic gain per transmitter |
| `output_strength_log` | `(T,)` | exp → positive | Pre-synaptic gain |

where `R` = number of unique neurotransmitters, `T` = number of unique cell types.

### Initialization

- Reversal potentials default to biologically motivated values: ACH → +10 mV, GLUT/GABA → -70 mV, leak → -50 mV. Any other neurotransmitter raises.
- `g_leak` = 1 and `C` = 0.04 (τ = 40 ms at rest), `threshold` = `[model].threshold_init` (default -50, the resting potential, so neurons start with no baseline firing), `stimulus_strength` = 30.
- `input_strength` and `output_strength` are initialized from fan-in/fan-out as in `CurrentDynamics` (totals across all transmitters), then Gaussian noise of std `init_std` is added to every parameter.
- Steady state is initialized to the leak reversal potential (rather than zero).
- `set_sigma_max` linearizes around rest (`g_tot = g_leak`, slope-1 nonlinearity) and rescales `output_strength` as in `CurrentDynamics`. `output_strength` is then multiplied by `[model].synapse_gain_init` (default 1).

## CellAnnDynamics

Each neuron is a small feed-forward network constrained by the connectome:

```
v_{t+1} = p * v_t + (1 - p) * (cell_ann(v_t) + b + c * u_t)
```

- The first layer is the signed connectome (NT sign plus region
  `sign_overrides`) with one learned positive weight per (post, pre)
  cell-type pair and hidden unit; later layers are dense per-neuron layers
  shared per cell type, of sizes `N, N-1, ..., 1` for `N =
  [model].cell_ann_depth` (default 1).
- Learned: per-type-pair `cell_ann.sparse_weights_log`, per-type
  `sparse_bias`, dense weights/biases, `persistence`, `stimulus_strength`,
  `offset`. First-layer weights start at the `CurrentDynamics` per-edge
  magnitude.
- Silencing sets the weights of type pairs from (and, with `silence_inputs`,
  into) the silenced types to `silence_val`. Trimming remaps type-pair rows by
  cell-type names.
- The sparse first layer is rebuilt every step, so it is several times slower
  than `CurrentDynamics`.

## ArbitraryDynamics

Connectome-free baseline that keeps only cell-type identities: an all-to-all
learned per-cell-type weight matrix `W`, applied to the mean rate of each
presynaptic type (`A[i, j] = W[type(i), type(j)] / n[type(j)]`, `n` = neurons
per type, so `W` stays O(1)),

```
v_{t+1} = p * v_t + (1 - p) * (A @ f(v_t) + b + c * u_t)
```

with learned `persistence`, `stimulus_strength` and `offset`. No sparsity or
sign constraints, so region `sign_overrides` do not apply. Silencing zeroes the
silenced types' columns of `W` (and rows, with `silence_inputs`).

---

## Helper Modules

### HexGridConv

Spatial convolution on a hex grid. Given per-column features, gathers features from neighbouring columns (determined by p,q lattice offsets within `conv_size // 2`), applies a learned weight tensor, and adds an optional bias. With `positive_weights=True` the weights are `exp`-ed and normalized by their sum over the valid neighbours (a weighted average). Boundary columns and absent neuron types (NaN features) are handled gracefully (zeroed).

Used for: the optic-lobe input pre-filter and contrast normalization (with `positive_weights=True`) and the hex-grid readouts.

### TemporalLRU

Temporal filter implemented as a complex diagonal linear recurrence. Each of `num_modes` independent modes is a damped oscillator with pole `a_k = exp(-gamma_k * dt + i * omega_k)`, parameterized by:
- `gamma` (decay rate in 1/s, constrained positive via `log_gamma`)
- `omega` (frequency in (0, π) radians per step, constrained via `logit_omega`)

The output is `y(t) = Σ_k Re(c_k z_k(t)) + d u(t)` with learned complex `b`, `c` and a positive feedthrough `d`. Provides natural bandpass filtering with learnable frequency response. Called per-timestep via `step(u_t, state)`.

### Readouts

Readouts map firing rates to per-column outputs. They are listed by class name
in `[model.regions.optic_lobe].readouts`, built by
`OpticLobeRegion.instantiate_readouts` (looked up by name in `readouts.py`, so
there is no registry; `readouts.configurable_names()` lists the valid names)
and stored in `model.regions['optic_lobe']._readout_modules`
(`model._readout_modules` aggregates every region's). Each readout runs on the
firing rates at every time step of `predict`; its output lands in
`outputs['optic_lobe'][<class name>]` with shape
`(batch, time, num_columns, *output_extra_dims)`.

| Readout | Output | Architecture |
|---|---|---|
| `FlowConv2NoDenomReadout` (default) | 2-D flow | Two hex-grid conv layers (8 then 2 channels, size 5) |
| `FlowConv2Readout` | 2-D flow | Two hex-grid conv layers (8 then 3 channels, size 5); the first two channels are divided by softplus of the third. Numerator and denominator can grow together without changing the prediction; in 2 of 5 seeds both grew while the prediction stayed near zero, and training never left `flow_loss` 1 |
| `FlowConv3Readout` | 2-D flow | Three layers (8, 6, 3 channels, size 5), same denominator |
| `FlowLinearReadout` | 2-D flow | One layer (2 channels, size 3) |
| `FlowNeuronReadout` | 2-D flow | Eight positive weights combining T4a–d / T5a–d firing rates per column |
| `MotionMagReadout` | Scalar motion magnitude | One positive-weight layer (size 5, no bias) |
| `MotionMagConv2Readout` | Scalar motion magnitude | Two layers (8 then 1 channels, size 5), positive weights in the last |

The hex-grid readouts subclass `HexGridReadout` (a base class, not listable
itself): per column they gather the firing rate of one neuron of each
`readout_neurons` cell type (absent types → NaN → zeroed) and pass the features
through `HexGridConv` layers with softplus between them. Flow readouts set
`predicts_flow` (used by `flow_loss` and the ego-motion solver) and
`eye_asymmetric` (their `flip_for_left_eye` negates the horizontal channel);
motion-magnitude readouts set `predicts_motion_mag` (used by `motion_mag`).
The hex-grid flow readouts also set `zero_init`: a newly built model zeroes
their last layer after the parameter-init noise (`zero_init_readouts`, with
any denominator channel's bias set so softplus gives 1), so training starts
from zero predicted flow (`flow_loss` = 1) instead of a large first-step
gradient that would otherwise sit in Adam's second-moment estimate.

Regularizers (`training_only = True`) go in the same `readouts` list. They are
evaluated once per `predict` (after the time loop, in training mode only) on
the model, and their scalar output is added to the training loss:

- `L1Reg` — L1 on the input/output strengths and the readout conv weights.
- `L2Reg` — L2 on the input/output strengths.
- `SigmaReg` — `relu(calc_sigma_max() - max_allowed_sigma) ** 2`.
- `DecoderWeightL1Reg` — not listed by name: built automatically (as
  `<readout>_DecoderWeightL1Reg`) for each hex-grid readout when
  `regularizer_parameters.input_layer_l1 > 0`; L1 on that readout's first conv
  layer, to encourage cell-type sparsity.

### CellTypeLayerNorm

Layer normalization where the learnable scale and bias are shared per cell type (not per neuron). Computes mean and variance within each cell type using scatter operations, then normalizes and applies the per-type affine transform. Scale is stored in log space to guarantee positivity. Applied after every RNN step when `use_layer_norm` is true.
