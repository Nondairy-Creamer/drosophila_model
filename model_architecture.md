# Model Architecture

Overview of the neural model classes.

## Module Layout

| Module | Contents |
|---|---|
| `neural_models.py` | `SerializableModel`, `BaseRnn`, `ConnectomeModel`, `compose_model_class` |
| `dynamics_models.py` | Dynamics mixins: `CurrentDynamics`, `ConductanceDynamics`, `CellAnnDynamics`, `HandTunedDynamics`, `ArbitraryDynamics` |
| `regions.py` | `RegionView` base + per-area subclasses (`OpticLobeRegion`, `AntennaLobeRegion`, …) and the area-name → region-class registry |
| `model_layers.py` | Reusable `nn.Module` building blocks: `HexGridConv`, `TemporalLRU`, `CellTypeLayerNorm` |
| `ego_motion.py` | Flow -> 6-DOF solver (`estimate_from_flow`), rotation math, velocity integration |
| `connectome_loading.py` | `ConnectomeData` dataclass + `load_connectome(region_specs)` |
| `engineering_models.py` | `ConvModel` (stateless hex-grid CNN, sister to `BaseRnn`) |

## Class Hierarchy

```
torch.nn.Module
├── BaseRnn (ABC)             Connectome-agnostic RNN scaffolding
│   └── ConnectomeModel       Loads any combination of areas; owns self.regions
│
│  Concrete models are composed at runtime by
│      compose_model_class(dynamics_type)
│  e.g. CurrentDynamics + ConnectomeModel → ConnectomeModel_CurrentDynamics
│
│  Anatomical scope is data-driven, not class-driven:
│      ConnectomeData built with
│          region_specs={'optic_lobe': {'neuropils': ['ME','LO','LOP','LA'],
│                                       'side': 'right'}}
│      → model.regions = {'optic_lobe': OpticLobeRegion(side='right', ...)}
│      → optic-lobe-specific methods live on that region object.
│      Region membership is union-by-neuropil: a neuron belongs to the region
│      if any of its synapses land in any of the region's resolved neuropils.
│
├── ConvModel                 Engineering hex-grid CNN (no dynamics)
│
├── Regions (regions.py, all nn.Module so submodules land in state_dict)
│   ├── RegionView            Base: name + cell_indices + side; weakref to model
│   ├── OpticLobeRegion       Owns ALL vision-specific state and methods:
│   │                            column_phi_theta / column_p_q / column_indices /
│   │                            num_columns / column_phi_theta_two_eyes,
│   │                            input_pre_conv / input_spatial_conv / temporal_lru
│   │                            (built when region_opts['use_input_conv']=true),
│   │                            plus its own config (use_input_conv, conv_size_x,
│   │                            input_pre_conv_size, num_temporal_modes),
│   │                            apply_input_conv, get_shared_columns, get_column_xyz,
│   │                            estimate_ego_motion_from_flow, predict_two_eyes,
│   │                            estimate_ego_motion_from_image.
│   │                          side='right'|'left'|'both' (eye-terminology alias num_eyes).
│   └── AntennaLobeRegion     (stub for future olfactory work)
│
└── Layers (model_layers.py)
    ├── HexGridConv           Hex-grid spatial convolution
    ├── TemporalLRU           Complex diagonal linear recurrence (temporal filter)
    └── CellTypeLayerNorm     Layer norm with per-cell-type weight/bias
```

## Configuration

A `ConnectomeModel` is parameterised by the set of `[model.regions.<area>]`
sub-tables (one per loaded area) plus the dynamics mixin. Each region owns
its own `side` filter (`'right'` / `'left'` / `'both'`); other keys in the
sub-table are region-private (input-conv sizes, spatial filter widths, etc.):

```toml
[model]
model_type    = 'ConnectomeModel'
dynamics_type = 'ConductanceDynamics'
readouts      = ['FlowConv2Readout']   # user owns: only list visual readouts
                                        # when 'optic_lobe' is in regions

# Per-region option sub-table. The `side` key is pulled out by build_regions
# and used as the per-area connectome filter; everything else is forwarded
# verbatim to OpticLobeRegion._load_extras as `opts`.
[model.regions.optic_lobe]
side                = 'right'          # 'right' | 'left' | 'both'
use_input_conv      = true
conv_size_x         = 7
input_pre_conv_size = 5
num_temporal_modes  = 4
```

The loader (`connectome_loading.load_connectome(region_specs=...)`) takes a
`{region_name: {'neuropils': [...], 'side': 'right'|'left'|'both'}}` mapping
and pulls the union of neurons that have at least one synapse in any of each
region's resolved neuropils. Each region carries its own side, so different
regions can come from different hemispheres in the same model. Bilateral
neuropil base names (e.g. `'ME'`) are expanded to `'ME_R'` / `'ME_L'` per
`side`; names that already end in `_L` / `_R` (or are central, like `'EB'`)
pass through unchanged.

`region_specs` is built at the call site from `config['model']['regions']` via
`regions.build_region_specs(...)`: that pulls the region class's default
`neuropils` list (overridable per-instance via
`config['model']['regions'][<name>]['neuropils']`) and the configured `side`.

`config['model']['regions']` is also passed to `ConnectomeModel.__init__` as
`region_kwargs={...}` and stored on the model so it round-trips through
save/load. `config_default.toml` only ships defaults for `optic_lobe`; configs
that introduce new areas (e.g. antenna lobe) supply their own region sub-table.

## Data layout

The connectome bundle is baked once by `format_connectome_data.py` and lives
under `config['data']['data_folder']`:

```
<data_folder>/
├── connectome_data/
│   ├── raw/                          # raw CSVs consumed by format_connectome_data.py
│   │   ├── consolidated_cell_types.csv    # root_id, primary_type, additional_type(s)
│   │   ├── connections_no_threshold.csv   # pre_root_id, post_root_id, neuropil, syn_count, nt_type
│   │   └── column_assignments_no_threshold.csv   # optic-lobe-only column geometry
│   └── baked/
│       ├── cell_info.parquet         # emitted by format_connectome_data.py
│       │     columns: root_id, primary_type, neurotransmitter, neuropils
│       │     - neuropils: list[str] of every neuropil this neuron has at
│       │       least one synapse in (union; pre + post)
│       │     - neurotransmitter: per-cell-type dominant pre-synaptic NT
│       │       (photoreceptors forced to 'HIS'; may be NaN for cell types
│       │       that never appear pre-synaptically)
│       ├── synapses.parquet          # emitted by format_connectome_data.py
│       │     columns: pre_root_id, post_root_id, synapse_counts, nt_type, neuropil
│       │     - neuropil is kept so the loader can drop edges that don't
│       │       land in any loaded region's neuropil set
│       ├── optic_lobe_column_index.parquet  # emitted by format_connectome_data.py
│       │     columns: root_id, column_index (0-indexed)
│       │     # loaded by OpticLobeRegion._load_extras
│       └── optic_lobe_columns.parquet  # sourced from a separate empirical dataset;
│             columns: column_index, phi, theta, p, q, x, y, z
│             # loaded by OpticLobeRegion._load_extras
└── stimulus/                         # stimulus datasets (panoramic images, drone rigs, etc.)
    ├── raw/                          # source data + process_raw_data.py caches
    │   ├── all_panoramic.pt          # cache from panorama_*.mat (process_raw_data.py random_pano)
    │   ├── panoramic/                # raw panorama_*.mat source images
    │   ├── small_drone.pt            # sampled-clip cache (process_raw_data.py small_drone)
    │   ├── rpi_drone.pt              # sampled-clip cache (process_raw_data.py rpi_drone)
    │   ├── fib_fly_view_dataset.pkl  # raw rendered fly-view source (responses/flow/depth)
    │   ├── drone_videos/             # small_drone equirectangular rig (left_N/right_N.mp4)
    │   ├── drone_poses/              # small_drone batch-opt pose .npz
    │   └── rpi_drone/                # rpi_drone dome rig recordings + dome_calib/
    └── baked/                        # fixed training datasets from create_training_data.py
        ├── small_drone.pt            # one <stimulus_type>.pt per baked type
        ├── rpi_drone.pt
        └── flyview.pt                # cache from fib_fly_view_dataset.pkl (process_raw_data.py
                                      # flyview): flipped flow + depth-based odometry, float32, no depth
```

Universal fields land on `ConnectomeData`; per-region union-membership masks
land in `region_membership` (one bool array per region, aligned with
`root_ids`). Per-region geometry / per-neuron lookups (today: optic-lobe
column index and hex geometry) are loaded by the corresponding `RegionView`
subclass from `connectome_data/baked/`, so the model itself stays
anatomy-agnostic. All visual-specific state and operations
(`column_phi_theta`, `column_p_q`, `num_columns`, `column_indices_per_neuron`,
the input-conv stack — `input_pre_conv` / `input_spatial_conv` / `temporal_lru`
—, `apply_input_conv`, `get_shared_columns`, `get_column_xyz`,
`estimate_ego_motion_from_flow`, `predict_two_eyes`,
`estimate_ego_motion_from_image`) live on `model.regions['optic_lobe']` and
are accessed through it.

To add an area (e.g. antenna lobe):
1. Subclass `RegionView` in `regions.py`, set its `neuropils = [...]` (bilateral
   base names; the loader appends `_L` / `_R` per `side`), override
   `_load_extras` to read any per-area files (and emit them from
   `format_connectome_data.py` if needed), and register it in
   `REGION_REGISTRY`. No change to the universal `cell_info.parquet` /
   `synapses.parquet` is required — region membership is computed at load
   time from the neuropil column.

---

## BaseRnn (base class)

Abstract base class for connectome-constrained recurrent neural networks. Handles everything shared between synapse model variants: connectome ingestion, input processing, output decoding, simulation, serialization, and ego-motion estimation.

### Construction

`ConnectomeModel` takes a single `connectome: ConnectomeData` (built via
`connectome_loading.load_connectome(region_specs)`). The dataclass holds:

| Field | Description |
|---|---|
| `synapse_counts_coo` | `(E, 3)` int64 array — `[pre_root_id, post_root_id, count]` per synapse |
| `root_ids` | `(N,)` int64 array of neuron root IDs |
| `cell_types` | `(N,)` string array of primary cell-type labels |
| `neurotransmitters` | `(N,)` string array of neurotransmitter labels per neuron |
| `region_membership` | dict `{region_name: (N,) bool}` — true where the neuron has at least one synapse in any of that region's resolved neuropils (union; the same neuron can belong to multiple regions but only contributes one row to `root_ids`) |
| `region_specs` | input filter: `{region_name: {neuropils, side}}` actually loaded |
| `regions` | property: `list(region_specs.keys())` |
| `data_folder` | path used by `RegionView._load_extras` to find `connectome_data/baked/` |

All visual-column accessors and helpers (`column_phi_theta`, `column_p_q`,
`num_columns`, `column_indices_per_neuron`, `get_shared_columns`,
`get_column_xyz`) live on the optic-lobe region and are accessed as
`model.regions['optic_lobe'].<attr>`. The model itself owns only the
universal `region_membership` masks.

Key model-level configuration arguments:

| Argument | Default | Description |
|---|---|---|
| `excitatory_nt` | `['ACH']` | Neurotransmitter labels treated as excitatory |
| `inhibitory_nt` | `['GLUT', 'GABA']` | Neurotransmitter labels treated as inhibitory |
| `nonlinearity` | `'Softplus'` | Activation function (any `torch.nn` class name, or `'identity'`) |
| `dt` | `0.01` | Simulation timestep |
| `count_threshold` | `2` | Minimum synapse count to keep a connection |
| `readouts` | `[]` | List of readout class names: `'FlowNeuronReadout'`, `'FlowLinearReadout'`, `'FlowConv2Readout'`, `'MotionMagReadout'` |
| `region_kwargs` | `{}` | Per-region option sub-dict: `{<area>: {...}}`. Wired up from `config['model']['regions']`. |
| `use_layer_norm` | `True` | Whether to apply `CellTypeLayerNorm` after each RNN step |

Region-owned configuration arguments (sit in `region_kwargs['<area>']`):

| Region | Key | Default | Description |
|---|---|---|---|
| `optic_lobe` | `use_input_conv` | `True` | Build + apply the input-conv stack (HexGridConv + contrast norm + TemporalLRU) |
| `optic_lobe` | `conv_size_x` | `7` | Hex-grid diameter for the contrast-norm spatial conv |
| `optic_lobe` | `input_pre_conv_size` | `5` | Hex-grid diameter for the pre-filter conv |
| `optic_lobe` | `num_temporal_modes` | `4` | Complex modes in the temporal LRU |

### Connectome preprocessing (`_preprocess_connectome`)

At construction time the raw connectome is filtered in order:
1. Remove connections below `count_threshold`.
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

Root IDs are then converted to dense matrix indices. The result is a sparse `(N, N)` synapse count matrix stored as `self.synapse_counts`.

### Input processing pipeline

When `region_kwargs['optic_lobe']['use_input_conv'] = true`, raw columnar
stimulus passes through two stages
before entering the RNN. The entire pipeline is owned by
`OpticLobeRegion` — the model invokes
`self.regions['optic_lobe'].apply_input_conv(u_t, filter_state)` inside its
`predict` loop:

1. **Spatial contrast normalization** (`HexGridConv` with positive, normalized weights) — computes a local weighted mean and variance over hex-grid neighbours, then normalizes: `y = -(x - mean) / sqrt(var + eps)`.
2. **Temporal filtering** (`TemporalLRU`) — a complex diagonal linear recurrence that applies learned bandpass filtering independently per column.

Submodules live at `model.regions['optic_lobe'].input_pre_conv`,
`...input_spatial_conv`, and `...temporal_lru`; their parameters land in
`state_dict()` under the `regions.optic_lobe.*` prefix.

### Abstract methods

Subclasses must implement:

- **`forward(v_prev, u_t, stim_to_cell_mapping)`** — one-step RNN update. Takes the previous voltage state `v_prev` `(B, N)` and external input `u_t`, returns the next voltage `v_next` `(B, N)`.
- **`firing_rate(v)`** — maps voltage to a non-negative firing rate.

### Key methods

| Method | Description |
|---|---|
| `predict(inputs, ...)` | Run the full simulation loop. Returns per-cell-type voltage/firing-rate traces, optional decoder output, and final state. |
| `find_steady_state(...)` | Iterate the model with zero input until state converges; stores result in `self.steady_state`. |
| `save(path)` / `load(path)` | Serialize/deserialize the model (constructor args + state dict). `load` re-composes the concrete class from the saved `dynamics_type` string via `compose_model_class`. |
| `silence_neurons(names)` | Zero the input and output strength of specified cell types (for lesion experiments). |
| `get_shared_columns(neuron_list)` | Find columns that contain all specified cell types; returns neuron and column index arrays. |
| `regions['optic_lobe'].estimate_ego_motion_from_flow(phi, theta, d_right, d_up)` | Solve for 6-DOF ego-motion from one eye's optic flow (zero offset) via least-squares rotation estimation followed by SVD-based translation estimation; translation up to scale. Wraps `ego_motion.estimate_from_flow` (which takes a list of `EyeFlow`, each with a body-frame camera `offset`). |
| `model.regions['optic_lobe'].estimate_ego_motion_from_image(...)` | Two-eye ego-motion pipeline (lives on `OpticLobeRegion`): run both eyes through the model, decode optic flow, then solve rotation + translation. Each eye carries its body-frame camera offset (loaded from `model_definition/eye_geometry.json`, mirrored per eye), so the rotation-induced inter-eye parallax recovers absolute translational scale, not just direction. |
| `model.regions['optic_lobe'].predict_two_eyes(inputs_right, inputs_left, ...)` | Per-eye `predict()` results; mirror-trick on single-eye hosts, native slice on two-eye hosts. |

### `predict` in detail

`predict` is the main simulation entry point. Given `inputs` of shape `(batch, time, columns)`:

1. Maps columnar stimulus to specific neuron indices via `stim_to_cell_mapping` (built from `input_neurons` list or provided directly).
2. Initializes voltage from `self.steady_state` (or a provided `init_state`).
3. For each timestep:
   - Optionally applies `apply_input_conv` (spatial + temporal filtering).
   - Calls `self.forward(v, u_t, ...)` to get the next voltage.
   - Optionally applies `self.layer_norm`.
   - Computes firing rates via `self.firing_rate`.
   - Optionally runs `self.decoder` on the firing rates to produce optic flow.
   - Records voltage and firing rate for each readout cell type.
4. Returns `(outputs, final_state, mappings)` where `outputs` contains per-cell-type voltage and firing-rate tensors of shape `(batch, time, columns)`.

---

## CurrentModel

Current-based synapse model. Synaptic input is modeled as injected current proportional to presynaptic firing rate and (signed) synapse count.

### Dynamics

```
z = D_in @ synapse_counts @ D_out @ f(v_prev) + b * u + offset
v_next = α * v_prev + (1 - α) * z
```

where:
- `f` is the nonlinearity (default softplus)
- `α = sigmoid(persistence_logit)` — per-cell-type persistence (how much voltage carries over)
- `D_in`, `D_out` — diagonal scaling matrices (per-cell-type, learned in log space)
- `b` — per-cell-type stimulus strength (learned in log space)
- `offset` — per-cell-type bias

Synapse counts are signed at construction: excitatory neurotransmitters get positive sign, inhibitory get negative.

### Learned parameters

All parameters are **per cell type** (not per neuron), expanded to per-neuron via `neuron_to_type_index`:

| Parameter | Shape | Constraint | Role |
|---|---|---|---|
| `persistence_logit` | `(T,)` | sigmoid → (0,1) | Voltage decay rate |
| `input_strength_log` | `(T,)` | exp → positive | Post-synaptic gain |
| `output_strength_log` | `(T,)` | exp → positive | Pre-synaptic gain |
| `stimulus_strength_log` | `(T,)` | exp → positive | External input gain |
| `offset` | `(T,)` | unconstrained | Resting drive |

where `T` = number of unique cell types.

### Initialization

- `input_strength` and `output_strength` are initialized to `fan_in^(-1/2)` and `fan_out^(-1/2)` respectively, so that the variance of synaptic currents is approximately normalized.
- After parameter initialization, `set_sigma_max` iteratively rescales `output_strength` until the spectral norm of the linearized dynamics matrix is near `max_allowed_sigma` (default 0.90), keeping the network near the edge of stability.

---

## ConductanceModel

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

Key differences from `CurrentModel`:
- Synapse counts are stored as **separate sparse matrices per neurotransmitter** (not signed), since each transmitter drives toward its own reversal potential.
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

- Reversal potentials default to biologically motivated values: ACH → +10 mV, GLUT/GABA → -70 mV, leak → -50 mV.
- `input_strength` and `output_strength` are initialized from fan-in/fan-out as in `CurrentModel`, but computed per-transmitter for `input_strength`.
- Steady state is initialized to the leak reversal potential (rather than zero).

---

## Helper Modules

### HexGridConv

Spatial convolution on a hex grid. Given per-column features, gathers features from neighbouring columns (determined by p,q lattice offsets), applies a learned weight tensor, and adds a bias. Boundary columns and absent neuron types (NaN features) are handled gracefully.

Used for: input spatial filtering (with `positive_weights=True` for weighted averaging) and decoder layers.

### TemporalLRU

Temporal filter implemented as a complex diagonal linear recurrence. Each of `num_modes` independent modes is a damped oscillator parameterized by:
- `gamma` (damping, constrained positive via `log_gamma`)
- `omega` (frequency in [0, π], constrained via `logit_omega`)

Provides natural bandpass filtering with learnable frequency response. Called per-timestep via `step(u_t, state)`.

### MotionDecoder

Multi-layer hex-grid convolutional decoder that maps firing rates to 2D optic flow per column. Architecture:
- Maps neuron firing rates to per-column features using a precomputed `column_to_decoder_neurons` index.
- Passes features through N conv layers (`HexGridConv`), with BatchNorm + softplus between all but the last.
- Final layer outputs 3 channels: the first two are divided by softplus of the third to produce the 2D flow vector.

The decoder reads from a fixed set of medulla/lobula cell types (T1–T5, Tm, TmY families).

### CellTypeLayerNorm

Layer normalization where the learnable scale and bias are shared per cell type (not per neuron). Computes mean and variance within each cell type using scatter operations, then normalizes and applies the per-type affine transform. Scale is stored in log space to guarantee positivity.
