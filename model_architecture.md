# Model Architecture

Overview of the neural model classes.

## Module Layout

| Module | Contents |
|---|---|
| `neural_models.py` | `SerializableModel`, `BaseRnn`, `ConnectomeModel`, `compose_model_class` |
| `dynamics_models.py` | Dynamics mixins: `CurrentDynamics`, `ConductanceDynamics`, `CellAnnDynamics`, `HandTunedDynamics`, `ArbitraryDynamics` |
| `regions.py` | `RegionView` base + per-area subclasses (`OpticLobeRegion`, `AntennaLobeRegion`, …) and the area-name → region-class registry |
| `model_layers.py` | Reusable `nn.Module` building blocks: `HexGridConv`, `TemporalLRU`, `CellTypeLayerNorm` |
| `ego_motion.py` | Rotation math + velocity integration: `integrate_angular_velocity`, `integrate_translational_velocity` |
| `connectome_loading.py` | `ConnectomeData` dataclass + `load_connectome(areas, side)` |
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
│      ConnectomeData built with areas=['optic_lobe'] side='right'
│      → model.regions = {'optic_lobe': OpticLobeRegion(side='right', ...)}
│      → optic-lobe-specific methods live on that region object.
│
├── ConvModel                 Engineering hex-grid CNN (no dynamics)
│
├── Regions (regions.py)
│   ├── RegionView            Base: name + cell_indices + column_indices + side
│   ├── OpticLobeRegion       side='right'|'left'|'both' (eye-terminology alias num_eyes);
│   │                            predict_two_eyes, estimate_ego_motion_from_image,
│   │                            column_phi_theta_two_eyes
│   └── AntennaLobeRegion     (stub for future olfactory work)
│
└── Layers (model_layers.py)
    ├── HexGridConv           Hex-grid spatial convolution
    ├── TemporalLRU           Complex diagonal linear recurrence (temporal filter)
    └── CellTypeLayerNorm     Layer norm with per-cell-type weight/bias
```

## Configuration

A `ConnectomeModel` is parameterised by a flat areas list + a side filter,
plus the dynamics mixin to use:

```toml
[model]
model_type    = 'ConnectomeModel'
dynamics_type = 'ConductanceDynamics'
areas         = ['optic_lobe']         # any subset of region-registry keys
side          = 'right'                # 'right' | 'left' | 'both'
```

The loader (`connectome_loading.load_connectome(areas=..., side=...)`) filters
rows of the master parquet by these two fields, builds the
`ConnectomeData`, and returns it for the model constructor to consume. The
brain is bilaterally symmetric, so `side` is a model-wide filter that applies
to every bilateral region (optic lobe, antenna lobe, mushroom body, …) — not
just the eyes.

---

## BaseRnn (base class)

Abstract base class for connectome-constrained recurrent neural networks. Handles everything shared between synapse model variants: connectome ingestion, input processing, output decoding, simulation, serialization, and ego-motion estimation.

### Construction

All constructor arguments are keyword-only. The main data inputs are:

| Argument | Description |
|---|---|
| `synapse_counts_coo` | `(E, 3)` int64 array — `[pre_root_id, post_root_id, count]` per synapse |
| `root_ids` | `(N,)` int64 array of neuron root IDs |
| `cell_types` | `(N,)` string array of cell type labels |
| `neurotransmitters` | `(N,)` string array of neurotransmitter labels per neuron |
| `column_indices` | `(N,)` float array of column assignments (NaN = unassigned) |
| `column_phi_theta` | `(C, 2)` array of column angular positions (azimuth, polar) |
| `column_p_q` | `(C, 2)` int array of hex-lattice coordinates (needed for convolutions) |

Key configuration arguments:

| Argument | Default | Description |
|---|---|---|
| `excitatory_nt` | `['ACH']` | Neurotransmitter labels treated as excitatory |
| `inhibitory_nt` | `['GLUT', 'GABA']` | Neurotransmitter labels treated as inhibitory |
| `nonlinearity` | `'Softplus'` | Activation function (any `torch.nn` class name, or `'identity'`) |
| `dt` | `0.01` | Simulation timestep |
| `count_threshold` | `2` | Minimum synapse count to keep a connection |
| `readouts` | `[]` | List of readout class names: `'FlowNeuronReadout'`, `'FlowLinearReadout'`, `'FlowConv2Readout'`, `'MotionMagReadout'` |
| `use_input_conv` | `True` | Whether to apply spatial/temporal input filtering |
| `conv_size_x` | `5` | Hex-grid diameter for the input spatial convolution |
| `num_temporal_modes` | `4` | Number of complex modes in the input temporal filter |
| `use_layer_norm` | `True` | Whether to apply `CellTypeLayerNorm` after each RNN step |

### Connectome preprocessing (`_preprocess_connectome`)

At construction time the raw connectome is filtered in order:
1. Remove connections below `count_threshold`.
2. Remove neurons whose neurotransmitter is not in `excitatory_nt` or `inhibitory_nt`.
3. Remove connections involving removed neurons.
4. Remove neurons with no remaining connections.

Root IDs are then converted to dense matrix indices. The result is a sparse `(N, N)` synapse count matrix stored as `self.synapse_counts`.

### Input processing pipeline

When `use_input_conv=True`, raw columnar stimulus passes through two stages before entering the RNN:

1. **Spatial contrast normalization** (`HexGridConv` with positive, normalized weights) — computes a local weighted mean and variance over hex-grid neighbours, then normalizes: `y = -(x - mean) / sqrt(var + eps)`.
2. **Temporal filtering** (`TemporalLRU`) — a complex diagonal linear recurrence that applies learned bandpass filtering independently per column.

Both are applied per-timestep inside `apply_input_conv`.

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
| `estimate_ego_motion_from_flow(phi, theta, d_right, d_up)` | Solve for 6-DOF ego-motion from optic flow via least-squares rotation estimation followed by SVD-based translation estimation. |
| `model.regions['optic_lobe'].estimate_ego_motion_from_image(...)` | Two-eye ego-motion pipeline (lives on `OpticLobeRegion`): run both eyes through the model, decode optic flow, then solve for rotation and translation. |
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
