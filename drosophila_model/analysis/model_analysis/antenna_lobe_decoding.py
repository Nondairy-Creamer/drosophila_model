"""ORN vs PN odor-identity decoding from a CurrentDivisive AL model.

Runs an antenna-lobe-only ConnectomeModel with ``CurrentDivisiveDynamics``
against ``NUM_ODORS`` sparse non-negative odor patterns. Each odor is
presented ``NUM_TRIALS`` times at independent log-uniform concentrations
(spanning ``LOG_INT_HI - LOG_INT_LO`` decades) plus additive input noise.
Odor identity is decoded from type-averaged firing rates of (a) ORNs and
(b) PNs using multinomial logistic regression.

Outputs a 3-panel figure: overall held-out accuracy, accuracy binned by
log10 concentration, and per-cell-type concentration tuning for one demo
odor.

This script does not save the model or the activity tensors. It's a
single-shot analysis of how much odor-identity information survives the
ORN -> AL -> PN transformation under the divisive-normalisation dynamics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from drosophila_model.config import load_config
from drosophila_model.utilities import connectome_loading as cl
from drosophila_model.model_definition import neural_models as nm
from drosophila_model.model_definition import regions as regions_mod


# --- Run-time knobs ---------------------------------------------------------
NUM_ODORS = 1000
NUM_TRIALS = 10
# Sparse non-negative odor encoding: each odor activates this fraction of
# input channels with abs-Gaussian weights. Mirrors sparse glomerular
# activation in real ORN responses and keeps the input non-negative so
# the network's ReLU doesn't throw half of every pattern away.
INPUT_SPARSITY = 0.3
# Per-trial concentration is log-uniform across LOG_INT_LO..LOG_INT_HI
# decades. This is the variable PN divisive normalisation is supposed
# to compensate for — without it, ORN vs PN decoding is measured at a
# single operating point and div norm has little reason to help.
LOG_INT_LO = -3.0
LOG_INT_HI = 1.0
# Transduction-style noise on the stimulus *channels* (one channel per ORN
# type, shared across all cells of that type).
INPUT_NOISE_STD = 0.0
# Trial timeline:
#   [0, STIM_ONSET_S)               equilibration, stim = 0
#   [STIM_ONSET_S, STIM_DURATION_S) stim on (clean pattern + per-step noise)
STIM_ONSET_S = 0.25
STIM_DURATION_S = 0.75
DT = 0.01
# Decoder eval window: average firing rates over [EVAL_START_S, EVAL_END_S].
# The last 0.25s of the stim window soaks up transient + steady-state info.
EVAL_START_S = STIM_ONSET_S + 0.25
EVAL_END_S = STIM_DURATION_S
# Trials per predict() call. Logical batch dim is NUM_ODORS*NUM_TRIALS;
# we chunk it to keep the per-call activity tensor (B*T*N_neurons floats)
# bounded. Set to NUM_ODORS * NUM_TRIALS to run the whole set in a single
# predict() call (no chunking).
BATCH_SIZE = int(NUM_ODORS * NUM_TRIALS / 2)
# 50/50 train/test.
TEST_FRAC = 0.5
SEED = 0

# Inverse L2 regularization strength for the logistic decoder (sklearn's
# ``C``). Larger = weaker penalty; ``np.inf`` disables regularization
# entirely (the sklearn-1.8 recommended way, replacing ``penalty=None``).
LOGREG_C = 5.0

# Restrict both decoders to N glomeruli that have BOTH an ORN type and a
# PN type, matched 1-to-1 (column j of the ORN feature matrix and column j
# of the PN feature matrix are the same glomerulus). Fewer features ->
# lower accuracy and less overfitting. Set to ``None`` to decode from all
# available cell types (ORN and PN columns then need not correspond).
N_DECODE_GLOMERULI = 40

# Per-cell-type dynamics parameters. ``set_sigma_max`` is skipped and the
# scalar input/output strength overrides below replace the per-cell-type
# fan-in / fan-out init, giving a single global recurrent gain knob.
INIT_SIGMA = 1e-1
INIT_OFFSET = 0.0
INIT_INPUT_STRENGTH = 1e-3
INIT_OUTPUT_STRENGTH = 1.0
# Per-step state retention; v_next = p*v_prev + (1-p)*(...). Effective
# time constant tau ≈ -dt/ln(p).
INIT_PERSISTENCE = 0.75
# Stimulus enters the numerator of the divisive ratio, so it must be on
# the same scale as the inhibitory denominator. Pinned to 1.0 (the
# constructor default of ~10 swamps the divisive term).
INIT_STIMULUS_STRENGTH = 1.0

# Dynamics flavour for the AL model, passed straight to
# ``compose_model_class``. Options:
#   'CurrentDivisiveDynamics' -> excitation divided by inhibition
#                                (Carandini-Heeger normalisation).
#   'CurrentDynamics'         -> subtractive: inhibition enters as
#                                negative current.
#   'ConductanceDynamics'     -> biophysical conductance model
#                                (per-transmitter reversal potentials,
#                                leak conductance, capacitance, threshold).
# The current/divisive pair share the uniform-gain parameterisation pinned
# by ``initialize_parameters``; the conductance model has a different
# parameter set (no persistence/offset/sigma) and keeps its own
# biophysical defaults instead (see ``_CURRENT_FAMILY`` / main).
DYNAMICS_TYPE = 'CurrentDivisiveDynamics'

# Dynamics that share the current-style uniform-gain init pinned by
# ``initialize_parameters``. ConductanceDynamics is intentionally absent:
# its biophysical defaults are already sensible and on a different scale.
_CURRENT_FAMILY = ('CurrentDynamics', 'CurrentDivisiveDynamics')
# DYNAMICS_TYPE = 'CurrentDynamics'

# Minimum cells-per-type to include a PN-substring type in the readout.
# Region default is 5; lowering to 2 recovers the long tail of singleton
# / doubleton PN types so PN feature dim matches ORN's.
MIN_CELLS_PER_PN_TYPE = 5

# Device/dtype come from the local config (config.toml overlaying
# config_default.toml) via load_config(), so the GPU/CPU choice is set in
# one place — config['model']['device'] — rather than hard-coded here.
_CONFIG = load_config()
DEVICE = _CONFIG['model']['device']
DTYPE = _CONFIG['model']['dtype']

# Concentration-binned accuracy and demo concentration sweep.
N_CONC_BINS = 8
N_DEMO_CONCS = 50
DEMO_ODOR_IDX = 0
# log10 concentration used for the time-trace diagnostic figure. Picking
# a moderately high concentration so any divisive-norm saturation that
# is going to happen is in the fully-engaged regime.
TRACE_LOG_INT = 1.0

# Project root is two levels up from drosophila_model/analysis/model_analysis/.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = _PROJECT_ROOT / 'figures' / 'antenna_lobe_decoding'
OUT_PNG = OUT_DIR / 'decoding_accuracy.png'
OUT_PNG_TRAIN = OUT_DIR / 'decoding_accuracy_train.png'
OUT_PNG_TRACES = OUT_DIR / 'time_traces.png'

ORN_COLOR = '#d35'
PN_COLOR = '#3a9'


def build_model():
    """Compose ``DYNAMICS_TYPE`` + AntennaLobeRegion and return it."""
    data_folder = _CONFIG['data']['data_folder']

    region_specs = {
        'antenna_lobe': {
            'neuropils': regions_mod.AntennaLobeRegion.neuropils,
            'side': 'both',
        }
    }
    region_kwargs = {'antenna_lobe': {'side': 'both'}}

    connectome = cl.load_connectome(
        region_specs=region_specs,
        data_folder=data_folder,
        count_threshold=2,
    )

    ModelClass = nm.compose_model_class(DYNAMICS_TYPE)
    model = ModelClass(
        connectome=connectome,
        nonlinearity='ReLU',
        count_threshold=2,
        use_layer_norm=False,
        device=DEVICE,
        dtype=DTYPE,
        region_kwargs=region_kwargs,
        dt=DT,
        seed=SEED,
    )
    # Pure inference: freeze every parameter so autograd doesn't waste
    # memory tracking the per-trial unrolls. Mirrors what
    # BaseRnn.load(freeze_params=True) does on the load path.
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def expand_pn_output_types(model, min_cells=MIN_CELLS_PER_PN_TYPE):
    """Override ``region.default_output_neurons`` with every PN-substring
    cell type in the AL slice that has at least ``min_cells`` cells.

    The region's own default uses ``_MIN_CELLS_PER_TYPE = 5`` which
    drops most uniglomerular PNs (one per glomerulus) because they
    typically appear as singletons or doubletons in the connectome's
    upstream typing. Lowering to ``2`` recovers the full PN census so
    PN decoding has comparable feature dimensionality to ORN decoding.
    """
    region = model.regions['antenna_lobe']
    slice_types = model.cell_types[region.cell_indices.cpu().numpy()]
    unique_types, type_counts = np.unique(slice_types, return_counts=True)
    pn_types = sorted(
        str(t) for t, c in zip(unique_types, type_counts)
        if c >= min_cells and region._PN_SUBSTRING in str(t)
    )
    region.default_output_neurons = pn_types
    return pn_types


def initialize_parameters(
    model,
    sigma=INIT_SIGMA,
    offset=INIT_OFFSET,
    input_strength=INIT_INPUT_STRENGTH,
    output_strength=INIT_OUTPUT_STRENGTH,
    persistence=INIT_PERSISTENCE,
    stimulus_strength=INIT_STIMULUS_STRENGTH,
):
    """Pin all per-cell-type dynamics parameters to uniform values.

    Skips ``set_sigma_max`` and instead pins ``input_strength`` and
    ``output_strength`` to scalars so the effective recurrent operator
    is the signed connectome times ``input_strength * output_strength``
    — i.e. a clean global gain knob. Sigma, offset, persistence, and
    stimulus_strength are pinned to uniform values too. Must be called
    *before* ``find_steady_state``.

    ``sigma`` is only meaningful for the divisive dynamics (the divisive
    denominator's semi-saturation constant); the subtractive
    ``CurrentDynamics`` has no ``sigma_log`` parameter, so it's pinned only
    when present.
    """
    import math
    p = float(np.clip(persistence, 1e-6, 1 - 1e-6))
    persistence_logit = math.log(p / (1 - p))
    with torch.no_grad():
        # Guard each attribute: the current/divisive models share these
        # parameters, but other dynamics (e.g. ConductanceDynamics) omit
        # some of them, so only pin what the instance actually has.
        if hasattr(model, 'sigma_log'):
            model.sigma_log.fill_(math.log(sigma))
        if hasattr(model, 'offset'):
            model.offset.fill_(offset)
        if hasattr(model, 'persistence_logit'):
            model.persistence_logit.fill_(persistence_logit)
        model.input_strength_log.fill_(math.log(input_strength))
        model.output_strength_log.fill_(math.log(output_strength))
        model.stimulus_strength_log.fill_(math.log(stimulus_strength))


def make_odor_patterns(n_input_channels, rng):
    """Sparse non-negative encoding patterns, shape (NUM_ODORS, n_channels)."""
    patterns = np.abs(rng.standard_normal((NUM_ODORS, n_input_channels)))
    patterns *= (rng.random(patterns.shape) < INPUT_SPARSITY)
    # Patch all-zero odors so every odor activates >=1 channel — matches the
    # external make_encoding_matrix. Without this ~0.8% of odors (0.9**n_chan)
    # are empty and undecodable at any concentration, dragging accuracy down.
    empty = np.where(patterns.sum(axis=1) == 0)[0]
    for o in empty:
        j = rng.integers(n_input_channels)
        patterns[o, j] = float(abs(rng.standard_normal()))
    return patterns


def make_trials(odor_patterns, rng):
    """Return ``(trials, odor_id, log_intensity)`` — clean, no noise.

    Each presentation gets an independent log-uniform concentration that
    scales the pattern. Noise is *not* added here — it's injected per
    timestep inside :func:`build_inputs_t` so each step gets an
    independent draw, not a static per-trial offset broadcast across
    time.

    - ``trials`` shape: ``(NUM_ODORS * NUM_TRIALS, n_input_channels)`` —
      rows grouped by odor (NUM_TRIALS presentations of odor 0, then
      odor 1, ...).
    - ``odor_id`` shape: ``(NUM_ODORS * NUM_TRIALS,)`` — integer identity.
    - ``log_intensity`` shape: ``(NUM_ODORS * NUM_TRIALS,)`` — log10 of
      the per-trial concentration multiplier.
    """
    n_total = NUM_ODORS * NUM_TRIALS
    trials = np.repeat(odor_patterns, NUM_TRIALS, axis=0)
    log_intensity = rng.uniform(LOG_INT_LO, LOG_INT_HI, size=n_total)
    trials = trials * (10.0 ** log_intensity)[:, None]
    odor_id = np.repeat(np.arange(NUM_ODORS), NUM_TRIALS)
    return trials.astype(np.float32), odor_id, log_intensity


def build_inputs_t(clean_trials, num_time, onset_idx, noise_std,
                   device, dtype):
    """Build ``(B, T, n_channels)`` stimulus tensor with onset gating + noise.

    For ``t < onset_idx`` the input is exactly zero (equilibration).
    For ``t >= onset_idx`` the input is ``clean_trials`` plus a fresh
    Gaussian noise draw at every (trial, timestep, channel) — i.e.
    noise is *not* a single static per-trial offset broadcast across
    time. Pass ``noise_std=0`` for a deterministic clean trial.
    """
    bsz, n_channels = clean_trials.shape
    inputs_t = torch.zeros((bsz, num_time, n_channels), device=device, dtype=dtype)
    if onset_idx >= num_time:
        return inputs_t
    clean = torch.from_numpy(clean_trials).to(device=device, dtype=dtype)
    on_slice = inputs_t[:, onset_idx:, :]
    on_slice.copy_(clean.unsqueeze(1).expand_as(on_slice))
    if noise_std > 0:
        on_slice.add_(torch.randn_like(on_slice) * float(noise_std))
    return inputs_t


def make_demo_trials(odor_pattern):
    """Noise-free concentration sweep of a single odor.

    Returns ``(demo_trials, log_conc)`` with shapes
    ``(N_DEMO_CONCS, n_channels)`` and ``(N_DEMO_CONCS,)``.
    """
    log_conc = np.linspace(LOG_INT_LO, LOG_INT_HI, N_DEMO_CONCS)
    demo = odor_pattern[None, :] * (10.0 ** log_conc)[:, None]
    return demo.astype(np.float32), log_conc


@torch.no_grad()
def collect_time_traces(model, odor_pattern, log_int):
    """Run a single noise-free trial and return per-cell-type mean firing
    rate as a function of time, for ORN and PN populations.

    Returns ``(t, orn_traces, pn_traces, orn_types, pn_types)`` where:

    - ``t`` shape ``(num_time,)`` — time vector in seconds, from 0 to
      ``STIM_DURATION_S`` (includes the equilibration window).
    - ``orn_traces`` shape ``(num_time, n_orn_types)`` — mean firing rate
      across all cells of each ORN type, one column per type.
    - ``pn_traces`` shape ``(num_time, n_pn_types)`` — same for PNs.
    """
    region = model.regions['antenna_lobe']
    num_time = int(round(STIM_DURATION_S / DT))
    onset_idx = int(round(STIM_ONSET_S / DT))

    present_types = set(model.unique_cell_types)
    record_types = [
        t for t in (
            list(region.default_input_neurons)
            + list(region.default_output_neurons)
        )
        if t in present_types
    ]

    clean = (odor_pattern[None, :] * (10.0 ** log_int)).astype(np.float32)
    inputs_t = build_inputs_t(
        clean, num_time=num_time, onset_idx=onset_idx, noise_std=0.0,
        device=model.device, dtype=model.dtype,
    )

    result = model.predict(inputs_t, output_neurons=record_types)
    fr_dict = result['outputs']['antenna_lobe']['firing_rate']

    orn_types = [t for t in region.default_input_neurons if t in fr_dict]
    pn_types = [t for t in region.default_output_neurons if t in fr_dict]

    orn_traces = np.stack(
        [fr_dict[ct].mean(dim=(0, 2)).detach().cpu().numpy()
         for ct in orn_types],
        axis=1,
    )
    pn_traces = np.stack(
        [fr_dict[ct].mean(dim=(0, 2)).detach().cpu().numpy()
         for ct in pn_types],
        axis=1,
    )
    t = np.arange(num_time) * DT
    return t, orn_traces, pn_traces, orn_types, pn_types


@torch.no_grad()
def collect_type_averaged_activity(model, trials, noise_std=0.0):
    """Run ``predict`` in chunks and return per-trial features.

    ``trials`` is the *clean* (noise-free) intensity-scaled stimulus per
    presentation, shape ``(N_total, n_input_channels)``. The full
    ``(B, T, n_channels)`` input tensor is built per chunk by
    :func:`build_inputs_t`, which zeros the equilibration window and
    adds an independent ``noise_std`` Gaussian draw at every
    ``(trial, timestep, channel)`` after onset.

    Per-trial features are mean firing rates over the eval window and
    over all cells of each type — one scalar per (trial, cell type).

    Returns:
        orn_features: ``(N_total, n_orn_types)``
        pn_features:  ``(N_total, n_pn_types)``
        orn_types, pn_types: cell-type name lists matching the last axis.
    """
    region = model.regions['antenna_lobe']
    num_time = int(round(STIM_DURATION_S / DT))
    onset_idx = int(round(STIM_ONSET_S / DT))
    eval_start = int(round(EVAL_START_S / DT))
    eval_end = int(round(EVAL_END_S / DT))

    # By default, predict() only records cell types in
    # ``default_output_neurons`` (the PNs). Opt the ORNs into the output
    # recording explicitly so we can decode from them too.
    present_types = set(model.unique_cell_types)
    record_types = [
        t for t in (
            list(region.default_input_neurons)
            + list(region.default_output_neurons)
        )
        if t in present_types
    ]

    n_total = trials.shape[0]

    orn_per_chunk = []
    pn_per_chunk = []
    orn_types = None
    pn_types = None
    cached_input_mapping = None
    cached_output_mapping = None

    for start in range(0, n_total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_total)

        chunk_np = trials[start:end]  # (bsz, n_channels), clean
        inputs_t = build_inputs_t(
            chunk_np, num_time=num_time, onset_idx=onset_idx,
            noise_std=noise_std,
            device=model.device, dtype=model.dtype,
        )
        bsz = inputs_t.shape[0]

        # First chunk: pass the combined ORN+PN type list so both get
        # recorded. Subsequent chunks reuse the cached mapping.
        out_arg = cached_output_mapping if cached_output_mapping is not None else record_types
        result = model.predict(
            inputs_t,
            input_neurons=cached_input_mapping,
            output_neurons=out_arg,
        )
        if cached_input_mapping is None:
            cached_input_mapping = result['mappings']['antenna_lobe']['input_mapping']
            cached_output_mapping = {
                'antenna_lobe': result['mappings']['antenna_lobe']['output_mapping']
            }

        fr_dict = result['outputs']['antenna_lobe']['firing_rate']

        if orn_types is None:
            orn_types = [t for t in region.default_input_neurons if t in fr_dict]
            pn_types = [t for t in region.default_output_neurons if t in fr_dict]

        # Per-trial feature: mean over post-transient time (dim=1) and
        # over all cells of that type (dim=2) -> one scalar per
        # (trial, type). One state vector per odor presentation, never
        # individual timesteps.
        orn_chunk = np.empty((bsz, len(orn_types)), dtype=np.float32)
        for j, ct in enumerate(orn_types):
            orn_chunk[:, j] = (
                fr_dict[ct][:, eval_start:eval_end, :]
                .mean(dim=(1, 2))
                .detach().cpu().numpy()
            )

        pn_chunk = np.empty((bsz, len(pn_types)), dtype=np.float32)
        for j, ct in enumerate(pn_types):
            pn_chunk[:, j] = (
                fr_dict[ct][:, eval_start:eval_end, :]
                .mean(dim=(1, 2))
                .detach().cpu().numpy()
            )

        orn_per_chunk.append(orn_chunk)
        pn_per_chunk.append(pn_chunk)

    orn_features = np.concatenate(orn_per_chunk, axis=0)
    pn_features = np.concatenate(pn_per_chunk, axis=0)
    return orn_features, pn_features, orn_types, pn_types


def _glomerulus_maps(model, orn_types, pn_types):
    """Parse glomerulus labels from ORN/PN type names.

    ORN types are named ``ORN_<glom>`` (so ``ORN_DA1`` -> glomerulus
    ``DA1``); PN types embed the glomerulus as the prefix before the first
    underscore (``DA1_lPN`` -> ``DA1``, ``VA1d_adPN`` -> ``VA1d``,
    ``M_vPNml53`` -> ``M`` for multiglomerular PNs).

    Returns ``(orn_by_glom, pn_by_glom, count_of)`` where ``orn_by_glom``
    maps glomerulus -> ORN type, ``pn_by_glom`` maps glomerulus -> list of
    PN types, and ``count_of`` maps cell-type name -> cell count.
    """
    region = model.regions['antenna_lobe']
    slice_types = model.cell_types[region.cell_indices.cpu().numpy()]
    uniq, cnts = np.unique(slice_types, return_counts=True)
    count_of = {str(t): int(c) for t, c in zip(uniq, cnts)}

    orn_by_glom = {
        t[len('ORN_'):]: t for t in orn_types if t.startswith('ORN_')
    }
    pn_by_glom: dict[str, list[str]] = {}
    for t in pn_types:
        pn_by_glom.setdefault(t.split('_', 1)[0], []).append(t)
    return orn_by_glom, pn_by_glom, count_of


def match_orn_pn_by_glomerulus(model, orn_types, pn_types, n_glomeruli):
    """Pair ORN and PN feature columns that share a glomerulus.

    A glomerulus is matchable when both an ORN type and at least one PN
    type exist for it. When a glomerulus has multiple PN types, the one
    with the most cells is chosen — the largest PN population for that
    glomerulus. See :func:`_glomerulus_maps` for the name-parsing rules.

    ``orn_types`` / ``pn_types`` are the feature-column label lists (in
    column order) returned by :func:`collect_type_averaged_activity`.

    Returns ``(glomeruli, orn_sel, pn_sel)`` truncated to the first
    ``n_glomeruli`` shared glomeruli (sorted by name); ``orn_sel[j]`` and
    ``pn_sel[j]`` are the ORN and PN cell-type names for ``glomeruli[j]``.
    Pass ``n_glomeruli=None`` to keep every shared glomerulus.
    """
    orn_by_glom, pn_by_glom, count_of = _glomerulus_maps(
        model, orn_types, pn_types,
    )

    glomeruli = sorted(set(orn_by_glom) & set(pn_by_glom))
    if n_glomeruli is not None:
        glomeruli = glomeruli[:n_glomeruli]

    orn_sel = [orn_by_glom[g] for g in glomeruli]
    pn_sel = [
        max(pn_by_glom[g], key=lambda t: count_of.get(t, 0))
        for g in glomeruli
    ]
    return glomeruli, orn_sel, pn_sel


def report_orn_pn_matching(model, orn_types, pn_types):
    """Print every matched ORN<->PN glomerulus pair, then the ORN types
    and PN types that have no partner in the other population.

    Uses the same name-parsing as :func:`match_orn_pn_by_glomerulus` so the
    report reflects exactly what the matcher would pair (no truncation).
    """
    orn_by_glom, pn_by_glom, count_of = _glomerulus_maps(
        model, orn_types, pn_types,
    )
    matched = sorted(set(orn_by_glom) & set(pn_by_glom))
    orn_only = sorted(set(orn_by_glom) - set(pn_by_glom))
    pn_only = sorted(set(pn_by_glom) - set(orn_by_glom))

    print(f"\nMatched ORN<->PN pairs ({len(matched)} glomeruli):")
    for g in matched:
        orn = orn_by_glom[g]
        pns = pn_by_glom[g]
        best = max(pns, key=lambda t: count_of.get(t, 0))
        extra = (f"  [also {[p for p in pns if p != best]}]"
                 if len(pns) > 1 else '')
        print(f"  {g:6s}  {orn} (n={count_of.get(orn)})  <->  "
              f"{best} (n={count_of.get(best)}){extra}")

    print(f"\nUnmatched ORN types (no PN in this slice) "
          f"({len(orn_only)}):")
    for g in orn_only:
        print(f"  {orn_by_glom[g]} (n={count_of.get(orn_by_glom[g])})")

    n_pn_types = sum(len(pn_by_glom[g]) for g in pn_only)
    print(f"\nUnmatched PN types (no ORN in this slice) "
          f"({n_pn_types} types across {len(pn_only)} glomerulus labels):")
    for g in pn_only:
        for p in sorted(pn_by_glom[g]):
            print(f"  {p} (n={count_of.get(p)})  [glom={g}]")


def decode(features, odor_id, name):
    """Train/test split, fit logistic regression.

    ``features`` shape: ``(N_total, n_features)`` where N_total = NUM_ODORS
    * NUM_TRIALS. Stratified split keeps each odor-identity class evenly
    represented in train and test.

    Returns a dict with both splits' accuracy, predictions, original-array
    indices, and ground-truth labels::

        {
            'train_acc', 'train_pred', 'train_idx', 'y_train',
            'test_acc',  'test_pred',  'test_idx',  'y_test',
        }

    The ``*_idx`` arrays index into the original ``features`` / ``odor_id``
    arrays so callers can align other per-trial metadata (e.g.
    log-intensity) with predictions for either split.
    """
    idx = np.arange(len(odor_id))
    idx_train, idx_test = train_test_split(
        idx, test_size=TEST_FRAC, stratify=odor_id, random_state=SEED,
    )
    X_train, X_test = features[idx_train], features[idx_test]
    y_train, y_test = odor_id[idx_train], odor_id[idx_test]

    # Multinomial logistic regression. Features are expected to be
    # pre-scaled by the caller (each population divided by a single global
    # STD; see main) so ORN and PN sit on equal footing without per-feature
    # (per-glomerulus) standardization. sklearn 1.7 dropped multi_class
    # (lbfgs is multinomial by default); 1.8 dropped n_jobs.
    clf = LogisticRegression(C=LOGREG_C, max_iter=1000, solver='lbfgs')
    clf.fit(X_train, y_train)
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    train_acc = float((train_pred == y_train).mean())
    test_acc = float((test_pred == y_test).mean())
    print(f"  {name}: train accuracy = {train_acc:.4f}, "
          f"test accuracy = {test_acc:.4f} "
          f"(features: {features.shape[1]}, train: {len(y_train)}, "
          f"test: {len(y_test)})")
    return {
        'train_acc': train_acc, 'train_pred': train_pred,
        'train_idx': idx_train, 'y_train': y_train,
        'test_acc': test_acc, 'test_pred': test_pred,
        'test_idx': idx_test, 'y_test': y_test,
    }


def accuracy_by_log_intensity(pred, y_test, log_intensity_test):
    """Bin test trials by log10 concentration and compute per-bin accuracy."""
    edges = np.linspace(LOG_INT_LO, LOG_INT_HI, N_CONC_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip(
        np.digitize(log_intensity_test, edges) - 1, 0, N_CONC_BINS - 1,
    )
    per_bin = np.array([
        (pred[bin_idx == b] == y_test[bin_idx == b]).mean()
        if (bin_idx == b).any() else np.nan
        for b in range(N_CONC_BINS)
    ])
    return centers, per_bin


def plot_results(
    orn_acc, pn_acc,
    bin_centers, orn_per_bin, pn_per_bin,
    demo_log_conc, demo_orn, demo_pn,
    n_orn_types, n_pn_types,
    out_path,
    split_label='test',
):
    """Three-panel figure: overall accuracy, accuracy-vs-concentration,
    and per-type concentration tuning for one demo odor.

    ``split_label`` ('train' or 'test') only affects titles/suptitle; the
    same function renders either split so train and test figures stay
    visually identical apart from the label.
    """
    chance = 1.0 / NUM_ODORS
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: overall accuracy bar.
    bars = axes[0].bar(
        ['ORN', 'PN'], [orn_acc, pn_acc],
        color=[ORN_COLOR, PN_COLOR],
    )
    for b, v in zip(bars, [orn_acc, pn_acc]):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 0.005,
                     f"{v:.3f}", ha='center', va='bottom', fontsize=10)
    axes[0].axhline(chance, color='grey', ls='--', lw=1,
                    label=f'chance = {chance:.3f}')
    axes[0].set_ylabel('classification accuracy')
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title(f'Overall identity decoding ({split_label})')
    axes[0].legend(loc='upper right', fontsize=9)

    # Panel 2: accuracy vs log10 concentration.
    axes[1].plot(bin_centers, orn_per_bin, 'o-',
                 color=ORN_COLOR, label='ORN')
    axes[1].plot(bin_centers, pn_per_bin, 'o-',
                 color=PN_COLOR, label='PN')
    axes[1].axhline(chance, color='grey', ls='--', lw=1)
    axes[1].set_xlabel(r'$\log_{10}$ concentration')
    axes[1].set_ylabel('accuracy')
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title(f'Accuracy vs concentration ({split_label})')
    axes[1].legend(loc='lower right', fontsize=9)

    # Panel 3: per-type response of one odor across concentrations.
    demo_conc = 10.0 ** demo_log_conc
    axes[2].plot(demo_conc, demo_orn, color=ORN_COLOR, alpha=0.4, lw=0.7)
    axes[2].plot(demo_conc, demo_pn, color=PN_COLOR, alpha=0.4, lw=0.7)
    axes[2].set_xscale('log')
    axes[2].set_xlabel('concentration')
    axes[2].set_ylabel('mean firing rate per cell type')
    axes[2].set_title(
        f'Odor {DEMO_ODOR_IDX}: one line per cell type\n'
        f'red=ORN ({n_orn_types}), teal=PN ({n_pn_types})'
    )

    fig.suptitle(
        f'AL {DYNAMICS_TYPE} [{split_label}]: '
        f'{NUM_ODORS} odors x {NUM_TRIALS} trials, '
        f'noise={INPUT_NOISE_STD}, sparsity={INPUT_SPARSITY}, '
        f'log10 conc in [{LOG_INT_LO}, {LOG_INT_HI}]',
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure to {out_path}")


def plot_time_traces(
    t, orn_traces, pn_traces, eval_start_s, eval_end_s,
    log_int, demo_odor_idx, out_path,
):
    """Two-panel figure: per-cell-type firing rate vs time, ORN and PN.

    Each line is one cell type (mean across all cells of that type for
    the single noise-free trial). The decoder's eval window is shaded
    so it is visible alongside the rest of the response.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    axes[0].plot(t, orn_traces, color=ORN_COLOR, alpha=0.4, lw=0.8)
    axes[0].axvspan(eval_start_s, eval_end_s, color='grey', alpha=0.15,
                    label=f'decoder window [{eval_start_s:g}, {eval_end_s:g}]s')
    axes[0].axvline(STIM_ONSET_S, color='k', ls=':', lw=1,
                    label=f'stim onset ({STIM_ONSET_S:g}s)')
    axes[0].set_ylabel('mean firing rate per cell type')
    axes[0].set_xlabel('time (s)')
    axes[0].set_title(f'ORN time traces ({orn_traces.shape[1]} types)')
    axes[0].legend(loc='upper right', fontsize=9)

    axes[1].plot(t, pn_traces, color=PN_COLOR, alpha=0.4, lw=0.8)
    axes[1].axvspan(eval_start_s, eval_end_s, color='grey', alpha=0.15)
    axes[1].axvline(STIM_ONSET_S, color='k', ls=':', lw=1)
    axes[1].set_xlabel('time (s)')
    axes[1].set_title(f'PN time traces ({pn_traces.shape[1]} types)')

    fig.suptitle(
        f'Odor {demo_odor_idx} at log10(conc)={log_int:g} '
        f'(conc={10.0**log_int:g}), noise-free',
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved time-trace figure to {out_path}")


def main():
    rng = np.random.default_rng(SEED)
    print(f"Building AL-only {DYNAMICS_TYPE} model ...")
    model = build_model()
    region = model.regions['antenna_lobe']
    pn_types = expand_pn_output_types(model)
    print(f"  Expanded PN readout (min_cells={MIN_CELLS_PER_PN_TYPE}): "
          f"{len(pn_types)} types")
    n_input_channels = region.input_stim_size
    print(f"  num_neurons = {model.num_neurons}, "
          f"num_input_channels (glomeruli) = {n_input_channels}, "
          f"num_input_types = {len(region.default_input_neurons)}, "
          f"num_output_types (PN) = {len(region.default_output_neurons)}")

    if DYNAMICS_TYPE in _CURRENT_FAMILY:
        print(f"Initializing parameters (uniform): "
              f"input_strength={INIT_INPUT_STRENGTH}, "
              f"output_strength={INIT_OUTPUT_STRENGTH}, "
              f"sigma={INIT_SIGMA}, offset={INIT_OFFSET}, "
              f"persistence={INIT_PERSISTENCE} "
              f"(skipping set_sigma_max)")
        initialize_parameters(model)
    else:
        # ConductanceDynamics: keep the biophysical defaults (reversal
        # potentials, leak conductance, capacitance, threshold), which are
        # already spectral-normalised via set_sigma_max in its __init__ and
        # live on a different scale than the current-style gain knobs.
        print(f"Using {DYNAMICS_TYPE} built-in biophysical init "
              f"(no uniform parameter pinning)")

    print("Finding steady state ...")
    model.find_steady_state()

    print(f"Generating {NUM_ODORS} odors x {NUM_TRIALS} trials "
          f"(sparsity={INPUT_SPARSITY}, "
          f"log10 conc in [{LOG_INT_LO}, {LOG_INT_HI}], "
          f"noise std={INPUT_NOISE_STD}) ...")
    odor_patterns = make_odor_patterns(n_input_channels, rng)
    trials, odor_id, log_intensity = make_trials(odor_patterns, rng)
    print(f"  trials shape: {trials.shape}  "
          f"(NUM_ODORS*NUM_TRIALS, n_input_channels)")

    print(f"Running predict() in chunks of {BATCH_SIZE} "
          f"({trials.shape[0]} total trials, {STIM_DURATION_S}s each, "
          f"stim onset at {STIM_ONSET_S}s, "
          f"channel noise std={INPUT_NOISE_STD}) ...")
    model.eval()
    torch.manual_seed(SEED)
    orn_features, pn_features, orn_types, pn_types = (
        collect_type_averaged_activity(model, trials, noise_std=INPUT_NOISE_STD)
    )
    print(f"  ORN features shape: {orn_features.shape} "
          f"(types: {len(orn_types)})")
    print(f"  PN features shape:  {pn_features.shape} "
          f"(types: {len(pn_types)})")

    report_orn_pn_matching(model, orn_types, pn_types)

    # Optionally restrict both decoders to N glomeruli that have both an
    # ORN and a PN type, matched 1-to-1 by glomerulus so ORN column j and
    # PN column j refer to the same glomerulus. The simulation still runs
    # the full network; we just hand each decoder a matched subset of the
    # type-averaged readouts.
    if N_DECODE_GLOMERULI is not None:
        gloms, orn_sel, pn_sel = match_orn_pn_by_glomerulus(
            model, orn_types, pn_types, N_DECODE_GLOMERULI,
        )
        orn_cols = [orn_types.index(t) for t in orn_sel]
        pn_cols = [pn_types.index(t) for t in pn_sel]
        orn_features = orn_features[:, orn_cols]
        pn_features = pn_features[:, pn_cols]
        print(f"Restricting decoder to {len(gloms)} ORN<->PN matched "
              f"glomeruli: {gloms}")
        print(f"  ORN cols: {orn_sel}")
        print(f"  PN  cols: {pn_sel}")

    # Put ORN and PN on equal footing with a single per-population scale:
    # divide each whole feature matrix by one global STD (not per feature),
    # so the lower-magnitude population isn't disadvantaged by the L2
    # penalty while relative magnitudes *across* glomeruli are preserved.
    orn_scale = float(orn_features.std())
    pn_scale = float(pn_features.std())
    orn_features = orn_features / orn_scale
    pn_features = pn_features / pn_scale
    print(f"Per-population scaling: ORN /= {orn_scale:.4g}, "
          f"PN /= {pn_scale:.4g} (single global STD each)")

    print("Decoding odor identity ...")
    orn_res = decode(orn_features, odor_id, name='ORN type-averaged')
    pn_res = decode(pn_features, odor_id, name='PN  type-averaged')

    # Per-bin accuracy for both splits. Each split uses its own split's
    # predictions, ground-truth labels, and the log-intensity rows the
    # split's indices point at.
    bin_centers, orn_test_per_bin = accuracy_by_log_intensity(
        orn_res['test_pred'], orn_res['y_test'],
        log_intensity[orn_res['test_idx']],
    )
    _, pn_test_per_bin = accuracy_by_log_intensity(
        pn_res['test_pred'], pn_res['y_test'],
        log_intensity[pn_res['test_idx']],
    )
    _, orn_train_per_bin = accuracy_by_log_intensity(
        orn_res['train_pred'], orn_res['y_train'],
        log_intensity[orn_res['train_idx']],
    )
    _, pn_train_per_bin = accuracy_by_log_intensity(
        pn_res['train_pred'], pn_res['y_train'],
        log_intensity[pn_res['train_idx']],
    )

    print(f"Running concentration sweep for demo odor {DEMO_ODOR_IDX} "
          f"({N_DEMO_CONCS} concentrations, noise-free) ...")
    demo_trials, demo_log_conc = make_demo_trials(odor_patterns[DEMO_ODOR_IDX])
    demo_orn, demo_pn, _, _ = collect_type_averaged_activity(model, demo_trials)

    plot_results(
        orn_res['test_acc'], pn_res['test_acc'],
        bin_centers, orn_test_per_bin, pn_test_per_bin,
        demo_log_conc, demo_orn, demo_pn,
        len(orn_types), len(pn_types),
        OUT_PNG, split_label='test',
    )
    plot_results(
        orn_res['train_acc'], pn_res['train_acc'],
        bin_centers, orn_train_per_bin, pn_train_per_bin,
        demo_log_conc, demo_orn, demo_pn,
        len(orn_types), len(pn_types),
        OUT_PNG_TRAIN, split_label='train',
    )

    print(f"Recording time traces for demo odor {DEMO_ODOR_IDX} "
          f"at log10(conc)={TRACE_LOG_INT} ...")
    t, orn_traces, pn_traces, _, _ = collect_time_traces(
        model, odor_patterns[DEMO_ODOR_IDX], TRACE_LOG_INT,
    )
    plot_time_traces(
        t, orn_traces, pn_traces, EVAL_START_S, EVAL_END_S,
        TRACE_LOG_INT, DEMO_ODOR_IDX, OUT_PNG_TRACES,
    )


if __name__ == '__main__':
    main()
