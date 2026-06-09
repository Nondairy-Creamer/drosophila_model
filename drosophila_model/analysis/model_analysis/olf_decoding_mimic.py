"""Mimic of OlfactionConnectomeModel `concentration_divnorm.py`.

Faithful port of the external `concentration_divnorm.py` concentration /
divisive-normalisation experiment, with **one change**: the antennal-lobe
dynamics are produced by antenna-lobe :class:`ConnectomeModel` instances built
from :mod:`drosophila_model.model_definition.neural_models` instead of the
reference's hand-signed ``W`` matrix + ``BatchedALSim`` simulator.

What the experiment does
------------------------
Each odor is a sparse non-negative pattern over the connectome's *matched*
glomeruli (those with both an ORN type and a uniglomerular ALPN type). Each
trial draws a log-uniform concentration spanning ~3 decades. Four
representations are decoded for odor identity:

  1. Static ORN drive       = ReLU(E @ c + noise)            (olfactory.py style)
  2. Static PN (div norm)    = ORN / (sigma + mean(ORN))      (olfactory.py style)
  3. Connectome ORN/ALPN with **subtractive** inhibition  (classical LN model)
  4. Connectome ORN/ALPN with **divisive**   inhibition  (Carandini-Heeger)

The two connectome modes map onto the model's dynamics flavours:

  * subtractive -> :class:`dynamics.CurrentDynamics`
        v_{t+1} = p v_t + (1-p)(gain * W_signed.T @ ReLU(v) + I_ext)
  * divisive    -> :class:`dynamics.CurrentDivisiveDynamics`
        v_{t+1} = p v_t + (1-p)(gain_exc*(W+ @ r) + I_ext) / (sigma + gain_inh*(W- @ r))

with ``p = 1 - dt/tau`` and a global recurrent gain. The reference's divisive
mode uses ``gain_inh`` (0.1) much larger than ``gain_exc`` (1e-3) so the
inhibitory pool engages the normaliser; ``CurrentDivisiveDynamics`` ties both
gains to ``input_strength``, so we reproduce the separate ``gain_inh`` by
scaling the model's inhibitory split (``synapse_counts_neg``) by
``gain_inh/gain_exc``.

Architectural adaptation (unavoidable): the reference injects a *per-neuron*
external input; the model drives input *per ORN cell type* (one channel per
ORN glomerulus). Odors therefore drive the matched glomeruli's ORN channels,
background noise is injected on the ORN input channels, and connectome
representations are per-glomerulus averages of the recorded ORN / PN firing
rates at the readout time.

A 2x2 figure is written under ``figures/olf_decoding_mimic/`` and the
connectome simulation is cached to disk.

Run with::

    python -m drosophila_model.analysis.model_analysis.olf_decoding_mimic
"""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from drosophila_model.config import load_config
from drosophila_model.model_definition import connectome_loading as cl
from drosophila_model.model_definition import neural_models as nm
from drosophila_model.model_definition import regions as regions_mod


# ===========================================================================
# Experiment parameters (mirroring concentration_divnorm.py)
# ===========================================================================

# ---- odor stimulus design (olfactory.py-style) ----------------------------
N_ODORS = 1000
N_TRIALS = 6000                  # ~6 trials/odor at random concentrations
SPARSITY = 0.1                   # fraction of gloms each odor activates
NOISE_STD_STATIC = 0.1 * 0.1     # additive noise on the static glomerular drive
LOG_INT_LO = -1.5                # log10 concentration range
LOG_INT_HI = 1.5
NORM_SIGMA = 0.1                 # pooled divisive-normalisation saturation

# ---- connectome simulator (match the reference dynamics) ------------------
SIM_TAU = 0.05
GAIN_EXC = 1e-3
SIM_DT = 2e-3
SIM_T = 0.7                      # trial duration (s)
SIM_T_ON = 0.10                  # odor onset (s)
SIM_T_OFF = 0.60                 # odor offset (s)
STIMULUS_STRENGTH = 1.0          # I_ext enters the recurrence with unit weight

# Single readout time at the plateau midpoint, so connectome features are
# apples-to-apples with the static one-shot drive vectors. Set N_READ_BINS > 1
# to average over [SIM_T_READ_LO, SIM_T_READ_HI] instead.
N_READ_BINS = 1
SIM_T_READ_LO = 0.45
SIM_T_READ_HI = 0.55
BG_NOISE_SD = 0.10               # background noise on every ORN input channel
SIG_NOISE_SD = 0.0               # signal-gated transduction noise (off, as ref)

# Inhibition modes. ``dynamics`` selects the model flavour; ``gain_inh`` is the
# inhibitory gain (subtractive: equal to GAIN_EXC; divisive: larger, reproduced
# by scaling the model's inhibitory split by gain_inh/GAIN_EXC); ``sigma`` is
# the divisive semi-saturation constant.
INH_MODES = {
    'subtractive': {'dynamics': 'CurrentDynamics',
                    'gain_inh': 1e-3, 'sigma': 1.0},
    'divisive':    {'dynamics': 'CurrentDivisiveDynamics',
                    'gain_inh': 1e-1, 'sigma': 1.0},
}
MODES_TO_RUN = ['subtractive', 'divisive']

# ---- decoding -------------------------------------------------------------
DECODER_C = 1.0
TRAIN_FRAC = 0.5

# Batch size for model.predict. Smaller than the reference's 200 because the
# model records full (B, T, n_cells) buffers for every output type per call.
MODEL_BATCH_SIZE = 100

DEVICE = 'cpu'
DTYPE = 'float32'
SEED = 0

# ---- demo concentration sweep ---------------------------------------------
M_DEMO_CONCS = 40
K_DEMO_REPEATS = 3

# Population colours (matching the reference convention: colour = population,
# linestyle / hatch = inhibition mode).
COL_STATIC_ORN = '#d35'
COL_STATIC_PN = '#3a9'
COL_CNX_ORN = '#1f77b4'
COL_CNX_ALPN = '#ff7f0e'
MODE_STYLE = {
    'subtractive': {'hatch': '', 'linestyle': '-', 'marker': 'o'},
    'divisive':    {'hatch': '//', 'linestyle': '--', 'marker': '^'},
}

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = _PROJECT_ROOT / 'figures' / 'olf_decoding_mimic'
CACHE_DIR = OUT_DIR / 'cache'
OUT_PNG = OUT_DIR / 'concentration_divnorm.png'


# ===========================================================================
# Model construction (replaces the reference W matrix + BatchedALSim)
# ===========================================================================


def build_models():
    """Load the AL connectome once and build one model per inhibition mode.

    Returns ``(models, region, lookups)`` where ``models`` maps mode name ->
    composed ConnectomeModel, ``region`` is the (shared-geometry) antenna-lobe
    region of the first model, and ``lookups`` is the tuple returned by
    :func:`glom_lookups`.
    """
    config = load_config()
    data_folder = config['data']['data_folder']
    region_specs = {
        'antenna_lobe': {
            'neuropils': regions_mod.AntennaLobeRegion.neuropils,
            'side': 'both',
        }
    }
    region_kwargs = {'antenna_lobe': {'side': 'both'}}
    connectome = cl.load_connectome(
        region_specs=region_specs, data_folder=data_folder,
    )

    models = {}
    for mode in MODES_TO_RUN:
        ModelClass = nm.compose_model_class(INH_MODES[mode]['dynamics'])
        model = ModelClass(
            connectome=connectome,
            nonlinearity='ReLU',
            count_threshold=2,
            use_layer_norm=False,
            device=DEVICE,
            dtype=DTYPE,
            region_kwargs=region_kwargs,
            dt=SIM_DT,
            seed=SEED,
            init_new_model=False,   # pin the gain ourselves, skip set_sigma_max
        )
        for param in model.parameters():
            param.requires_grad_(False)
        model.eval()
        models[mode] = model

    first = models[MODES_TO_RUN[0]]
    return models, first.regions['antenna_lobe'], glom_lookups(first)


def glom_lookups(model):
    """Return ORN/ALPN channel info and matched-glomerulus structures.

    - ``orn_types``: ordered list of ORN input channels (one per ORN
      glomerulus type present after preprocessing); defines odor-vector and
      stimulus channel order.
    - ``pn_types``: ordered list of recorded projection-neuron types.
    - ``matched_gloms``: glomeruli that have both an ORN type and >=1 PN type
      (sorted), the shared feature axis for every representation.
    - ``orn_type_of_glom``: ``{glom: 'ORN_<glom>'}``.
    - ``pn_types_of_glom``: ``{glom: [PN types whose prefix == glom]}``.
    """
    region = model.regions['antenna_lobe']
    present = set(model.unique_cell_types)
    orn_types = [t for t in region.default_input_neurons if t in present]
    pn_types = [t for t in region.default_output_neurons if t in present]

    prefix = regions_mod.AntennaLobeRegion._ORN_PREFIX
    orn_type_of_glom = {t[len(prefix):]: t for t in orn_types}
    pn_types_of_glom: dict[str, list[str]] = {}
    for pt in pn_types:
        pn_types_of_glom.setdefault(pt.split('_', 1)[0], []).append(pt)

    matched_gloms = sorted(set(orn_type_of_glom) & set(pn_types_of_glom))
    return (orn_types, pn_types, matched_gloms,
            orn_type_of_glom, pn_types_of_glom)


def set_dynamics(model, mode, dt=SIM_DT, tau=SIM_TAU, gain_exc=GAIN_EXC,
                 stimulus_strength=STIMULUS_STRENGTH):
    """Pin per-cell-type dynamics parameters to uniform scalar values.

    ``input_strength * output_strength = gain_exc`` makes the recurrent
    operator the signed connectome times a global gain; ``persistence =
    1 - dt/tau`` reproduces the leaky-integrator step. For divisive dynamics
    the inhibitory split is scaled by ``gain_inh / gain_exc`` so the
    denominator's inhibitory gain matches the reference (which the tied-gain
    model cannot otherwise express), and ``sigma`` is set from the mode config.
    """
    cfg = INH_MODES[mode]
    persistence = float(np.clip(1.0 - dt / tau, 1e-6, 1 - 1e-6))
    persistence_logit = math.log(persistence / (1 - persistence))
    with torch.no_grad():
        model.offset.fill_(0.0)
        model.input_strength_log.fill_(math.log(gain_exc))
        model.output_strength_log.fill_(math.log(1.0))
        model.persistence_logit.fill_(persistence_logit)
        model.stimulus_strength_log.fill_(math.log(stimulus_strength))
        if hasattr(model, 'sigma_log'):
            model.sigma_log.fill_(math.log(cfg['sigma']))
        if hasattr(model, 'synapse_counts_neg'):
            _scale_inhibition(model, cfg['gain_inh'] / gain_exc)
    model.find_steady_state()


def _scale_inhibition(model, factor):
    """Scale the divisive model's inhibitory split (``W-``) by ``factor``.

    Reproduces a ``gain_inh`` independent of ``gain_exc``: with
    ``input_strength = gain_exc``, the divisive denominator term is
    ``input_strength * (factor * W-) @ r = gain_inh * W- @ r``.
    """
    sc = model.synapse_counts_neg
    model.synapse_counts_neg = torch.sparse_csr_tensor(
        sc.crow_indices(), sc.col_indices(), sc.values() * float(factor),
        size=sc.size(),
    )


# ===========================================================================
# Static representations + helpers (olfactory.py-style)
# ===========================================================================


def make_encoding_matrix(rng, n_odors, n_gloms, sparsity):
    """Sparse non-negative E (odor x glom); no all-zero rows."""
    E = np.abs(rng.normal(size=(n_odors, n_gloms))).astype(np.float32)
    E *= (rng.random(size=E.shape) < sparsity)
    empty = np.where(E.sum(axis=1) == 0)[0]
    for o in empty:
        E[o, rng.integers(n_gloms)] = float(abs(rng.normal()))
    return E


def divisive_norm(x, sigma):
    """Pooled divisive normalisation: x / (sigma + mean(x))."""
    return x / (sigma + x.mean(axis=1, keepdims=True))


def fit_predict(X_train, y_train, X_test, C=DECODER_C, max_iter=300):
    pipe = Pipeline([
        #('scaler', StandardScaler()),
        ('lr', LogisticRegression(C=C, max_iter=max_iter, solver='lbfgs')),
    ])
    pipe.fit(X_train, y_train)
    return pipe.predict(X_test)


def experiment_hash(*items, n_bytes=4):
    h = hashlib.blake2b(digest_size=n_bytes)
    for it in items:
        if isinstance(it, np.ndarray):
            h.update(np.ascontiguousarray(it).tobytes())
        else:
            h.update(repr(it).encode())
    return h.hexdigest()


def network_hash(model, sim_kwargs, n_bytes=4):
    h = hashlib.blake2b(digest_size=n_bytes)
    h.update(model.synapse_counts.values().detach().cpu().numpy().tobytes())
    h.update(repr(sorted(sim_kwargs.items())).encode())
    return h.hexdigest()


# ===========================================================================
# Connectome simulation: drive the AL with per-trial glom intensity vectors
# ===========================================================================


def build_channel_input(batch_intensities, n_steps, dt, t_on, t_off,
                        bg_noise_sd, signal_noise_sd, seeds):
    """Per-ORN-channel external input for a batch, shape (B, n_steps, n_chan).

    ``batch_intensities`` is ``(B, n_chan)`` -- per-trial drive in each ORN
    channel (zero on channels of unmatched / inactive glomeruli). Background
    noise is added to every channel; transduction noise is dwell-gated and
    applied only on active channels.
    """
    B, n_chan = batch_intensities.shape
    ss = np.random.SeedSequence(np.asarray(seeds, dtype=np.int64).tolist())
    rng = np.random.default_rng(ss)

    I = rng.standard_normal((B, n_steps, n_chan)).astype(np.float32) * bg_noise_sd
    t_axis = np.arange(n_steps, dtype=np.float32) * dt
    envelope = ((t_axis >= t_on) & (t_axis < t_off)).astype(np.float32)
    I += (envelope[None, :, None] * batch_intensities[:, None, :]).astype(np.float32)
    if signal_noise_sd > 0:
        active = (batch_intensities > 0).astype(np.float32)
        sig = rng.standard_normal((B, n_steps, n_chan)).astype(np.float32)
        sig *= signal_noise_sd
        sig *= envelope[None, :, None]
        sig *= active[:, None, :]
        I += sig
    return I


@torch.no_grad()
def simulate_per_glom(model, region, intensities_channels, seeds, read_steps,
                      orn_types, pn_types, matched_gloms,
                      orn_type_of_glom, pn_types_of_glom,
                      batch_size=MODEL_BATCH_SIZE, label=None):
    """Run all trials through ``model.predict``; return per-glom ORN/ALPN rates.

    Returns ``(orn_per_glom, alpn_per_glom)`` each ``(n_trials, n_matched)`` --
    the mean firing rate over the readout steps and over the cells of each
    matched glomerulus (ORN type for ORN, all matched PN types for ALPN).
    """
    n_trials = intensities_channels.shape[0]
    n_steps = int(round(SIM_T / SIM_DT))
    n_matched = len(matched_gloms)
    record_types = list(orn_types) + list(pn_types)
    read_t = torch.as_tensor(np.asarray(read_steps), dtype=torch.long,
                             device=model.device)

    orn_out = np.zeros((n_trials, n_matched), dtype=np.float32)
    alpn_out = np.zeros((n_trials, n_matched), dtype=np.float32)
    cached_in = cached_out = None
    t0 = time.time()
    for start in range(0, n_trials, batch_size):
        stop = min(start + batch_size, n_trials)
        I_ext = build_channel_input(
            intensities_channels[start:stop], n_steps, SIM_DT,
            SIM_T_ON, SIM_T_OFF, BG_NOISE_SD, SIG_NOISE_SD, seeds[start:stop],
        )
        chunk = torch.as_tensor(I_ext, device=model.device, dtype=model.dtype)
        out_arg = cached_out if cached_out is not None else record_types
        result = model.predict(
            chunk, input_neurons=cached_in, output_neurons=out_arg,
        )
        if cached_in is None:
            cached_in = result['mappings']['antenna_lobe']['input_mapping']
            cached_out = {
                'antenna_lobe':
                    result['mappings']['antenna_lobe']['output_mapping']
            }
        fr = result['outputs']['antenna_lobe']['firing_rate']
        # Mean over readout steps -> (B, n_cells) per recorded type.
        read = {ct: fr[ct][:, read_t, :].mean(dim=1).detach().cpu().numpy()
                for ct in fr}
        for j, g in enumerate(matched_gloms):
            ot = orn_type_of_glom[g]
            if ot in read:
                orn_out[start:stop, j] = read[ot].mean(axis=1)
            pn_cols = [read[pt] for pt in pn_types_of_glom[g] if pt in read]
            if pn_cols:
                alpn_out[start:stop, j] = np.concatenate(pn_cols, axis=1).mean(axis=1)
        if label is not None:
            print(f"  {label}: {stop}/{n_trials} trials ({time.time() - t0:.0f}s)")
    return orn_out, alpn_out


def run_or_load_connectome(model, region, mode, intensities_channels, seeds,
                           read_steps, lookups, exp_hash):
    """Per-mode connectome simulation with disk caching."""
    orn_types, pn_types, matched_gloms, orn_type_of_glom, pn_types_of_glom = lookups
    cfg = INH_MODES[mode]
    sim_kwargs = dict(dt=SIM_DT, tau=SIM_TAU, gain_exc=GAIN_EXC,
                      gain_inh=cfg['gain_inh'], sigma=cfg['sigma'],
                      dynamics=cfg['dynamics'])
    net_hash = network_hash(model, sim_kwargs)
    cache_path = CACHE_DIR / (
        f"divnorm_n{N_ODORS}_tr{N_TRIALS}_sp{SPARSITY}"
        f"_log{LOG_INT_LO}_{LOG_INT_HI}_T{SIM_T}_dt{int(SIM_DT * 1000)}ms"
        f"_bg{BG_NOISE_SD:g}_sn{SIG_NOISE_SD:g}"
        f"_inh{mode}_gi{cfg['gain_inh']:g}_sig{cfg['sigma']:g}"
        f"_net{net_hash}_exp{exp_hash}.npz"
    )
    if cache_path.exists():
        print(f"  [{mode}] loading cached sim: {cache_path}")
        cached = np.load(cache_path)
        return cached['connectome_orn'], cached['connectome_alpn']

    print(f"  [{mode}] running connectome simulation -> {cache_path}")
    print(f"    dynamics={cfg['dynamics']}  gain_exc={GAIN_EXC:g}  "
          f"gain_inh={cfg['gain_inh']:g}  sigma={cfg['sigma']:g}")
    set_dynamics(model, mode)
    c_orn, c_alpn = simulate_per_glom(
        model, region, intensities_channels, seeds, read_steps,
        orn_types, pn_types, matched_gloms, orn_type_of_glom, pn_types_of_glom,
        label=f'{mode} trials',
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, connectome_orn=c_orn, connectome_alpn=c_alpn)
    print(f"    saved cache to {cache_path}")
    return c_orn, c_alpn


# ===========================================================================
# Main
# ===========================================================================


def main():
    torch.manual_seed(SEED)
    print("Building antenna-lobe models (subtractive + divisive)...")
    models, region, lookups = build_models()
    orn_types, pn_types, matched_gloms, orn_type_of_glom, pn_types_of_glom = lookups
    n_gloms = len(matched_gloms)
    print(f"  AL cells: {models[MODES_TO_RUN[0]].num_neurons} | "
          f"ORN channels: {len(orn_types)} | PN types: {len(pn_types)} | "
          f"matched gloms (ORN & ALPN): {n_gloms}")

    # ---- experiment design ------------------------------------------------
    rng = np.random.default_rng(SEED)
    E = make_encoding_matrix(rng, N_ODORS, n_gloms, SPARSITY)
    trial_odor = rng.integers(0, N_ODORS, size=N_TRIALS)
    trial_log_conc = rng.uniform(LOG_INT_LO, LOG_INT_HI, size=N_TRIALS).astype(np.float32)
    trial_conc = (10.0 ** trial_log_conc).astype(np.float32)
    trial_intensities = (E[trial_odor] * trial_conc[:, None]).astype(np.float32)
    trial_seeds = rng.integers(2 ** 31 - 1, size=N_TRIALS).astype(np.int64)
    print(f"  N_ODORS={N_ODORS} | N_TRIALS={N_TRIALS} | sparsity={SPARSITY} | "
          f"conc range = 10^[{LOG_INT_LO}, {LOG_INT_HI}]")

    # ---- static representations (olfactory.py-style) ----------------------
    static_drive = trial_intensities + (
        rng.normal(size=trial_intensities.shape).astype(np.float32) * NOISE_STD_STATIC
    )
    static_orn = np.clip(static_drive, 0.0, None)
    static_pn = divisive_norm(static_orn, NORM_SIGMA)

    # ---- map matched-glom intensities onto ORN input channels -------------
    channel_of_matched = np.array(
        [orn_types.index(orn_type_of_glom[g]) for g in matched_gloms], dtype=int,
    )
    intensities_channels = np.zeros((N_TRIALS, len(orn_types)), dtype=np.float32)
    intensities_channels[:, channel_of_matched] = trial_intensities

    # ---- readout steps ----------------------------------------------------
    n_steps_sim = int(round(SIM_T / SIM_DT))
    t_read = np.linspace(SIM_T_READ_LO, SIM_T_READ_HI, N_READ_BINS)
    read_steps = np.clip(np.round(t_read / SIM_DT).astype(int), 0, n_steps_sim - 1)
    print(f"\nReadout window: t = {t_read.round(3).tolist()} s "
          f"(steps {read_steps.tolist()})")

    exp_hash = experiment_hash(
        E, trial_odor, trial_log_conc, trial_seeds,
        SIM_T, SIM_T_ON, SIM_T_OFF, SIM_T_READ_LO, SIM_T_READ_HI,
        N_READ_BINS, BG_NOISE_SD, SIG_NOISE_SD, matched_gloms,
    )

    # ---- connectome simulation per inhibition mode (cached) ---------------
    connectome = {}
    for mode in MODES_TO_RUN:
        c_orn, c_alpn = run_or_load_connectome(
            models[mode], region, mode, intensities_channels, trial_seeds,
            read_steps, lookups, exp_hash,
        )
        connectome[mode] = {'orn': c_orn, 'alpn': c_alpn}

    # ---- assemble representations -----------------------------------------
    representations = [
        {'name': 'Static ORN (raw drive)', 'short': 'Static\nORN',
         'X': static_orn, 'color': COL_STATIC_ORN,
         'hatch': '', 'linestyle': '-', 'marker': 'o'},
        {'name': f'Static PN (div norm, sigma={NORM_SIGMA:.2f})',
         'short': 'Static PN\n(div norm)',
         'X': static_pn, 'color': COL_STATIC_PN,
         'hatch': '', 'linestyle': '-', 'marker': 'o'},
    ]
    for mode in MODES_TO_RUN:
        cfg = INH_MODES[mode]
        tag = (f"{mode} inh, gi={cfg['gain_inh']:g}"
               + (f", sigma={cfg['sigma']:g}" if mode == 'divisive' else ""))
        st = MODE_STYLE[mode]
        representations.append({
            'name': f"Connectome ORN ({tag})", 'short': f"Cnx ORN\n({mode})",
            'X': connectome[mode]['orn'], 'color': COL_CNX_ORN, **st,
        })
        representations.append({
            'name': f"Connectome ALPN ({tag})", 'short': f"Cnx ALPN\n({mode})",
            'X': connectome[mode]['alpn'], 'color': COL_CNX_ALPN, **st,
        })

    # ---- train/test split + decode ---------------------------------------
    n_train = int(TRAIN_FRAC * N_TRIALS)
    idx = rng.permutation(N_TRIALS)
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    y_true = trial_odor[test_idx]
    test_log_int = trial_log_conc[test_idx]

    print(f"\nDecoding ({N_ODORS}-way odor identity, {n_train} train / "
          f"{N_TRIALS - n_train} test, C={DECODER_C}):")
    accs, preds = {}, {}
    for rep in representations:
        pred = fit_predict(rep['X'][train_idx], trial_odor[train_idx], rep['X'][test_idx])
        preds[rep['name']] = pred
        accs[rep['name']] = float((pred == y_true).mean())
        print(f"  {rep['name']:<60s}  acc = {accs[rep['name']]:.3f}")
    chance = 1.0 / N_ODORS
    print(f"  chance = {chance:.3f}")

    # ---- accuracy vs concentration ----------------------------------------
    n_bins = 8
    edges = np.linspace(LOG_INT_LO, LOG_INT_HI, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip(np.digitize(test_log_int, edges) - 1, 0, n_bins - 1)
    per_bin_acc = {}
    for name in accs:
        pred = preds[name]
        per_bin_acc[name] = np.array([
            (pred[bin_idx == b] == y_true[bin_idx == b]).mean()
            if (bin_idx == b).any() else np.nan
            for b in range(n_bins)
        ])

    # ---- demo: concentration sweep for one odor ---------------------------
    demo_odor = int(np.argmax(E.sum(axis=1)))
    demo_concs = np.logspace(LOG_INT_LO, LOG_INT_HI, M_DEMO_CONCS, dtype=np.float32)
    demo_drive = (E[demo_odor][None, :] * demo_concs[:, None]).astype(np.float32)
    demo_static_orn = np.clip(demo_drive, 0.0, None)
    demo_static_pn = divisive_norm(demo_static_orn, NORM_SIGMA)

    demo_intensities = np.repeat(demo_drive, K_DEMO_REPEATS, axis=0).astype(np.float32)
    demo_channels = np.zeros((demo_intensities.shape[0], len(orn_types)), dtype=np.float32)
    demo_channels[:, channel_of_matched] = demo_intensities
    demo_seeds = rng.integers(2 ** 31 - 1, size=demo_intensities.shape[0]).astype(np.int64)
    demo_conn = {}
    for mode in MODES_TO_RUN:
        print(f"\nDemo sweep through connectome ({mode} inh) -- "
              f"{M_DEMO_CONCS} concs x {K_DEMO_REPEATS} repeats...")
        set_dynamics(models[mode], mode)
        d_orn, d_alpn = simulate_per_glom(
            models[mode], region, demo_channels, demo_seeds, read_steps,
            orn_types, pn_types, matched_gloms, orn_type_of_glom, pn_types_of_glom,
        )
        demo_conn[mode] = {
            'orn': d_orn.reshape(M_DEMO_CONCS, K_DEMO_REPEATS, n_gloms).mean(axis=1),
            'alpn': d_alpn.reshape(M_DEMO_CONCS, K_DEMO_REPEATS, n_gloms).mean(axis=1),
        }

    plot_figure(representations, accs, per_bin_acc, centers, chance,
                demo_odor, demo_concs, demo_static_orn, demo_static_pn,
                demo_conn, n_gloms, t_read)
    print("\nDone.")


def plot_figure(representations, accs, per_bin_acc, centers, chance,
                demo_odor, demo_concs, demo_static_orn, demo_static_pn,
                demo_conn, n_gloms, t_read):
    """2x2 figure: overall accuracy, accuracy vs concentration, static demo,
    connectome demo (mirrors concentration_divnorm.py)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # (0,0) overall accuracy bars -- colour = population, hatch = mode.
    ax = axes[0, 0]
    for xi, rep in enumerate(representations):
        ax.bar(xi, accs[rep['name']], color=rep['color'], hatch=rep['hatch'],
               edgecolor='black', linewidth=0.8)
        ax.text(xi, accs[rep['name']] + 0.02, f"{accs[rep['name']]:.2f}",
                ha='center', fontsize=9)
    ax.axhline(chance, ls='--', color='gray', label='chance')
    ax.set_xticks(np.arange(len(representations)))
    ax.set_xticklabels([rep['short'] for rep in representations], fontsize=8.5)
    ax.set_ylabel('classification accuracy')
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Overall identity decoding ({N_ODORS}-way)\n"
                 "solid = subtractive inhibition,  hatched = divisive inhibition")
    ax.legend(loc='lower right', fontsize=8, frameon=False)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # (0,1) accuracy vs concentration -- colour = population, linestyle = mode.
    ax = axes[0, 1]
    for rep in representations:
        ax.plot(centers, per_bin_acc[rep['name']], color=rep['color'],
                linestyle=rep['linestyle'], marker=rep['marker'], lw=1.8,
                label=rep['short'].replace('\n', ' '))
    ax.axhline(chance, ls='--', color='gray')
    ax.set_xlabel(r'$\log_{10}$ concentration')
    ax.set_ylabel('accuracy')
    ax.set_ylim(0, 1.05)
    ax.set_title('Identity accuracy vs concentration\n'
                 'solid = subtractive inhibition,  dashed = divisive inhibition')
    ax.legend(loc='lower right', fontsize=8, frameon=False, ncol=2)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # (1,0) static demo: per-glom response curves for one odor.
    ax = axes[1, 0]
    for k in range(n_gloms):
        ax.plot(demo_concs, demo_static_orn[:, k], color=COL_STATIC_ORN, alpha=0.35, lw=0.7)
        ax.plot(demo_concs, demo_static_pn[:, k], color=COL_STATIC_PN, alpha=0.35, lw=0.7)
    ax.set_xscale('log')
    ax.set_xlabel('concentration')
    ax.set_ylabel('per-glomerulus response')
    ax.set_title(f"STATIC model -- odor {demo_odor}\neach line is one glomerulus")
    ax.legend(handles=[
        plt.Line2D([0], [0], color=COL_STATIC_ORN, lw=2, label='Static ORN'),
        plt.Line2D([0], [0], color=COL_STATIC_PN, lw=2, label='Static PN (div norm)'),
    ], fontsize=8, loc='best', frameon=False)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # (1,1) connectome demo: per-glom response curves, colour = population,
    # linestyle = inhibition mode.
    ax = axes[1, 1]
    handles = []
    for mode in MODES_TO_RUN:
        st = MODE_STYLE[mode]
        for k in range(n_gloms):
            ax.plot(demo_concs, demo_conn[mode]['orn'][:, k], color=COL_CNX_ORN,
                    linestyle=st['linestyle'], alpha=0.20, lw=0.7)
            ax.plot(demo_concs, demo_conn[mode]['alpn'][:, k], color=COL_CNX_ALPN,
                    linestyle=st['linestyle'], alpha=0.30, lw=0.7)
        handles.append(plt.Line2D([0], [0], color=COL_CNX_ORN, linestyle=st['linestyle'],
                                  lw=2, label=f"Cnx ORN ({mode})"))
        handles.append(plt.Line2D([0], [0], color=COL_CNX_ALPN, linestyle=st['linestyle'],
                                  lw=2, label=f"Cnx ALPN ({mode})"))
    ax.set_xscale('log')
    ax.set_xlabel('concentration')
    ax.set_ylabel('per-glomerulus response')
    ax.set_title(f"CONNECTOME model -- odor {demo_odor}\neach line is one glomerulus")
    ax.legend(handles=handles, fontsize=7.5, loc='best', frameon=False)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    readout_desc = (f"single time point t={SIM_T_READ_LO:.2f}s" if N_READ_BINS == 1
                    else f"avg over t in [{SIM_T_READ_LO:.2f}, {SIM_T_READ_HI:.2f}]s")
    fig.suptitle(
        "Concentration-invariant odor identity: static divisive normalisation "
        "vs the AL connectome\n"
        f"(N={N_TRIALS} trials, {N_ODORS} odors, sparsity={SPARSITY}, "
        f"conc range = 10^[{LOG_INT_LO}, {LOG_INT_HI}], "
        f"connectome readout = {readout_desc})",
        fontsize=12,
    )
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"\nsaved {OUT_PNG}")


if __name__ == '__main__':
    main()
