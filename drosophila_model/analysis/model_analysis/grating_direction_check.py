"""
Check that drifting gratings travel across a model's columns in the right direction, in both eyes.

Renders right / left / up / down drifting sine gratings on both eyes with
``create_directional_sine_wave`` in ``get_stimulus``'s two-eye layout (right-eye columns, then the
left eye at ``-phi``; ``right`` moves toward -phi, rightward in the
world: front-to-back on the right eye, back-to-front on the left eye; ``up`` moves up), runs the
model on them (``OpticLobeRegion.predict_two_eyes``) and, for the stimulus and each recorded cell
type, estimates which way the response phase travels across the columns. Directions are in each
eye's network frame: the left eye runs as a mirrored right eye, so they are given in right-eye
column coordinates (front-to-back stays front-to-back). It also compares the flow targets and the
model's predicted flow with that travel direction, and reports the T4 / T5 subtypes' tuning
(anatomical preferred directions: a front-to-back, b back-to-front, c up, d down).

Saves to ``<model_folder>/figures/grating_direction_check/``:

- ``travel_direction.png``: travel direction per eye and grating (one arrow per cell type), with
  the expected direction and the flow target's direction.
- ``phase_maps.png``: the L1 response phase across each eye's columns for the right and up gratings.
- ``t4_t5_tuning.png``: mean T4 / T5 firing per grating direction, per eye.
- ``tuning.json``: those firing rates, ``{eye: {cell type: [right, left, up, down]}}``.

Usage (from the repo root):
    python -m drosophila_model.analysis.model_analysis.grating_direction_check models/<model>/
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from drosophila_model.data_handling.stimulus_generators import create_directional_sine_wave
from drosophila_model.model_definition import neural_models as nm
from drosophila_model.analysis.model_analysis.analysis_utilities import get_flow_key

DIRECTIONS = ['right', 'left', 'up', 'down']
COLUMN_TYPES = ['L1', 'L2', 'L3', 'Mi1', 'Tm1', 'Tm3', 'Mi4', 'Mi9']
DS_TYPES = [f'{fam}{sub}' for fam in ('T4', 'T5') for sub in 'abcd']
EYES = ['right', 'left']
# Expected travel in each eye's network frame, as (back, up).
EXPECTED = {
    'right': {'right': (1, 0), 'left': (-1, 0), 'up': (0, 1), 'down': (0, -1)},
    'left': {'right': (-1, 0), 'left': (1, 0), 'up': (0, 1), 'down': (0, -1)},
}
PREFERRED = {'a': (1, 0), 'b': (-1, 0), 'c': (0, 1), 'd': (0, -1)}
STIM_TIME = 2.0      # s of drifting grating
SKIP_TIME = 0.5      # s dropped at the start before measuring phase / tuning
TEMPORAL_FREQUENCY = 4.0  # Hz, the directional_sine default


def tangent_frame(phi, theta):
    """Unit vectors toward the back (-e_phi) and up (-e_theta) at each column."""
    e_back = np.stack([np.sin(phi), -np.cos(phi), np.zeros_like(phi)], axis=1)
    e_up = np.stack([-np.cos(theta) * np.cos(phi), -np.cos(theta) * np.sin(phi), np.sin(theta)], axis=1)
    return e_back, e_up


def lattice_edges(p, q):
    index = {(a, b): i for i, (a, b) in enumerate(zip(p, q))}
    return np.array([(i, index[(a + da, b + db)]) for (a, b), i in index.items()
                     for da, db in ((1, 0), (0, 1), (1, 1)) if (a + da, b + db) in index])


def travel_direction(x, dt, edges, xyz, e_back, e_up):
    """Unit (back, up) direction in which the phase of ``x`` (time, columns) lags, and the
    fraction of lattice edges whose lag sign agrees with it. NaN columns are ignored."""
    t = np.arange(x.shape[0]) * dt
    x = x - np.nanmean(x, axis=0)
    c = np.nansum(x * np.exp(-2j * np.pi * TEMPORAL_FREQUENCY * t)[:, None], axis=0)
    amp = np.abs(c)
    ok = np.isfinite(x).all(axis=0) & (amp > 1e-3 * np.nanmax(amp))
    e = edges[ok[edges[:, 0]] & ok[edges[:, 1]]]
    lag = np.angle(c[e[:, 0]] * np.conj(c[e[:, 1]]))          # phase lag from i to j
    d = xyz[e[:, 1]] - xyz[e[:, 0]]
    disp = np.degrees(np.stack([(d * e_back[e[:, 0]]).sum(1), (d * e_up[e[:, 0]]).sum(1)], axis=1))
    g, *_ = np.linalg.lstsq(disp, lag, rcond=None)
    agree = np.mean(np.sign(disp @ g) == np.sign(lag))
    return g / np.linalg.norm(g), agree, np.angle(c)


def angle_between(u, v):
    return np.degrees(np.arccos(np.clip(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check drifting gratings travel the right way across columns.")
    parser.add_argument("model_folder", type=str, help="Path to model folder or model.pt")
    parser.add_argument("--device", type=str, default="cpu", help="Device (default: cpu)")
    args = parser.parse_args()
    model_path = Path(args.model_folder)
    model_folder = model_path.parent if model_path.is_file() else model_path
    out_dir = model_folder / 'figures' / 'grating_direction_check'
    out_dir.mkdir(parents=True, exist_ok=True)

    model = nm.BaseRnn.load(model_folder / 'model.pt', device=args.device)
    model.eval()
    region = model.regions['optic_lobe']
    N = region.num_columns
    phi, theta = region.column_phi_theta[:, 0], region.column_phi_theta[:, 1]
    xyz = region.get_column_xyz()
    e_back, e_up = tangent_frame(phi, theta)
    edges = lattice_edges(*region.column_p_q.T)
    dt = model.dt
    skip = int(round(SKIP_TIME / dt))

    # Two-eye layout as get_stimulus(num_eyes=2) builds it: right-eye columns, then the left eye at -phi.
    data = [create_directional_sine_wave(np.concatenate([phi, -phi]), np.concatenate([theta, theta]), d,
                                         dt=dt, stim_time=STIM_TIME) for d in DIRECTIONS]
    stim = torch.tensor(np.concatenate([s[0] for s in data]), device=model.device, dtype=model.dtype)
    target = np.concatenate([s[1] for s in data])
    print(f"Running {len(DIRECTIONS)} gratings x 2 eyes, {stim.shape[1]} steps, on {model.device} ...")
    with torch.no_grad():
        results = region.predict_two_eyes(stim[:, :, :N], stim[:, :, N:],
                                          output_neurons=COLUMN_TYPES + DS_TYPES)
    flow_key = get_flow_key(model)

    rows = []  # (eye, grating, label, direction, agree)
    phases = {}
    tuning = {eye: {} for eye in EYES}
    for e_i, (eye, res) in enumerate(zip(EYES, results)):
        out = res['outputs']['optic_lobe']
        cols = slice(0, N) if eye == 'right' else slice(N, 2 * N)
        sign = 1.0 if eye == 'right' else -1.0   # physical u_right -> network-frame "back"
        for g_i, grating in enumerate(DIRECTIONS):
            series = {'stimulus': stim[g_i, skip:, cols].cpu().numpy()}
            for ct in COLUMN_TYPES:
                series[ct] = out['firing_rate'][ct][g_i, skip:].cpu().numpy()
            for label, x in series.items():
                direction, agree, phase = travel_direction(x, dt, edges, xyz, e_back, e_up)
                rows.append((eye, grating, label, direction, agree))
                if label == 'L1':
                    phases[(eye, grating)] = phase
            for label, flow in (('flow target', target[g_i, skip:, cols]),
                                ('predicted flow', out[flow_key][g_i, skip:].cpu().numpy())):
                m = np.nanmean(flow, axis=(0, 1))
                v = np.array([sign * m[0], m[1]])
                rows.append((eye, grating, label, v / np.linalg.norm(v), np.nan))
            for ct in DS_TYPES:
                tuning[eye].setdefault(ct, []).append(
                    float(np.nanmean(out['firing_rate'][ct][g_i, skip:].cpu().numpy())))

    print(f"{'eye':6s} {'grating':8s} {'signal':15s} {'(back, up)':>16s}  error vs expected  edges agreeing")
    worst = 0.0
    for eye, grating, label, d, agree in rows:
        err = angle_between(d, np.array(EXPECTED[eye][grating], dtype=float))
        worst = max(worst, err)
        print(f"{eye:6s} {grating:8s} {label:15s} ({d[0]:+.2f}, {d[1]:+.2f})   {err:6.1f} deg"
              + (f"        {agree:.0%}" if np.isfinite(agree) else ""))
    print(f"Largest error vs expected: {worst:.1f} deg")

    labels = ['stimulus'] + COLUMN_TYPES + ['flow target', 'predicted flow']
    colours = dict(zip(labels, plt.cm.tab20(np.linspace(0, 1, len(labels)))))
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5))
    for (eye, grating, label, d, _) in rows:
        ax = axes[EYES.index(eye), DIRECTIONS.index(grating)]
        style = dict(color=colours[label], lw=2.5 if 'flow' in label else 1.5,
                     ls='--' if 'flow' in label else '-')
        ax.annotate('', xy=d, xytext=(0, 0), arrowprops=dict(arrowstyle='->', **style))
    for r, eye in enumerate(EYES):
        for c, grating in enumerate(DIRECTIONS):
            ax = axes[r, c]
            exp = np.array(EXPECTED[eye][grating]) * 1.15
            ax.annotate('', xy=exp, xytext=(0, 0),
                        arrowprops=dict(arrowstyle='simple', color='0.85', lw=0.5), zorder=0)
            ax.set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), title=f'{eye} eye, {grating} grating',
                   xlabel='-> back (network frame)', ylabel='-> up')
            ax.set_aspect('equal')
            ax.axhline(0, color='0.9', lw=0.5)
            ax.axvline(0, color='0.9', lw=0.5)
    for label in labels:
        axes[0, 0].plot([], [], color=colours[label], ls='--' if 'flow' in label else '-', label=label)
    axes[0, 0].plot([], [], color='0.85', lw=6, label='expected')
    axes[0, 0].legend(fontsize=7, loc='lower left')
    fig.suptitle(f'{model_folder.name}: direction each signal travels across the columns')
    fig.tight_layout()
    fig.savefig(out_dir / 'travel_direction.png', dpi=120)
    plt.close(fig)

    az, el = -np.degrees(phi), 90 - np.degrees(theta)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for r, eye in enumerate(EYES):
        for c, grating in enumerate(('right', 'up')):
            ax = axes[r, c]
            sc = ax.scatter(az, el, c=phases[(eye, grating)], cmap='twilight', s=12)
            ax.set(title=f'L1 response phase: {eye} eye, {grating} grating (network frame)',
                   xlabel='azimuth to the right of the right-eye column (deg)', ylabel='elevation (deg)')
            ax.set_aspect('equal')
            fig.colorbar(sc, ax=ax, label='phase (rad)')
    fig.tight_layout()
    fig.savefig(out_dir / 'phase_maps.png', dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    width = 0.2
    for r, eye in enumerate(EYES):
        for c, fam in enumerate(('T4', 'T5')):
            ax = axes[r, c]
            for k, grating in enumerate(DIRECTIONS):
                vals = [tuning[eye][f'{fam}{s}'][k] for s in 'abcd']
                ax.bar(np.arange(4) + (k - 1.5) * width, vals, width, label=f'{grating} grating')
            ax.set_xticks(range(4), [f'{fam}{s}' for s in 'abcd'])
            ax.set(title=f'{eye} eye: mean firing rate per grating', ylabel='firing rate')
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / 't4_t5_tuning.png', dpi=110)
    plt.close(fig)

    print("T4/T5 preferred grating (expected in brackets):")
    for eye in EYES:
        line = []
        for ct in DS_TYPES:
            want = [g for g in DIRECTIONS if EXPECTED[eye][g] == PREFERRED[ct[-1]]][0]
            got = DIRECTIONS[int(np.argmax(tuning[eye][ct]))]
            line.append(f"{ct} {got} [{want}]")
        print(f"  {eye} eye: " + ", ".join(line))
    (out_dir / 'tuning.json').write_text(json.dumps(tuning, indent=1))
    print(f"Figures in {out_dir}")


if __name__ == '__main__':
    main()
