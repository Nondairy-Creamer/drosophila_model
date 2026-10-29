"""
Sanity checks of the optic-lobe column geometry (eye map), the CT1 split and the column wiring.

Visual coordinates in every figure: azimuth to the right (``-phi`` in degrees: 0 = straight ahead,
90 = the right side) and elevation (``90 - theta`` in degrees: up is positive); see
``coordinate_conventions.md``.

Saves to ``figures/eye_map_checks/``:

- ``eye_map.png``: right-eye column viewing directions, coloured by hex ``p`` and ``q``, the hex
  lattice coloured by azimuth and elevation, both eyes' coverage, and the angle between lattice
  neighbours.
- ``ct1_synapses.png``: the right CT1's medulla and lobula synapses in the plane of each neuropil,
  coloured by the piece (column) they are assigned to and by that column's azimuth / elevation.
- ``column_wiring.png``: for L1, L2 and L3, how many synapses onto each main partner type stay in
  the same column, and how far (in hex steps) the rest go.
- ``t4_t5_layout.png``: per T4 / T5 subtype, the visual-space direction from the inputs at the
  dendrite tip (Mi9 / Tm9) to the inputs at its base (Mi4 / CT1_L) and centre (Mi1 / Tm1). T4 and
  T5 dendrites point against the preferred direction, so tip -> base should be the subtype's
  preferred direction: a front-to-back, b back-to-front, c up, d down.

Reads the baked connectome (and, for the CT1 per-synapse assignment, the raw FlyWire synapse
table) from ``[data].data_folder``.

Usage (from the repo root):
    python -m drosophila_model.analysis.model_analysis.eye_map_checks
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from drosophila_model.config import get_data_folder, load_config
from drosophila_model.data_handling import split_ct1

OUT_DIR = Path('figures') / 'eye_map_checks'
SUBTYPES = ['a', 'b', 'c', 'd']
# Preferred direction of each subtype in (azimuth to the right, elevation).
EXPECTED_PD = {'a': (1.0, 0.0), 'b': (-1.0, 0.0), 'c': (0.0, 1.0), 'd': (0.0, -1.0)}
_HEX_BASIS = np.array([[1.0, 0.0], [-0.5, np.sqrt(3) / 2]])


def visual_coords(phi, theta):
    """(azimuth to the right, elevation) in degrees."""
    return -np.degrees(phi), 90.0 - np.degrees(theta)


def load_tables(data_folder):
    baked = Path(data_folder) / 'connectome_data' / 'baked'
    cols = pd.read_parquet(baked / 'optic_lobe_columns.parquet')
    column_of = pd.read_parquet(baked / 'optic_lobe_column_index.parquet') \
        .set_index('root_id')['column_index']
    info = pd.read_parquet(baked / 'cell_info.parquet', columns=['root_id', 'primary_type', 'side'])
    pieces = pd.read_parquet(baked / 'ct1_pieces.parquet')
    type_of = pd.concat([
        info[info['side'] == 'right'].set_index('root_id')['primary_type'],
        pieces[pieces['hemisphere'] == 'right'].set_index('root_id')['primary_type'],
    ])
    syn = pq.read_table(baked / 'synapses.parquet',
                        columns=['pre_root_id', 'post_root_id', 'synapse_counts', 'neuropil'],
                        filters=[('neuropil', 'in', ['ME_R', 'LO_R'])]).to_pandas()
    syn['pre_type'] = syn['pre_root_id'].map(type_of)
    syn['post_type'] = syn['post_root_id'].map(type_of)
    syn['pre_col'] = syn['pre_root_id'].map(column_of)
    syn['post_col'] = syn['post_root_id'].map(column_of)
    return cols, syn


def plot_eye_map(cols, out_dir):
    az, el = visual_coords(cols['phi'].to_numpy(), cols['theta'].to_numpy())
    interp = (cols['source'] == 'interpolated').to_numpy()
    hex_xy = cols[['p', 'q']].to_numpy() @ _HEX_BASIS
    xyz = cols[['x', 'y', 'z']].to_numpy()
    index = {(p, q): i for i, (p, q) in enumerate(zip(cols['p'], cols['q']))}
    edges = [(i, index[(p + dp, q + dq)]) for (p, q), i in index.items()
             for dp, dq in ((1, 0), (0, 1), (1, 1)) if (p + dp, q + dq) in index]
    angles = np.degrees(np.arccos(np.clip([xyz[i] @ xyz[j] for i, j in edges], -1, 1)))

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for ax, key in zip(axes[0, :2], ('p', 'q')):
        for i, j in edges:
            ax.plot(az[[i, j]], el[[i, j]], color='0.85', lw=0.4, zorder=0)
        sc = ax.scatter(az, el, c=cols[key], cmap='viridis', s=14)
        ax.scatter(az[interp], el[interp], facecolors='none', edgecolors='r', s=30, lw=0.8,
                   label=f'interpolated ({interp.sum()})')
        fig.colorbar(sc, ax=ax, label=key)
        ax.set(xlabel='azimuth to the right (deg)', ylabel='elevation (deg)',
               title=f'Right-eye columns coloured by hex {key}')
        ax.set_aspect('equal')
        ax.legend(loc='lower left', fontsize=8)
    ax = axes[0, 2]
    ax.scatter(np.degrees(cols['phi']), el, s=6, label='right eye (phi)')
    ax.scatter(-np.degrees(cols['phi']), el, s=6, label='left eye (-phi)')
    ax.axvline(0, color='k', lw=0.8)
    ax.set(xlabel='phi (deg; +90 = left, -90 = right)', ylabel='elevation (deg)',
           title='Both eyes (left = sagittal mirror)', xlim=(185, -185))
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    for ax, values, label in ((axes[1, 0], az, 'azimuth to the right (deg)'),
                              (axes[1, 1], el, 'elevation (deg)')):
        sc = ax.scatter(hex_xy[:, 0], hex_xy[:, 1], c=values, cmap='viridis'
                        if 'azimuth' in label else 'coolwarm', s=14)
        ax.scatter(hex_xy[interp, 0], hex_xy[interp, 1], facecolors='none', edgecolors='k',
                   s=30, lw=0.8)
        fig.colorbar(sc, ax=ax, label=label)
        ax.set(xlabel='hex plane x (p axis)', ylabel='hex plane y',
               title=f'Hex lattice coloured by {label.split(" (")[0]}')
        ax.set_aspect('equal')
    ax = axes[1, 2]
    ax.hist(angles, bins=60)
    ax.set(xlabel='angle between lattice neighbours (deg)', ylabel='neighbour pairs',
           title=f'Neighbour spacing: median {np.median(angles):.1f} deg, max {angles.max():.1f}')
    fig.tight_layout()
    fig.savefig(out_dir / 'eye_map.png', dpi=120)
    plt.close(fig)
    print(f"eye map: {len(cols)} columns ({interp.sum()} interpolated); azimuth {az.min():.0f} to "
          f"{az.max():.0f} deg, elevation {el.min():.0f} to {el.max():.0f} deg; lattice neighbours "
          f"median {np.median(angles):.1f} deg apart, max {angles.max():.1f}")


def plot_ct1_synapses(cols, data_folder, out_dir):
    """Re-run split_ct1's per-synapse column vote for the right CT1 and plot it."""
    raw = Path(data_folder) / 'connectome_data' / 'raw'
    cell_types = pd.read_csv(raw / 'consolidated_cell_types.csv.gz')
    ct1_ids = np.sort(cell_types.loc[cell_types['primary_type'] == 'CT1', 'root_id'].to_numpy())
    columns = pd.read_csv(raw / 'column_assignment.csv.gz').dropna(subset=['column_id'])
    columns['column_id'] = columns['column_id'].astype(np.int64)
    reference = columns[~columns['type'].isin(split_ct1.MULTICOLUMNAR_TYPES)]
    print("CT1: reading the raw synapse table (about a minute) ...")
    ct1_syn, ref_syn = split_ct1.read_synapses(raw / split_ct1.SYNAPSE_TABLE, ct1_ids,
                                               reference['root_id'].to_numpy())
    ct1_syn = ct1_syn[ct1_syn['pre_root_id'] != ct1_syn['post_root_id']].reset_index(drop=True)
    ct1_syn['column_id'] = split_ct1._assign_columns(ct1_syn, ref_syn, reference)

    az, el = visual_coords(cols['phi'].to_numpy(), cols['theta'].to_numpy())
    rng = np.random.default_rng(0)
    piece_colour = rng.random((len(cols) + 1, 3)) * 0.85 + 0.1
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for row, neuropil in enumerate(('ME_R', 'LO_R')):
        s = ct1_syn[(ct1_syn['neuropil'] == neuropil)].dropna(subset=['column_id'])
        xyz = s[['ctr_x', 'ctr_y', 'ctr_z']].to_numpy(np.float64)
        centred = xyz - xyz.mean(axis=0)
        _, _, vt = np.linalg.svd(centred[::max(1, len(centred) // 20000)], full_matrices=False)
        plane = centred @ vt[:2].T / 1e3
        col_index = s['column_id'].to_numpy(np.int64) - 1
        name = 'medulla' if neuropil == 'ME_R' else 'lobula'
        panels = [(piece_colour[col_index], None, f'{name}: by piece'),
                  (az[col_index], 'viridis', f'{name}: column azimuth (deg)'),
                  (el[col_index], 'coolwarm', f'{name}: column elevation (deg)')]
        for ax, (colour, cmap, title) in zip(axes[row], panels):
            sc = ax.scatter(plane[:, 0], plane[:, 1], c=colour, cmap=cmap, s=0.3, rasterized=True)
            if cmap:
                fig.colorbar(sc, ax=ax)
            ax.set(title=f'{title}\n{len(s):,} synapses, {s["column_id"].nunique()} pieces',
                   xlabel='sheet axis 1 (um)', ylabel='sheet axis 2 (um)')
            ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(out_dir / 'ct1_synapses.png', dpi=110)
    plt.close(fig)


def plot_column_wiring(syn, cols, out_dir, top=6):
    pq_of = cols.set_index('column_index')[['p', 'q']]
    both = syn.dropna(subset=['pre_col', 'post_col'])
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for ax, pre in zip(axes, ('L1', 'L2', 'L3')):
        s = both[(both['pre_type'] == pre) & (both['neuropil'] == 'ME_R')]
        partners = s.groupby('post_type')['synapse_counts'].sum().nlargest(top).index
        dp = pq_of.loc[s['post_col'], 'p'].to_numpy() - pq_of.loc[s['pre_col'], 'p'].to_numpy()
        dq = pq_of.loc[s['post_col'], 'q'].to_numpy() - pq_of.loc[s['pre_col'], 'q'].to_numpy()
        # Hex steps between the two columns (0 = same column, capped at 2).
        steps = np.minimum(np.maximum.reduce([np.abs(dp), np.abs(dq), np.abs(dp - dq)]), 2)
        counts = s.assign(steps=steps).groupby(['post_type', 'steps'])['synapse_counts'].sum() \
            .unstack(fill_value=0).reindex(index=partners, columns=[0, 1, 2], fill_value=0)
        frac = counts.div(counts.sum(axis=1), axis=0)
        bottom = np.zeros(len(partners))
        for d, colour, label in zip((0, 1, 2), ('tab:green', 'tab:orange', 'tab:red'),
                                    ('same column', '1 hex step', '2+ steps')):
            ax.bar(partners, frac[d], bottom=bottom, color=colour, label=label)
            bottom += frac[d].to_numpy()
        print(f"{pre} -> partners, fraction of synapses in the same column: "
              + ", ".join(f"{t} {frac.loc[t, 0]:.2f}" for t in partners))
        ax.set(title=f'{pre} -> top partners in the medulla', ylabel='fraction of synapses', ylim=(0, 1))
        ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(out_dir / 'column_wiring.png', dpi=120)
    plt.close(fig)


def _input_vectors(syn, cols, post_type, tip_type, base_type, centre_type, neuropil):
    """Per post cell: tip -> base and tip -> centre directions (deg) in the local visual frame."""
    xyz = cols.set_index('column_index')[['x', 'y', 'z']]
    s = syn[(syn['post_type'] == post_type) & (syn['neuropil'] == neuropil)].dropna(subset=['pre_col'])
    s = s[s['pre_type'].isin([tip_type, base_type, centre_type])]
    w = s['synapse_counts'].to_numpy(np.float64)[:, None]
    weighted = pd.DataFrame(xyz.loc[s['pre_col']].to_numpy() * w, columns=['x', 'y', 'z'])
    weighted[['post', 'pre_type']] = s[['post_root_id', 'pre_type']].to_numpy()
    cent = weighted.groupby(['post', 'pre_type'])[['x', 'y', 'z']].sum()
    cent = cent.div(np.linalg.norm(cent.to_numpy(), axis=1), axis=0).unstack('pre_type').dropna()
    tip = np.stack([cent[(c, tip_type)] for c in 'xyz'], axis=1)
    out = {}
    for name, other in (('base', base_type), ('centre', centre_type)):
        d = np.stack([cent[(c, other)] for c in 'xyz'], axis=1) - tip
        phi = np.arctan2(tip[:, 1], tip[:, 0])
        theta = np.arccos(np.clip(tip[:, 2], -1, 1))
        e_az = np.stack([np.sin(phi), -np.cos(phi), np.zeros_like(phi)], axis=1)  # -e_phi
        e_el = np.stack([-np.cos(theta) * np.cos(phi), -np.cos(theta) * np.sin(phi),
                         np.sin(theta)], axis=1)  # -e_theta
        out[name] = np.degrees(np.stack([(d * e_az).sum(1), (d * e_el).sum(1)], axis=1))
    return out


def plot_t4_t5_layout(syn, cols, out_dir):
    families = [('T4', 'Mi9', 'Mi4', 'Mi1', 'ME_R'), ('T5', 'Tm9', 'CT1_L', 'Tm1', 'LO_R')]
    fig, axes = plt.subplots(2, 4, figsize=(17, 9))
    for row, (fam, tip, base, centre, neuropil) in enumerate(families):
        for ax, sub in zip(axes[row], SUBTYPES):
            v = _input_vectors(syn, cols, fam + sub, tip, base, centre, neuropil)
            exp = np.array(EXPECTED_PD[sub])
            for name, colour in (('base', 'tab:blue'), ('centre', 'tab:orange')):
                vec = v[name]
                ax.scatter(vec[:, 0], vec[:, 1], s=4, alpha=0.3, color=colour)
                mean = vec.mean(axis=0)
                ax.annotate('', xy=mean, xytext=(0, 0),
                            arrowprops=dict(arrowstyle='->', color=colour, lw=2.5))
                cos = (vec @ exp) / np.maximum(np.linalg.norm(vec, axis=1), 1e-9)
                frac = (cos > np.cos(np.radians(45))).mean()
                ax.plot([], [], color=colour, lw=2.5,
                        label=f'{tip} -> {"base " + base if name == "base" else "centre " + centre}: '
                              f'{frac:.0%} within 45 deg')
                if name == 'base':
                    print(f"{fam}{sub}: {len(vec)} cells; mean {tip}->{base} = ({mean[0]:+.1f}, "
                          f"{mean[1]:+.1f}) deg (az, el); {frac:.0%} within 45 deg of expected")
            lim = 1.1 * np.percentile(np.abs(np.concatenate([v['base'], v['centre']])), 98)
            ax.annotate('', xy=exp * lim * 0.8, xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color='k', lw=1.5, ls='--'))
            ax.plot([], [], color='k', ls='--', label='expected preferred direction')
            ax.axhline(0, color='0.8', lw=0.6)
            ax.axvline(0, color='0.8', lw=0.6)
            ax.set(xlim=(-lim, lim), ylim=(-lim, lim), title=f'{fam}{sub}',
                   xlabel='-> back (azimuth, deg)', ylabel='-> up (elevation, deg)')
            ax.set_aspect('equal')
            ax.legend(fontsize=7, loc='lower left')
    fig.suptitle('Direction from dendrite-tip inputs to base (blue) / centre (orange) inputs, '
                 'per cell, in visual space')
    fig.tight_layout()
    fig.savefig(out_dir / 't4_t5_layout.png', dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the eye map, the CT1 split and column wiring.")
    parser.add_argument('--skip-ct1', action='store_true',
                        help='Skip the CT1 scatter (it streams the raw synapse table).')
    args = parser.parse_args()
    data_folder = get_data_folder(load_config())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols, syn = load_tables(data_folder)
    plot_eye_map(cols, OUT_DIR)
    plot_column_wiring(syn, cols, OUT_DIR)
    plot_t4_t5_layout(syn, cols, OUT_DIR)
    if not args.skip_ct1:
        plot_ct1_synapses(cols, data_folder, OUT_DIR)
    print(f"Figures in {OUT_DIR}")


if __name__ == '__main__':
    main()
