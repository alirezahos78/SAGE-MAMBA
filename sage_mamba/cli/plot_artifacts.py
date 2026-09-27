"""Plot numeric model exports with comparable scales and explicit provenance."""

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Arc, Circle
import mne
from sage_mamba.preprocessing import make_info

plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
                     'ps.fonttype': 42, 'svg.fonttype': 'none'})


@dataclass
class Artifact:
    path: Path
    names: list
    attention: np.ndarray
    graphs: list
    metadata: dict
    attention_indices: np.ndarray


def load_artifact(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        names = data['channel_names'].tolist()
        attention = np.asarray(data['attention'], dtype=float)
        keys = sorted((k for k in data.files if k.startswith('adjacency_block_')),
                      key=lambda k: int(k.rsplit('_', 1)[-1]))
        graphs = [np.asarray(data[k], dtype=float) for k in keys]
        metadata = {key: data[key].item() for key in data.files if data[key].ndim == 0}
        indices = np.asarray(data['attention_indices'], dtype=np.int64)
    C = len(names)
    if attention.shape != (C,) or not np.isfinite(attention).all() or (attention < 0).any():
        raise ValueError('Invalid attention vector.')
    if not np.isclose(attention.sum(), 1, atol=2e-5):
        raise ValueError('Attention must sum to one; input is not renormalized silently.')
    if not graphs:
        raise ValueError('Artifact contains no block graphs.')
    for graph in graphs:
        if graph.shape != (C, C) or not np.isfinite(graph).all() or not np.allclose(graph, graph.T, atol=1e-6):
            raise ValueError('Each graph must be a finite symmetric C-by-C affinity matrix.')
    if len(indices) != metadata['n_attention_windows']:
        raise ValueError('Attention sample count does not match exported indices.')
    return Artifact(path, names, attention, graphs, metadata, indices)


def validate_pair(sparse, dense):
    if not 0 < sparse.metadata['phi'] < 1 or dense.metadata['phi'] != 1.0:
        raise ValueError('Pair requires sparse phi < 1 and dense phi = 1.')
    if sparse.names != dense.names or len(sparse.graphs) != len(dense.graphs):
        raise ValueError('Channel order or block count differs.')
    fields = ['dataset', 'backbone', 'seed', 'protocol', 'd_model',
              'train_split_sha256', 'test_split_sha256']
    fields += ['condition' if sparse.metadata['dataset'] == 'seed' else 'target']
    for key in fields:
        if key not in sparse.metadata or sparse.metadata.get(key) != dense.metadata.get(key):
            raise ValueError(f'Dense/sparse mismatch for {key}.')
    if not np.array_equal(sparse.attention_indices, dense.attention_indices):
        raise ValueError('Attention vectors must summarize the same ordered test windows.')


def graph_statistics(graph):
    values = graph[np.triu_indices_from(graph, k=1)]
    positive, negative = int((values > 1e-8).sum()), int((values < -1e-8).sum())
    retained = positive + negative
    return dict(positive=positive, negative=negative, retained=retained,
                possible=len(values), density=100 * retained / len(values),
                positive_share=100 * positive / retained if retained else 0.0)


def coordinates(artifact):
    fs = 200 if artifact.metadata['dataset'] == 'seed' else 128
    info = make_info(artifact.names, fs)
    xyz = np.array([ch['loc'][:3] for ch in info['chs']])
    radius = np.linalg.norm(xyz, axis=1)
    theta = np.arccos(np.clip(xyz[:, 2] / radius, -1, 1))
    azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
    xy = np.column_stack([theta * np.cos(azimuth), theta * np.sin(azimuth)])
    xy *= 0.92 / np.linalg.norm(xy, axis=1).max()
    return info, xy


def head_outline(ax):
    ax.add_patch(Circle((0, 0), 1, fill=False, lw=1.3, color='black'))
    ax.plot([-.12, 0, .12], [.994, 1.15, .994], color='black', lw=1.2)
    ax.add_patch(Arc((-1, 0), .23, .5, theta1=90, theta2=270, lw=1.1))
    ax.add_patch(Arc((1, 0), .23, .5, theta1=-90, theta2=90, lw=1.1))
    ax.set(xlim=(-1.17, 1.17), ylim=(-1.13, 1.19), aspect='equal')
    ax.axis('off')


def graph_panel(ax, artifact, block, xy, top_edges):
    graph = artifact.graphs[block]
    stats = graph_statistics(graph)
    rows, cols = np.triu_indices_from(graph, k=1)
    weights = graph[rows, cols]
    active = np.flatnonzero(np.abs(weights) > 1e-8)
    chosen = active[np.argsort(-np.abs(weights[active]), kind='stable')[:top_edges]]
    max_weight = max(float(np.abs(weights[chosen]).max()), 1e-12) if len(chosen) else 1.0
    for index in chosen[::-1]:
        points = xy[[rows[index], cols[index]]]
        ax.plot(points[:, 0], points[:, 1], color='#c94040' if weights[index] > 0 else '#287ca8',
                alpha=.68, lw=.45 + 1.1 * abs(weights[index]) / max_weight, zorder=1)
    head_outline(ax)
    # Matplotlib's s is area in points squared; no offset changes area ratios.
    ax.scatter(xy[:, 0], xy[:, 1], s=1600 * artifact.attention, c='#fbd650',
               edgecolors='#444444', linewidths=.5, zorder=3)
    for name, point in zip(artifact.names, xy):
        ax.annotate(name, point, xytext=(0, -4.5), textcoords='offset points',
                    ha='center', va='top', fontsize=4.8, color='#202020', zorder=4)
    ax.set_title(f'Block {block+1}\nDensity {stats["density"]:.1f}% | '
                 f'+{stats["positive"]} / -{stats["negative"]}', fontsize=8.5, pad=15)
    return stats


def save_figure(fig, directory, name):
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(directory / f'{name}.{extension}', dpi=300,
                    bbox_inches='tight', pad_inches=.06, facecolor='white')
    plt.close(fig)


def label(artifact):
    variant = 'Dense' if artifact.metadata['phi'] == 1 else 'Sparse'
    return f'{variant} (seed {artifact.metadata["seed"]})'


def graph_figures(sparse, dense, out, top_edges):
    _, xy = coordinates(sparse)
    n = len(sparse.graphs)
    fig, axes = plt.subplots(math.ceil(n / 2), 2, figsize=(7.2, 3.65 * math.ceil(n / 2)), squeeze=False)
    stats = []
    for block, ax in enumerate(axes.flat):
        if block < n:
            stats.append(dict(variant='sparse' if sparse.metadata['phi'] < 1 else 'dense',
                              block=block + 1, **graph_panel(ax, sparse, block, xy, top_edges)))
        else:
            ax.axis('off')
    fig.subplots_adjust(hspace=.30, wspace=.08, bottom=.075, top=.86)
    fig.suptitle(f'{label(sparse)}: learned inter-channel graphs', fontsize=12)
    fig.text(.5, .015, f'Top {top_edges} edges shown per block; statistics use all off-diagonal edges.\n'
             'Red: positive; blue: negative. Node area is proportional to channel attention.',
             ha='center', fontsize=7.5)
    save_figure(fig, out, 'graph_per_block')
    if dense is not None:
        fig, axes = plt.subplots(2, n, figsize=(3.3 * n, 7.1), squeeze=False)
        for row, artifact in enumerate([dense, sparse]):
            for block in range(n):
                st = graph_panel(axes[row, block], artifact, block, xy, top_edges)
                if row == 0:
                    stats.append(dict(variant='dense', block=block + 1, **st))
            axes[row, 0].text(-.1, .5, label(artifact), rotation=90,
                              va='center', ha='right', transform=axes[row, 0].transAxes, fontsize=10)
        fig.subplots_adjust(hspace=.25, wspace=.04, bottom=.07)
        fig.text(.5, .015, f'Top {top_edges} edges per block; full-graph statistics; '
                 'red positive, blue negative; node area proportional to attention.',
                 ha='center', fontsize=8)
        save_figure(fig, out, 'graph_dense_sparse')
    return stats


def topomaps(artifacts, out, name, interpolation):
    fig, axes = plt.subplots(1, len(artifacts), figsize=(3.8 * len(artifacts) + .8, 4.6), squeeze=False)
    images, field_min, field_max = [], 0.0, 0.0
    omitted = []
    for ax, artifact in zip(axes.flat, artifacts):
        info, _ = coordinates(artifact)
        keep = [i for i, ch in enumerate(artifact.names) if ch not in ['CB1', 'CB2']]
        omitted = [ch for ch in artifact.names if ch in ['CB1', 'CB2']]
        plot_info = mne.pick_info(info, keep)
        im, _ = mne.viz.plot_topomap(
            artifact.attention[keep], plot_info, axes=ax, show=False,
            image_interp=interpolation, extrapolate='head', contours=0,
            cmap='YlOrRd', sensors=True, res=256, vlim=(0, float(artifact.attention.max())))
        values = np.ma.asarray(im.get_array())
        field_min = min(field_min, float(values.min()))
        field_max = max(field_max, float(values.max()), float(artifact.attention.max()))
        images.append(im)
        top = np.argsort(-artifact.attention)[:3]
        ax.set_title(label(artifact) + '\n' + ', '.join(
            f'{artifact.names[i]} {100*artifact.attention[i]:.1f}%' for i in top), fontsize=9, pad=16)
    step = .02 if field_max <= .30 else .05
    vmin, vmax = math.floor(field_min / step) * step, math.ceil(field_max / step) * step
    if vmax <= vmin:
        vmax = vmin + step
    norm = Normalize(vmin=vmin, vmax=vmax, clip=False)
    for im in images:
        im.set_norm(norm)
    fig.subplots_adjust(left=.025, right=.86, top=.80, bottom=.19, wspace=.18)
    color_axis = fig.add_axes([.9, .24, .025, .49])
    bar = fig.colorbar(images[0], cax=color_axis)
    bar.set_label('Mean channel attention', fontsize=9)
    bar.ax.tick_params(labelsize=8)
    fig.suptitle('Channel attention', fontsize=13, y=.98)
    ref = artifacts[0]
    caption = (f'{ref.metadata["n_attention_windows"]:,} / {ref.metadata["n_test_windows"]:,} test windows; '
               f'{interpolation} interpolation; common color limits [{vmin:.2f}, {vmax:.2f}].')
    if omitted:
        caption += '\nCB1/CB2 are omitted from scalp interpolation; weights are not renormalized.'
    if interpolation == 'cubic':
        caption += '\nInterpolated extrema may exceed electrode values; the color scale includes them.'
    fig.text(.5, .035, caption, ha='center', fontsize=7)
    save_figure(fig, out, name)
    return dict(vmin=vmin, vmax=vmax, interpolated_min=field_min,
                interpolated_max=field_max, interpolation=interpolation,
                omitted_channels=omitted)


def plot(artifact_path, out_dir, dense_path=None, top_edges=60, interpolation='linear'):
    if top_edges < 1:
        raise ValueError('top_edges must be positive.')
    sparse = load_artifact(artifact_path)
    dense = load_artifact(dense_path) if dense_path else None
    if dense:
        validate_pair(sparse, dense)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats = graph_figures(sparse, dense, out, top_edges)
    scales = {'single': topomaps([sparse], out, 'attention_topomap', interpolation)}
    if dense:
        scales['comparison'] = topomaps([dense, sparse], out, 'attention_dense_sparse', interpolation)
    with (out / 'graph_statistics.csv').open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(stats[0]))
        writer.writeheader()
        writer.writerows(stats)
    with (out / 'channel_attention.csv').open('w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['channel', 'sparse_attention' if dense else 'attention'] +
                        (['dense_attention'] if dense else []))
        for i, name in enumerate(sparse.names):
            writer.writerow([name, sparse.attention[i]] + ([dense.attention[i]] if dense else []))
    report = dict(sparse=sparse.metadata, dense=dense.metadata if dense else None,
                  color_scales=scales, top_edges=top_edges,
                  graph_coordinates='standard_1020; approximate CB1/CB2 positions',
                  interpretation='Learned model affinities; not causal or anatomical connectivity.')
    (out / 'plot_metadata.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Saved figures to {out.resolve()}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', required=True)
    parser.add_argument('--dense', help='Matching phi=1 artifact for a paired comparison.')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--top-edges', type=int, default=60)
    parser.add_argument('--interpolation', choices=['linear', 'cubic'], default='linear')
    args = parser.parse_args()
    mne.set_log_level('ERROR')
    plot(args.artifact, args.out_dir, args.dense, args.top_edges, args.interpolation)


if __name__ == '__main__':
    main()
