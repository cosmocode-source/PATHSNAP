"""Create partition-labelled figures from completed offline evaluation results."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
outdir = HERE / 'figures'
outdir.mkdir(exist_ok=True)
data = pd.read_csv(HERE / 'results/per_trip_metrics.csv')
for part, rows in data.groupby('partition'):
    base = rows[rows.method == 'classifier'].set_index('trip_id').sort_index()
    hmm = rows[rows.method == 'hmm'].set_index('trip_id').loc[base.index]
    for metric, label in [('top1_all', 'Point-level top-1'), ('continuity', 'Route continuity')]:
        fig, ax = plt.subplots(figsize=(max(8, len(base)*.16), 4))
        ax.plot(range(len(base)), base[metric], '.', label='Classifier')
        ax.plot(range(len(base)), hmm[metric], '.', label='HMM')
        ax.set(title=f'{part}: {label} by trip', xlabel='Trip (sorted by ID)', ylabel=label)
        ax.legend(); fig.tight_layout(); fig.savefig(outdir / f'{part}_{metric}.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    diff = hmm.top1_all - base.top1_all
    for ax, column in zip(axes, ['gap_share', 'far_share']):
        ax.scatter(hmm[column], diff, s=16); ax.axhline(0, color='grey', lw=.8)
        ax.set(xlabel=column.replace('_', ' '), ylabel='HMM minus classifier top-1')
    fig.suptitle(f'{part}: improvement and trip characteristics')
    fig.tight_layout(); fig.savefig(outdir / f'{part}_characteristics.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(hmm.n_points, hmm.decode_seconds, s=16)
    ax.set(title=f'{part}: decode time by trip size', xlabel='GPS points', ylabel='Seconds')
    fig.tight_layout(); fig.savefig(outdir / f'{part}_runtime.png', dpi=160); plt.close(fig)
    means = rows.groupby('method').top1_all.mean().reindex(['classifier', 'hmm', 'no_gap_relaxation', 'no_uturn_penalty'])
    fig, ax = plt.subplots(figsize=(8, 4))
    means.plot.bar(ax=ax); ax.set(title=f'{part}: component ablations', ylabel='Mean trip top-1', xlabel='')
    fig.tight_layout(); fig.savefig(outdir / f'{part}_ablations.png', dpi=160); plt.close(fig)
