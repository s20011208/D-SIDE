"""Portable actual-run summary figure; never invents illustrative observations."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ._backend.global_consensus import mixture_parts
from .io import read


def plot_run(directory):
    root = Path(directory)
    case = sorted(root.glob('*/seed_*'))[0]
    training = read(case/'training.json'); generated = read(case/'generation.json')
    selection = read(case/'density_only_selection.json')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for tr, truth in zip(training, generated):
        color = '#147d92' if tr['source'] < 6 else ('#c48036' if tr['source'] == 6 else '#a05f83')
        axes[0, 0].plot(truth['time'], truth['position'], color=color, alpha=.7, lw=1.2)
        axes[0, 0].scatter(tr['time'], tr['position'], color=color, s=8, alpha=.3)
    axes[0, 0].set(title='01 / Training trajectories', xlabel='Time', ylabel='Position')
    rows = selection.get('rows', [])
    z = np.array([r['z'] for r in rows]); w = np.array([r['global_weight'] for r in rows])
    if len(z):
        axes[0, 1].hist(z, bins=np.linspace(-np.pi/2, np.pi/2, 40), weights=w, color='#147d92', alpha=.65)
        axes[0, 1].axvspan(*selection['interval'], color='#c48036', alpha=.15, label='Descriptive peak region')
    axes[0, 1].set(title='02 / Weighted scalar evidence', xlabel='arctan(T)', ylabel='Probability mass')
    axes[0, 1].legend(frameon=False, fontsize=8)
    e = np.array(selection.get('edge_evidence', np.zeros((8, 8))), float)
    limit = max(.1, np.nanmax(abs(e)))
    im = axes[1, 0].imshow(e, cmap='RdBu', vmin=-limit, vmax=limit)
    axes[1, 0].set(title='03 / Source evidence; selected '+str(selection['core']), xlabel='Source', ylabel='Source', xticks=range(8), yticks=range(8))
    fig.colorbar(im, ax=axes[1, 0], shrink=.75)
    group = pd.read_csv(root/'group_metrics.csv'); group = group.groupby(['method', 'group']).mse.mean().unstack('method')
    axes[1, 1].bar(np.arange(8)-.18, group.independent, width=.36, color='#bcc5cf', label='Independent')
    axes[1, 1].bar(np.arange(8)+.18, group.density_only, width=.36, color='#147d92', label='D-SIDE')
    axes[1, 1].set(title='04 / Held-out prediction', xlabel='Source', ylabel='Position MSE', xticks=range(8))
    axes[1, 1].legend(frameon=False)
    fig.suptitle('D-SIDE | Distributional Selective Integration for Dynamics Estimation', fontsize=14, color='#18334c')
    fig.savefig(root/'overview.png', dpi=180)
    fig.savefig(root/'overview.svg')
    plt.close(fig)
    return root/'overview.png'
