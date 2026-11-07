#!/usr/bin/env python3
"""Aggregate measured runs (mean/sd) and render an experiment chart."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=Path('results/experiment_runs.csv'))
    p.add_argument('--output', type=Path, default=Path('results/summary.csv'))
    p.add_argument('--plot', type=Path, default=Path('figures/performance_comparison.pdf'))
    a = p.parse_args()
    if not a.input.exists():
        p.error('No experimental output was found. Run the experiments first.')
    raw = pd.read_csv(a.input)
    if len(raw) == 0:
        p.error('CSV has no measured runs; refusing to create invented results.')
    values = ['roc_auc', 'average_precision', 'fit_seconds', 'score_seconds']
    agg = raw.groupby(['dataset', 'model']).agg(**{
        f'{metric}_{label}': (metric, func)
        for metric in values for label, func in [('mean', 'mean'), ('sd', 'std')]
    }, runs=('seed', 'count')).reset_index()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(a.output, index=False, float_format='%.5f')
    print(agg.to_string(index=False))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    datasets = list(agg['dataset'].unique())
    models = list(agg['model'].unique())
    width = .8 / len(models)
    for j, model in enumerate(models):
        rows = agg[agg.model == model].set_index('dataset')
        means = [rows.at[d, 'roc_auc_mean'] if d in rows.index else np.nan for d in datasets]
        sds = [rows.at[d, 'roc_auc_sd'] if d in rows.index else np.nan for d in datasets]
        ax.bar(np.arange(len(datasets)) + (j + .5 - len(models)/2)*width, means,
               width, label=model, yerr=sds, capsize=2)
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_xticklabels(datasets, rotation=15)
    ax.set_ylim(0, 1)
    ax.set_ylabel('ROC-AUC (mean with SD)')
    ax.set_title('Measured anomaly-detection performance')
    ax.legend()
    fig.tight_layout()
    a.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.plot)
    print('Wrote', a.output, 'and', a.plot)

if __name__ == '__main__':
    main()
