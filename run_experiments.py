#!/usr/bin/env python3
"""Evaluate published OptIForest or a transparent iForest comparator.

Run within a clone of https://github.com/xiagll/OptIForest (or provide
--upstream) after placing CSVs in a directory. Both scores are oriented so
higher == more anomalous. Writes RAW per-seed runs, not fictitious summaries.
"""
import argparse
import csv
import importlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

FIELDS = ['dataset', 'model', 'seed', 'threshold', 'branch', 'trees', 'n_rows', 'n_features',
          'n_positive', 'roc_auc', 'average_precision', 'fit_seconds', 'score_seconds']


def run_dataset(file, model_names, seeds, threshold, branch, trees, upstream, output):
    table = pd.read_csv(file, header=None)
    if table.shape[1] < 3 or table.isna().any().any():
        raise ValueError(f'Invalid dataset: {file}')
    X = table.iloc[:, :-1].to_numpy(dtype=float)
    y = table.iloc[:, -1].to_numpy(dtype=int)
    if not np.isfinite(X).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError(f'Input must contain finite numerical features and BOTH binary labels: {file}')
    if 'OptIForest' in model_names:
        if not upstream.exists():
            raise FileNotFoundError(f'Clone upstream repository and supply --upstream: {upstream}')
        sys.path.insert(0, str(upstream.resolve()))
        OptIForest = getattr(importlib.import_module('detectors'), 'OptIForest')
    else:
        OptIForest = None
    output.parent.mkdir(parents=True, exist_ok=True)
    is_new = not output.exists() or output.stat().st_size == 0
    with output.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            w.writeheader()
        for name in model_names:
            for seed in seeds:
                np.random.seed(seed)
                random.seed(seed)
                if name == 'OptIForest':
                    detector = OptIForest('L2OPT', trees, threshold, branch)
                else:
                    detector = IsolationForest(n_estimators=trees, max_samples=min(256, len(X)),
                                               random_state=seed, n_jobs=1)
                start = time.perf_counter()
                detector.fit(X)  # transductive: matches official demo method
                fit_seconds = time.perf_counter() - start
                start = time.perf_counter()
                anomaly = -np.asarray(detector.decision_function(X), dtype=float)
                score_seconds = time.perf_counter() - start
                row = {'dataset': file.stem, 'model': name, 'seed': seed,
                       'threshold': threshold if name == 'OptIForest' else '',
                       'branch': branch if name == 'OptIForest' else '',
                       'trees': trees, 'n_rows': len(y), 'n_features': X.shape[1],
                       'n_positive': int(y.sum()),
                       'roc_auc': float(roc_auc_score(y, anomaly)),
                       'average_precision': float(average_precision_score(y, anomaly)),
                       'fit_seconds': fit_seconds, 'score_seconds': score_seconds}
                w.writerow(row)
                f.flush()
                print(f'{file.stem:16s} {name:12s} seed={seed} ROC={row["roc_auc"]:.4f} AP={row["average_precision"]:.4f}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True, help='headerless features+label CSV')
    p.add_argument('--models', choices=['both', 'OptIForest', 'iForest'], default='both')
    p.add_argument('--upstream', type=Path, default=Path('OptIForest'))
    p.add_argument('--threshold', type=int, default=403, help='not necessarily optimal for all datasets')
    p.add_argument('--branch', type=int, default=0, help='upstream branch=0 selects expected e')
    p.add_argument('--trees', type=int, default=100)
    p.add_argument('--seeds', default=','.join(str(869 + i) for i in range(15)))
    p.add_argument('--out', type=Path, default=Path('results/experiment_runs.csv'))
    args = p.parse_args()
    seeds = [int(v) for v in args.seeds.split(',') if v.strip()]
    models = ['OptIForest', 'iForest'] if args.models == 'both' else [args.models]
    run_dataset(args.data, models, seeds, args.threshold, args.branch, args.trees, args.upstream, args.out)

if __name__ == '__main__':
    main()
