#!/usr/bin/env python3
"""Construct two documented, label-reserved anomaly datasets from UCI.

Install `ucimlrepo pandas numpy scikit-learn`. Requires network access during
first fetch. Original labels are used ONLY for evaluating unsupervised methods.
Writes headerless features + binary label, as expected by upstream demo.py.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(2 ** 20), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare(name, dataset_id, positive_label, seed, target_fraction, out_dir):
    from ucimlrepo import fetch_ucirepo  # installed by the student

    ds = fetch_ucirepo(id=dataset_id)
    X = ds.data.features.copy()
    target = ds.data.targets.iloc[:, 0].copy()
    source_rows = len(X)
    X = X.apply(pd.to_numeric, errors='raise')
    frame = X.reset_index(drop=True)
    frame['_source_label'] = target.astype(str).str.strip().str.upper().reset_index(drop=True)
    positive_label = str(positive_label).upper()
    if positive_label not in set(frame['_source_label']):
        raise ValueError(f'{name}: positive label {positive_label!r} not in {sorted(set(frame["_source_label"]))}')
    if frame.isna().any().any() or not np.isfinite(frame[X.columns].values).all():
        raise ValueError(f'{name}: missing or nonfinite source data; inspect manually first')
    contradictions = frame.groupby(list(X.columns), dropna=False)['_source_label'].nunique()
    if (contradictions > 1).any():
        raise ValueError(f'{name}: duplicate feature vectors have inconsistent labels')
    duplicated = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().reset_index(drop=True)
    y = (frame['_source_label'] == positive_label)
    if y.sum() == 0 or (~y).sum() == 0:
        raise ValueError(f'{name}: both classes required')
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y)[0]
    neg_idx = np.where(~y)[0]
    # Keeping all available normals and sampling positives defines a reproducible
    # benchmark at approximately the desired contamination. This is not a natural prevalence.
    max_pos = int(np.floor(target_fraction / (1.0 - target_fraction) * len(neg_idx)))
    n_pos = min(len(pos_idx), max_pos)
    if n_pos < 2:
        raise ValueError(f'{name}: insufficient positives at {target_fraction} target fraction')
    chosen = np.concatenate([neg_idx, rng.choice(pos_idx, size=n_pos, replace=False)])
    rng.shuffle(chosen)
    chosen_X = frame.iloc[chosen][list(X.columns)].to_numpy(dtype=float)
    labels = (frame.iloc[chosen]['_source_label'].to_numpy() == positive_label).astype(int)
    # Standardisation is transductive (fit on unlabeled complete matrix), matching
    # the same-data fit/score protocol. For a held-out protocol fit on train only.
    scaled = StandardScaler().fit_transform(chosen_X)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f'{name}.csv'
    pd.DataFrame(np.column_stack([scaled, labels])).to_csv(csv_path, header=False, index=False, float_format='%.10g')
    preview_path = out_dir / f'{name}_preview.csv'
    preview = pd.DataFrame(chosen_X[:5], columns=X.columns)
    preview['evaluation_label'] = labels[:5]
    preview.to_csv(preview_path, index=False)
    meta = {
        'dataset': name, 'source_dataset_id': dataset_id, 'source_url': f'https://archive.ics.uci.edu/dataset/{dataset_id}',
        'retrieved_utc': datetime.now(timezone.utc).isoformat(), 'source_rows': source_rows,
        'removed_exact_duplicate_rows': duplicated, 'cleaned_rows': len(frame),
        'features': list(X.columns), 'numeric_features': len(X.columns),
        'positive_source_label': positive_label, 'positive_definition': 'pseudo-anomaly' if name == 'drybean_sira' else 'forged specimen',
        'negatives': int((labels == 0).sum()), 'positives': int((labels == 1).sum()),
        'fraction_positive': float(labels.mean()), 'target_fraction_positive': target_fraction,
        'seed': seed, 'preprocessing': 'drop exact duplicates; sample positives to 5%; StandardScaler fitted to unlabeled whole matrix',
        'evaluation_protocol': 'transductive fit and score on the same observations; labels held out of training features',
        'sha256_csv': sha256(csv_path)
    }
    (out_dir / f'{name}_manifest.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(f'{name}: {len(labels)} rows, {len(X.columns)} features, {int(labels.sum())} positives ({100*labels.mean():.2f}%), {duplicated} duplicates removed')
    print(f'  -> {csv_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=Path('data_new'))
    ap.add_argument('--seed', type=int, default=869)
    ap.add_argument('--prevalence', type=float, default=0.05)
    args = ap.parse_args()
    if not 0 < args.prevalence < 0.5:
        ap.error('prevalence must be between 0 and 0.5')
    prepare('banknote', 267, '1', args.seed, args.prevalence, args.output)
    prepare('drybean_sira', 602, 'SIRA', args.seed, args.prevalence, args.output)

if __name__ == '__main__':
    main()
