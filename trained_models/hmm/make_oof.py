"""Regenerate untuned LightGBM out-of-fold probabilities with two CPU threads."""
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'trained_models'))
from common.data_loading import load_candidates, fit_mask
from model4 import train_lightgbm as training

training.PARAMS['num_threads'] = 2
data = load_candidates('train', scaled=False)
folds = sorted(set(data.fold.tolist()))
oof = np.full(data.n, np.nan)
trips = data.keys.trip_id.to_numpy()
for fold in folds:
    mask = fit_mask(data, [f for f in folds if f != fold])
    valid = data.fold == fold
    path = Path(__file__).parent / 'results'
    path.mkdir(exist_ok=True)
    checkpoint = path / f'oof_fold{fold}.parquet'
    if checkpoint.exists():
        old = pd.read_parquet(checkpoint)
        if not old[data.keys.columns].reset_index(drop=True).equals(data.keys.loc[valid].reset_index(drop=True)):
            raise ValueError(f'Fold {fold} checkpoint keys do not match the current candidate table')
        oof[valid] = old.probability.to_numpy()
        print(f'Reused fold {fold}', flush=True)
        continue
    print(f'Training fold {fold}', flush=True)
    model, history = training.fit(data.X[mask], data.y[mask], data.w[mask], trips[mask], 42)
    oof[valid] = model.predict(data.X[valid], num_threads=2)
    del model
    # Fold checkpoints allow interrupted work to resume without losing predictions.
    part = data.keys.loc[valid].copy()
    part['probability'] = oof[valid]
    part.to_parquet(checkpoint, index=False)
    print(f'Completed fold {fold}', flush=True)
assert np.isfinite(oof).all()
out = data.keys.copy()
out['probability'] = oof
out.to_parquet(Path(__file__).parent / 'results/oof_predictions.parquet', index=False)
