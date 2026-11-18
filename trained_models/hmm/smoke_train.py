"""Smoke-test the untuned decoder on training trips; not an accuracy estimate."""
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'trained_models'))
sys.path.insert(0, str(ROOT / 'data_preprocessing' / 'data_preprocessing'))
import lightgbm as lgb
import numpy as np
import pandas as pd
from pathsnap_prep import load_trip
from common.data_loading import feature_list
from trained_models.hmm.hmm_decoder import decode_trip

cfg = json.loads((Path(__file__).parent / 'config.json').read_text())
table = pd.read_parquet(ROOT / 'data_preprocessing/cleaned_data/model_data/train_candidates.parquet')
model = lgb.Booster(model_file=str(ROOT / cfg['model_path']))
sizes = table.groupby('trip_id').gps_row.nunique().sort_values()
results = []
for trip_id in sizes.index[:3]:
    part = table[table.trip_id == trip_id].reset_index(drop=True)
    trip = load_trip(ROOT / 'Dataset', 'train', trip_id)
    scores = model.predict(part[feature_list()], num_threads=2)
    out = decode_trip(part[['gps_row', 'arc_id']], scores, trip, trip.gps_xy, trip.gps_t, cfg)
    results.append({'trip_id': trip_id, 'n_points': len(trip.gps_t), **out['diagnostics']})
    print(results[-1], flush=True)
outdir = Path(__file__).parent / 'results'
outdir.mkdir(exist_ok=True)
(outdir / 'smoke_train.json').write_text(json.dumps(results, indent=2))
