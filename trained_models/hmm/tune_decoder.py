"""Tune using training-trip OOF predictions only; never read the student split."""
import itertools
import json
import hashlib
import os
import sys
from pathlib import Path
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'data_preprocessing/data_preprocessing'))
from pathsnap_prep import load_trip
from trained_models.hmm.hmm_decoder import decode_trip

HERE = Path(__file__).parent
base = json.loads((HERE / 'config.json').read_text())
raw = pd.read_parquet(ROOT / 'data_preprocessing/cleaned_data/model_data/train_candidates.parquet')
raw = raw.sort_values(['trip_id', 'gps_row'], kind='stable').reset_index(drop=True)
oof = pd.read_parquet(HERE / 'results/oof_predictions.parquet')
keys = ['trip_id', 'point_id', 'gps_row', 'arc_id']
if not raw[keys].equals(oof[keys]):
    raise ValueError('OOF rows do not match the training candidates')
raw['probability'] = oof.probability
points = raw.drop_duplicates(['trip_id', 'gps_row'])
characteristics = points.groupby('trip_id').agg(size=('gps_row', 'size'), gap=('after_gap', 'mean'),
                                                  far=('label_quality', lambda x: (x == 'far').mean()))
# Cover the range of sizes, plus the most gap-heavy and far-heavy trips.
ordered = characteristics.sort_values('size').index.to_numpy()
sample = set(ordered[np.linspace(0, len(ordered)-1, 6).round().astype(int)])
sample.update(characteristics.nlargest(2, 'gap').index)
sample.add(characteristics['far'].idxmax())
sample = sorted(sample)
# Seeded sample of a modest Cartesian space: weight, distance scale, U-turn cost,
# gap relaxation, speed floor, and pruning are each varied, not fitted on student.
space = list(itertools.product([0.5, 1., 2.], [15., 30., 60.], [0., 2., 5.],
                               [2., 3.], [50., 100.], [10, 15]))
rng = np.random.default_rng(42)
chosen = rng.choice(len(space), size=8, replace=False)
configs = []
for ix in chosen:
    cfg = base.copy()
    cfg.update(dict(zip(['alpha', 'beta', 'u_turn_penalty', 'gap_beta_factor', 'min_cutoff_m', 'hmm_top_k'], space[ix])))
    configs.append(cfg)
progress_path = HERE / 'results/tune_progress.csv'
signature = hashlib.sha256(json.dumps({'configs': configs, 'sample': sample}, sort_keys=True).encode()).hexdigest()
signature_path = HERE / 'results/tune_progress.sha256'
if progress_path.exists() and (not signature_path.exists() or signature_path.read_text().strip() != signature):
    raise ValueError('Existing tuning checkpoint belongs to a different search')
signature_path.write_text(signature + '\n')
if progress_path.exists():
    progress = pd.read_csv(progress_path, dtype={'trip_id': str}).to_dict('records')
else:
    progress = []
completed = {(x['trip_id'], int(x['trial'])) for x in progress}
for trip_id in sample:
    part = raw[raw.trip_id == trip_id].reset_index(drop=True)
    trip = load_trip(ROOT / 'Dataset', 'train', trip_id)
    positive = part[part.label == 1].drop_duplicates('gps_row').set_index('gps_row').arc_id
    for i, cfg in enumerate(configs):
        if (trip_id, i) in completed:
            continue
        out = decode_trip(part[['gps_row', 'arc_id']], part.probability.to_numpy(), trip, trip.gps_xy, trip.gps_t, cfg)
        idx = positive.index.to_numpy(dtype=int)
        correct = int((out['arc_per_point'][idx] == positive.to_numpy()).sum())
        seq = out['predicted_route']
        connected = int((trip.arcs[seq[:-1], 1] == trip.arcs[seq[1:], 0]).sum())
        progress.append({'trip_id': trip_id, 'trial': i, 'correct': correct, 'labelled': len(idx),
                         'connected': connected, 'pairs': max(0, len(seq)-1)})
        pd.DataFrame(progress).to_csv(progress_path, index=False)
    print(f'Finished subset trip {trip_id}', flush=True)
    del trip
rows = []
for i, cfg in enumerate(configs):
    counts = pd.DataFrame(progress).query('trial == @i')[['correct', 'labelled', 'connected', 'pairs']].sum().to_numpy()
    rows.append({'trial': i, **{k: cfg[k] for k in ['alpha', 'beta', 'u_turn_penalty', 'gap_beta_factor', 'min_cutoff_m', 'hmm_top_k']},
                 'top1_in_candidates': counts[0]/counts[1], 'continuity': counts[2]/counts[3] if counts[3] else np.nan,
                 'n_labelled_points': int(counts[1]), 'subset_trips': ','.join(sample)})
trials = pd.DataFrame(rows).sort_values(['top1_in_candidates', 'continuity'], ascending=False)
trials.to_csv(HERE / 'results/decoder_trials.csv', index=False)
best = configs[int(trials.iloc[0].trial)]
(HERE / 'config.json').write_text(json.dumps(best, indent=2) + '\n')
print('Selected training-only parameters:', best, flush=True)
