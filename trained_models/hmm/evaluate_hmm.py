"""Evaluate frozen parameters on OOF train and then student data, once."""
import json
import hashlib
import os
import sys
from collections import Counter
from pathlib import Path
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'trained_models'))
sys.path.insert(0, str(ROOT / 'data_preprocessing/data_preprocessing'))
from pathsnap_prep import load_trip, load_route
from common.data_loading import feature_list
from common.metrics import evaluate_with_clean, lcs_length, point_ids, top_rows
from trained_models.hmm.hmm_decoder import decode_trip

HERE = Path(__file__).parent
cfg = json.loads((HERE / 'config.json').read_text())
result_path = HERE / 'results/per_trip_metrics.csv'
pruning_path = HERE / 'results/pruning.csv'
signature_path = HERE / 'results/evaluation.sha256'
signature = hashlib.sha256((json.dumps(cfg, sort_keys=True) +
    hashlib.sha256((ROOT / cfg['model_path']).read_bytes()).hexdigest() +
    hashlib.sha256((HERE / 'results/oof_predictions.parquet').read_bytes()).hexdigest()).encode()).hexdigest()
if result_path.exists():
    if not pruning_path.exists() or not signature_path.exists() or signature_path.read_text().strip() != signature:
        raise ValueError('Existing evaluation results have different inputs or incomplete checkpoints')
    records = pd.read_csv(result_path, dtype={'trip_id': str}).to_dict('records')
    pruning = pd.read_csv(pruning_path, dtype={'trip_id': str}).to_dict('records')
else:
    records, pruning = [], []
signature_path.write_text(signature + '\n')
completed = {(r['partition'], r['trip_id']) for r in records if r['method'] == 'hmm'}
model = lgb.Booster(model_file=str(ROOT / cfg['model_path']))
partitions = ['train'] if '--train-only' in sys.argv else ['train', 'student']
for partition in partitions:
    raw = pd.read_parquet(ROOT / f'data_preprocessing/cleaned_data/model_data/{partition}_candidates.parquet')
    raw = raw.sort_values(['trip_id', 'gps_row'], kind='stable').reset_index(drop=True)
    if partition == 'train':
        oof = pd.read_parquet(HERE / 'results/oof_predictions.parquet')
        keys = ['trip_id', 'point_id', 'gps_row', 'arc_id']
        if not raw[keys].equals(oof[keys]):
            raise ValueError('OOF rows do not match the candidate table')
        raw['probability'] = oof.probability
    else:
        raw['probability'] = model.predict(raw[feature_list()], num_threads=2)
    for trip_id, part in raw.groupby('trip_id', sort=True):
        if (partition, trip_id) in completed:
            continue
        part = part.reset_index(drop=True)
        split = 'train' if partition == 'train' else 'student_test'
        trip = load_trip(ROOT / 'Dataset', split, trip_id)
        # Reference routes are read here in offline evaluation only; never passed to decoder.
        reference = load_route(ROOT / 'Dataset', split, trip_id, trip.n_arcs)
        probs = part.probability.to_numpy()
        pid = point_ids(part)
        top = top_rows(probs, pid)
        classifier = part.arc_id.to_numpy()[top]
        candidate_ids = part[['gps_row', 'arc_id']]
        hmm = decode_trip(candidate_ids, probs, trip, trip.gps_xy, trip.gps_t, cfg)
        if len(classifier) != len(trip.gps_t):
            raise ValueError(f'{trip_id}: missing candidates require special evaluation accounting')
        rankings = part.assign(_p=probs).sort_values(['gps_row', '_p'], ascending=[True, False], kind='stable')
        rankings['_rank'] = rankings.groupby('gps_row').cumcount()
        positive = rankings[rankings.label == 1]
        pruning.append({'partition': partition, 'trip_id': trip_id, 'label_in_candidates': len(positive),
                        'label_lost_top_k': int((positive._rank >= cfg['hmm_top_k']).sum())})
        # Emission-only equals classifier by construction. Other ablations use the
        # same frozen settings and same observations, changing one component each.
        variants = {'classifier': classifier, 'hmm': hmm['arc_per_point']}
        for name, change in [('no_gap_relaxation', {'gap_beta_factor': 1.}), ('no_uturn_penalty', {'u_turn_penalty': 0.})]:
            variants[name] = decode_trip(candidate_ids, probs, trip, trip.gps_xy, trip.gps_t, {**cfg, **change})['arc_per_point']
        for name, pred in variants.items():
            selected = (part.arc_id.to_numpy() == pred[part.gps_row.to_numpy(dtype=int)]).astype(float)
            metrics = evaluate_with_clean(part, part.label.to_numpy(), selected, part.label_quality.to_numpy())
            route = pred[np.r_[True, pred[1:] != pred[:-1]]]
            n_pairs = max(0, len(route)-1)
            connected = int((trip.arcs[route[:-1], 1] == trip.arcs[route[1:], 0]).sum())
            lcs = lcs_length(route, reference)
            # Sum of lengths in the multiset symmetric difference divided by the
            # reference length. Arc multiplicities are preserved; order is not.
            pc, rc = Counter(route.tolist()), Counter(reference.tolist())
            mismatch = sum(abs(pc[a]-rc[a]) * trip.arc_len[a] for a in pc.keys() | rc.keys())
            ref_length = float(trip.arc_len[reference].sum())
            info = hmm['diagnostics'] if name == 'hmm' else {}
            records.append({'partition': partition, 'trip_id': trip_id, 'method': name,
                'n_points': len(pred), 'top1_all': metrics['all']['top1_all'],
                'top1_in_candidates': metrics['all']['top1_in_cands'],
                'top1_clean': metrics['clean']['top1_all'] if metrics['clean'] else np.nan,
                'candidate_miss_rate': metrics['all']['miss_rate'],
                'continuity': connected/n_pairs if n_pairs else np.nan, 'broken_transitions': n_pairs-connected,
                'arc_precision_vs_reference': lcs/len(route) if len(route) else np.nan,
                'arc_recall_vs_reference': lcs/len(reference) if len(reference) else np.nan,
                'arc_f1_vs_reference': 2*lcs/(len(route)+len(reference)),
                'length_mismatch_vs_reference': mismatch/ref_length if ref_length else np.nan,
                'gap_share': float(part.drop_duplicates('gps_row').after_gap.mean()),
                'far_share': float((part.drop_duplicates('gps_row').label_quality == 'far').mean()),
                **info})
        print(f'Evaluated {partition} {trip_id}', flush=True)
        pd.DataFrame(records).to_csv(result_path, index=False)
        pd.DataFrame(pruning).to_csv(pruning_path, index=False)
        del trip
    del raw
result = pd.DataFrame(records)
result.to_csv(result_path, index=False)
pd.DataFrame(pruning).to_csv(pruning_path, index=False)
rng = np.random.default_rng(42)
summary = {}
for partition, part in result.groupby('partition'):
    a = part[part.method == 'classifier'].set_index('trip_id')
    b = part[part.method == 'hmm'].set_index('trip_id').loc[a.index]
    summary[partition] = {}
    for metric in ['top1_all', 'top1_in_candidates', 'continuity']:
        delta = (b[metric]-a[metric]).dropna().to_numpy()
        boot = np.mean(rng.choice(delta, size=(2000, len(delta)), replace=True), axis=1)
        summary[partition][metric] = {'classifier_trip_mean': float(a[metric].mean()),
            'hmm_trip_mean': float(b[metric].mean()), 'paired_difference': float(delta.mean()),
            'bootstrap_95_ci': np.quantile(boot, [.025, .975]).tolist()}
(HERE / 'results/summary.json').write_text(json.dumps(summary, indent=2))
