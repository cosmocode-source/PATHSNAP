"""Compare saved predictions and existing HMM results without reselecting models."""
import os
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
import sys
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
HERE = REPO
sys.path[:0] = [str(REPO / 'trained_models'), str(REPO / 'data_preprocessing/data_preprocessing')]
from common.metrics import evaluate_with_clean, lcs_length, point_ids, top_rows, classification_from_counts
from pathsnap_prep import load_trip, load_route

OUT = REPO / 'trained_models/comparison/final_comparison'
OUT.mkdir(exist_ok=True)
if (OUT / 'comparison_summary.json').exists():
    raise FileExistsError('Final comparison already exists; preserve the original results')
KEYS = ['trip_id', 'point_id', 'gps_row', 'arc_id']
original = pd.read_csv(REPO / 'trained_models/hmm/results/per_trip_metrics.csv', dtype={'trip_id':str})
folds = pd.read_csv(REPO / 'data_preprocessing/cleaned_data/model_data/folds.csv', dtype={'trip_id':str}).set_index('trip_id')['fold']
rows, global_metrics, point_counts, hard_normalisers = [], {}, [], {}
for split in ('train', 'student'):
    raw = pd.read_parquet(REPO / f'data_preprocessing/cleaned_data/model_data/{split}_candidates.parquet',
                          columns=KEYS + ['label','label_quality','fold'])
    raw = raw.sort_values(['trip_id','gps_row'], kind='stable').reset_index(drop=True)
    sizes = raw.groupby(['trip_id','gps_row']).label.agg(n='size', positive='sum')
    positive_sizes = sizes.loc[sizes.positive > 0, 'n'].to_numpy()
    eps = 1e-15  # Match common.metrics.EPS for clipped one-hot decisions.
    hard_normalisers[split] = float(np.log((1-eps)+(positive_sizes-1)*eps).sum())
    if split == 'train':
        one = pd.read_parquet(REPO / 'trained_models/hmm/results/oof_predictions.parquet')
        two = pd.read_parquet(HERE / 'trained_models/final_model/variants/tuned/oof_predictions.parquet')
        p1, p2 = one.probability.to_numpy(), two.probability.to_numpy()
    else:
        one = pd.read_parquet(REPO / 'trained_models/model4/student_predictions.parquet')
        two = pd.read_parquet(HERE / 'trained_models/final_model/variants/tuned/student_predictions.parquet')
        p1, p2 = one.p_seed42.to_numpy(), two.p_seed42.to_numpy()
    if not raw[KEYS].equals(one[KEYS]) or not raw[KEYS].equals(two[KEYS]):
        raise ValueError(f'{split}: saved prediction keys do not match candidate rows')
    raw['p1'], raw['p2'] = p1, p2
    for name, prob in [('S1',p1),('S2',p2)]:
        m = evaluate_with_clean(raw, raw.label, prob, raw.label_quality)
        global_metrics[f'{split}_{name}'] = {'macro_f1':m['all']['macro_f1'], 'logloss_norm':m['all']['logloss_norm'],
              'top1_all':m['all']['top1_all'],'top1_in_candidates':m['all']['top1_in_cands'],
              'top1_clean':m['clean']['top1_all'], 'miss_rate':m['all']['miss_rate']}
    for trip_id, part in raw.groupby('trip_id', sort=True):
        part = part.reset_index(drop=True)
        data_split = 'train' if split == 'train' else 'student_test'
        trip = load_trip(REPO / 'Dataset', data_split, trip_id)
        # Reference routes are only used here for offline scoring.
        ref = load_route(REPO / 'Dataset', data_split, trip_id, trip.n_arcs)
        prob = part.p2.to_numpy()
        selected = part.iloc[top_rows(prob, point_ids(part))]
        predicted = selected.arc_id.to_numpy(dtype=int)
        route = predicted[np.r_[True, predicted[1:] != predicted[:-1]]]
        pairs = max(0,len(route)-1)
        connected = int((trip.arcs[route[:-1],1] == trip.arcs[route[1:],0]).sum())
        common = lcs_length(route, ref)
        pc, rc = Counter(route.tolist()), Counter(ref.tolist())
        mismatch = sum(abs(pc[a]-rc[a])*trip.arc_len[a] for a in pc.keys() | rc.keys())
        ref_length = float(trip.arc_len[ref].sum())
        metric = evaluate_with_clean(part, part.label, prob, part.label_quality)
        positive = int(part.label.sum())
        npoints = len(predicted)
        point_counts.append({'partition':split,'trip_id':trip_id,'n_candidates':len(part),
            'n_points':npoints,'n_positive':positive,'n_clean':int((selected.label_quality == 'clean').sum()),
            'fold':int(folds[trip_id]) if split == 'train' else -1})
        rows.append({'partition':split,'trip_id':trip_id,'method':'tuned_classifier','n_points':npoints,
          'top1_all':metric['all']['top1_all'],'top1_in_candidates':metric['all']['top1_in_cands'],
          'top1_clean':metric['clean']['top1_all'] if metric['clean'] else np.nan,
          'candidate_miss_rate':metric['all']['miss_rate'],'continuity':connected/pairs if pairs else np.nan,
          'broken_transitions':pairs-connected,'arc_precision_vs_reference':common/len(route) if len(route) else np.nan,
          'arc_recall_vs_reference':common/len(ref) if len(ref) else np.nan,
          'arc_f1_vs_reference':2*common/(len(route)+len(ref)),
          'length_mismatch_vs_reference':mismatch/ref_length if ref_length else np.nan,
          'candidate_macro_f1':metric['all']['macro_f1'],'logloss_norm':metric['all']['logloss_norm']})
        print(f'{split} {trip_id}', flush=True)
        del trip
    del raw, one, two

S2 = pd.DataFrame(rows)
counts = pd.DataFrame(point_counts)
S2.to_csv(OUT/'tuned_classifier_per_trip.csv', index=False)
counts.to_csv(OUT/'point_counts.csv', index=False)
data = original[original.method.isin(['classifier','hmm'])].copy()
data['system'] = data.method.map({'classifier':'S1','hmm':'S3'})
S2['system'] = 'S2'
allrows = pd.concat([data,S2], ignore_index=True).merge(counts, on=['partition','trip_id'], validate='many_to_one', suffixes=('','_counts'))
allrows['n_points'] = allrows.n_points.astype(int)
allrows.to_csv(OUT/'comparison_per_trip.csv', index=False)
for split in ('train','student'):
    for system in ('S1','S2','S3'):
        q = allrows[(allrows.partition == split) & (allrows.system == system)].copy()
        if system == 'S3':
            # One arc is selected per point. Candidate confusion counts follow
            # exactly from candidate rows, positive labels, and point-level hits.
            tp = int(np.rint((q.top1_all*q.n_points).sum()))
            fp = int(q.n_points.sum()) - tp
            fn = int(q.n_positive.sum()) - tp
            tn = int(q.n_candidates.sum()) - tp - fp - fn
            cm = classification_from_counts(tn,fp,fn,tp)
            macro = (cm['f1_0'] + cm['f1_1']) / 2
            global_metrics[f'{split}_S3'] = {'macro_f1':macro, 'logloss_norm':None,
                'hard_decision_logloss_norm':float((hard_normalisers[split]-tp*np.log(1-eps)-fn*np.log(eps))/(tp+fn)),
                'top1_all':float(np.average(q.top1_all,weights=q.n_points)),
                'top1_in_candidates':float(np.average(q.top1_in_candidates,weights=q.n_positive)),
                'top1_clean':float(np.average(q.loc[q.top1_clean.notna() & (q.n_clean > 0), 'top1_clean'], weights=q.loc[q.top1_clean.notna() & (q.n_clean > 0), 'n_clean'])),
                'miss_rate':1-float(q.n_positive.sum()/q.n_points.sum())}
        g = global_metrics[f'{split}_{system}']
        for metric in ['continuity','arc_precision_vs_reference','arc_recall_vs_reference','arc_f1_vs_reference','length_mismatch_vs_reference']:
            g[metric+'_trip_mean'] = float(q[metric].mean())
        g['broken_transitions'] = int(q.broken_transitions.sum())
        if system == 'S3':
            g.update(points_changed=int(q.n_points_changed.sum()), breaks=int(q.n_breaks.sum()),
                     gap_steps=int(q.n_gap_relaxed_steps.sum()),decode_seconds=float(q.decode_seconds.sum()),
                     decode_ms_per_1000=float(q.decode_seconds.sum()/q.n_points.sum()*1e6))
            largest = q.loc[q.n_points.idxmax()]
            g['largest_trip'] = {'trip_id':largest.trip_id,'points':int(largest.n_points),'seconds':float(largest.decode_seconds)}
        if split == 'train':
            # Match the comparison's CV convention: point-weighted within each fold,
            # then mean and sample standard deviation across the five folds.
            for metric,weight in [('top1_all','n_points'),('top1_in_candidates','n_positive')]:
                per_fold = [float(np.average(f[metric],weights=f[weight])) for _,f in q.groupby('fold')]
                g['cv_'+metric+'_mean'] = float(np.mean(per_fold))
                g['cv_'+metric+'_std'] = float(np.std(per_fold,ddof=1))

rng = np.random.default_rng(42)
paired = {}
for split in ('train','student'):
    q = allrows[allrows.partition == split]
    for lhs, rhs in [('S2','S1'),('S3','S1')]:
        a = q[q.system==lhs].set_index('trip_id')
        b = q[q.system==rhs].set_index('trip_id').loc[a.index]
        paired[f'{split}_{lhs}_minus_{rhs}'] = {}
        for metric in ('top1_all','top1_in_candidates','continuity'):
            delta = (a[metric]-b[metric]).dropna().to_numpy()
            boot = np.mean(rng.choice(delta,size=(2000,len(delta)),replace=True),axis=1)
            paired[f'{split}_{lhs}_minus_{rhs}'][metric] = {'mean_trip_difference':float(delta.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'n_trips':len(delta)}
(OUT/'comparison_summary.json').write_text(json.dumps({'metrics':global_metrics,'paired':paired},indent=2))
print('Completed comparison calculations', flush=True)
