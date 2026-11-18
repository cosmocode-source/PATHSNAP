"""Create a paired per-trip classifier/HMM table from evaluation results."""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).parent
data = pd.read_csv(HERE / 'results/per_trip_metrics.csv', dtype={'trip_id': str})
metrics = ['top1_all', 'top1_in_candidates', 'top1_clean', 'candidate_miss_rate',
           'continuity', 'broken_transitions', 'arc_precision_vs_reference',
           'arc_recall_vs_reference', 'arc_f1_vs_reference', 'length_mismatch_vs_reference']
base = data[data.method == 'classifier'].set_index(['partition', 'trip_id'])
hmm = data[data.method == 'hmm'].set_index(['partition', 'trip_id']).reindex(base.index)
out = pd.DataFrame(index=base.index)
out['n_points'] = base.n_points
for col in metrics:
    out[f'{col}_classifier'] = base[col]
    out[f'{col}_hmm'] = hmm[col]
    out[f'{col}_difference'] = hmm[col] - base[col]
for col in ['n_points_changed', 'n_breaks', 'n_gap_relaxed_steps', 'decode_seconds', 'gap_share', 'far_share']:
    out[col] = hmm[col]
out.to_csv(HERE / 'results/per_trip_comparison.csv')
