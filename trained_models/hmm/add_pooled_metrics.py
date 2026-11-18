"""Add point-weighted headline metrics to the per-trip bootstrap summary."""
import json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
data = pd.read_csv(HERE / 'results/per_trip_metrics.csv', dtype={'trip_id': str})
summary_path = HERE / 'results/summary.json'
summary = json.loads(summary_path.read_text())
for partition in data.partition.unique():
    points = pd.read_parquet(ROOT / f'data_preprocessing/cleaned_data/model_data/points_{partition}.parquet')
    clean = points.assign(clean=points.label_quality == 'clean').groupby('trip_id').clean.sum()
    subset = data[data.partition == partition]
    for method in ['classifier', 'hmm']:
        part = subset[subset.method == method].set_index('trip_id')
        weights = {'top1_all': part.n_points,
                   'top1_in_candidates': (part.n_points * (1 - part.candidate_miss_rate)).round(),
                   'top1_clean': clean.reindex(part.index)}
        for metric, weight in weights.items():
            valid = part[metric].notna() & (weight > 0)
            value = float((part.loc[valid, metric] * weight[valid]).sum() / weight[valid].sum())
            summary[partition].setdefault(metric, {})[f'{method}_point_weighted'] = value
summary_path.write_text(json.dumps(summary, indent=2) + '\n')
