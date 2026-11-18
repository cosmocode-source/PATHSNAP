"""Attach observed stationary and sparse-step shares to offline trip results."""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
path = HERE / 'results/per_trip_metrics.csv'
data = pd.read_csv(path, dtype={'trip_id': str})
for partition in data.partition.unique():
    raw = pd.read_parquet(ROOT / f'data_preprocessing/cleaned_data/model_data/{partition}_candidates.parquet',
                          columns=['trip_id', 'gps_row', 'stationary', 'd_prev_m'])
    points = raw.drop_duplicates(['trip_id', 'gps_row']).copy()
    points['sparse_step'] = points.d_prev_m > 50.0
    stats = points.groupby('trip_id').agg(stationary_share=('stationary', 'mean'), sparse_step_share=('sparse_step', 'mean'))
    mask = data.partition == partition
    for column in stats.columns:
        data.loc[mask, column] = data.loc[mask, 'trip_id'].map(stats[column])
data.to_csv(path, index=False)
