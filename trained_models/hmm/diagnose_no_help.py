"""Inspect reference repeats and weak road components for non-improving trips."""
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
data = pd.read_csv(HERE / 'results/per_trip_metrics.csv', dtype={'trip_id': str})
base = data[data.method == 'classifier'].set_index(['partition', 'trip_id'])
hmm = data[data.method == 'hmm'].set_index(['partition', 'trip_id']).reindex(base.index)
out = []
for (partition, trip_id), row in hmm.iterrows():
    difference = row.top1_all - base.loc[(partition, trip_id), 'top1_all']
    if difference > 0:
        continue
    directory = ROOT / 'Dataset' / ('train' if partition == 'train' else 'student_test') / trip_id
    arcs = pd.read_csv(directory / f'{trip_id}.arcs', sep=r'\s+', header=None).to_numpy(dtype=int)
    ref = pd.read_csv(directory / f'{trip_id}.route', header=None).iloc[:, 0].to_numpy(dtype=int)
    n = int(arcs.max()) + 1
    graph = coo_matrix((np.ones(len(arcs), dtype=np.int8), (arcs[:, 0], arcs[:, 1])), shape=(n, n)).tocsr()
    n_components, component = connected_components(graph, directed=False)
    sizes = np.bincount(component)
    out.append({'partition': partition, 'trip_id': trip_id, 'top1_difference': difference,
                'reference_repeated_arc_count': int(len(ref) - len(np.unique(ref))),
                'weak_component_count': int(n_components),
                'largest_weak_component_node_share': float(sizes.max()/n)})
    print(f'Diagnosed {partition} {trip_id}', flush=True)
pd.DataFrame(out, columns=['partition', 'trip_id', 'top1_difference', 'reference_repeated_arc_count',
                           'weak_component_count', 'largest_weak_component_node_share']).to_csv(HERE / 'results/no_help_topology.csv', index=False)
