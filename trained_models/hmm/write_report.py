"""Write report answers from completed evaluation; never fabricate results."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
data = pd.read_csv(HERE / 'results/per_trip_metrics.csv', dtype={'trip_id': str})
summary = json.loads((HERE / 'results/summary.json').read_text())
cfg = json.loads((HERE / 'config.json').read_text())
pruning = pd.read_csv(HERE / 'results/pruning.csv')
topology = pd.read_csv(HERE / 'results/no_help_topology.csv', dtype={'trip_id': str}).set_index(['partition', 'trip_id'])
lines = ['# Report answers', '',
    '1. **Hidden states.** Each state is a candidate directed road arc for one GPS point. The result chooses one arc for every point.', '',
    '2. **Emissions.** The classifier scores candidate arcs; the scores are normalised within each point. The decoder takes the log of each probability after flooring it and multiplies it by alpha.', '',
    '3. **Transitions.** The decoder compares directed network distance between projected candidate positions with straight-line GPS distance. It penalises disagreement and reverse-direction turns. Short backward motion on the same arc is treated as stationary jitter.', '',
    '4. **Missing candidates and gaps.** A point without candidates carries the previous state and is counted. A first point without candidates cannot be carried and raises an error. Gaps use the actual elapsed time, a larger search bound, and a relaxed distance penalty.', '',
    '5. **Disconnected observations.** If no candidate pair is reachable, the decoder records a break and restarts from the next emissions. If only some pairs are unreachable, those pairs receive a very low score.', '',
    '6. **Did it improve the route?** The results below compare classifier-only and HMM predictions. Training uses out-of-fold scores; student-test was reserved until parameters were frozen. Confidence intervals resample whole trips, so they describe uncertainty across trips.', '']
for part, values in summary.items():
    lines.append(f'## {part}')
    for metric, v in values.items():
        if 'classifier_trip_mean' not in v:
            lines.append(f"- {metric}, pooled over points: classifier {v['classifier_point_weighted']:.4f}, HMM {v['hmm_point_weighted']:.4f}.")
            continue
        lo, hi = v['bootstrap_95_ci']
        support = 'interval excludes zero' if lo > 0 or hi < 0 else 'interval includes zero'
        lines.append(f"- {metric}: classifier {v['classifier_trip_mean']:.4f}, HMM {v['hmm_trip_mean']:.4f}, difference {v['paired_difference']:+.4f}; 95% paired trip bootstrap interval [{lo:+.4f}, {hi:+.4f}] ({support}).")
        if 'classifier_point_weighted' in v:
            lines.append(f"  Pooled over points: classifier {v['classifier_point_weighted']:.4f}, HMM {v['hmm_point_weighted']:.4f}.")
    lines.append('')
    subset = data[data.partition == part]
    means = subset.groupby('method').top1_all.mean()
    lines.append('Ablations (mean trip top-1): ' + ', '.join(f'{name} {value:.4f}' for name, value in means.items()) + '. The classifier is also the emission-only result.')
    largest = subset[subset.method == 'hmm'].sort_values('n_points').iloc[-1]
    lines.append(f"Largest trip: {largest.trip_id}, {int(largest.n_points)} points, {largest.decode_seconds:.3f} seconds to decode.")
    p = pruning[pruning.partition == part]
    lost, total = int(p.label_lost_top_k.sum()), int(p.label_in_candidates.sum())
    lines.append(f'Pruning removed the label for {lost} of {total} points whose label was initially in the candidate set ({lost/total:.4%}).')
    lines.append('')
lines += ['## Where HMM did not help', '', 'The notes below are characteristics associated with each trip, not proven causes. Differences are in all-point top-1.', '']
for part, rows in data.groupby('partition'):
    a = rows[rows.method == 'classifier'].set_index('trip_id')
    b = rows[rows.method == 'hmm'].set_index('trip_id').loc[a.index]
    for trip_id in a.index:
        delta = b.loc[trip_id, 'top1_all'] - a.loc[trip_id, 'top1_all']
        if delta <= 0:
            g, f, miss, stationary, sparse = b.loc[trip_id, ['gap_share', 'far_share', 'candidate_miss_rate', 'stationary_share', 'sparse_step_share']]
            reason = max([(g, 'gap share'), (f, 'far-label share'), (miss, 'candidate misses'),
                          (stationary, 'stationary-point share'), (sparse, 'steps over 50 m')], key=lambda x: x[0])
            detail = f'{reason[1]} {reason[0]:.3f} (possible contributing factor)' if reason[0] > 0 else 'no cause established by these indicators'
            topo = topology.loc[(part, trip_id)]
            lines.append(f"- {part} {trip_id}: {delta:+.4f}; {detail}. Reference repeats {int(topo.reference_repeated_arc_count)} arcs; largest weak road component contains {topo.largest_weak_component_node_share:.1%} of indexed nodes.")
lines += ['', '## Interpretation notes', '',
    'Route precision, recall and F1 use the longest common subsequence against the offline reference route. Length mismatch uses the lengths of arcs in the multiset symmetric difference divided by reference length; it ignores arc order and can exceed one. Consecutive GPS observations can skip intermediate arcs, so the collapsed per-point prediction does not reconstruct every traversed arc. Repeated reference arcs and weak-component sizes are proxies for loops and network fragmentation, not proof that either caused an error. Listed trip characteristics are associations rather than established causes.', '',
    f"Top-k pruning retained {cfg['hmm_top_k']} candidates per point; counts of reference-label losses are in pruning.csv. Per-trip diagnostics and ablations are in per_trip_metrics.csv."]
(HERE / 'results/REPORT_ANSWERS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
