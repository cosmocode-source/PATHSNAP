"""Directed-network Viterbi decoding; this module never accepts a reference route."""
import time
import numpy as np
from .network_utils import RoadGraph


def decode_trip(candidates, probs, network, gps_xy, times, config):
    """Decode candidate DataFrame rows (gps_row, arc_id), with one score per row.

    gps_xy and times are arrays indexed by gps_row. A missing first point cannot be
    carried forward and raises ValueError; later missing points retain the last arc.
    """
    started = time.perf_counter()
    xy, times = np.asarray(gps_xy, float), np.asarray(times, float)
    probs = np.asarray(probs, float)
    rows = candidates['gps_row'].to_numpy(dtype=int)
    ids = candidates['arc_id'].to_numpy(dtype=int)
    if len(probs) != len(ids) or len(xy) != len(times):
        raise ValueError('candidate scores and GPS arrays must align')
    if not np.isfinite(probs).all() or np.any(probs < 0):
        raise ValueError('probabilities must be finite and nonnegative')
    if np.any(rows < 0) or np.any(rows >= len(times)) or np.any(np.diff(times) < 0):
        raise ValueError('invalid point indices or non-monotonic times')
    graph = RoadGraph(network, config.get('cache_size', 512), config.get('cutoff_bucket_m', 25))
    if np.any(ids < 0) or np.any(ids >= len(graph.arcs)):
        raise ValueError('candidate arc outside network')
    order = np.argsort(rows, kind='stable')
    rows, ids, probs = rows[order], ids[order], probs[order]
    cuts = np.searchsorted(rows, np.arange(len(times) + 1))
    states, emissions, baseline = [], [], []
    warnings = 0
    for t in range(len(times)):
        lo, hi = cuts[t:t+2]
        if hi == lo:
            if t == 0:
                raise ValueError('first point has no candidates to carry forward')
            states.append(None); emissions.append(None); baseline.append(None)
            warnings += 1
            continue
        p = probs[lo:hi]
        p = p / p.sum() if p.sum() else np.full(len(p), 1 / len(p))
        # Stable sort resolves ties in the original candidate order.
        top = np.argsort(-p, kind='stable')[:config['hmm_top_k']]
        states.append(ids[lo:hi][top])
        emissions.append(config['alpha'] * np.log(np.maximum(p[top], config['eps'])))
        baseline.append(int(ids[lo + int(np.argmax(p))]))
    # A missing point carries every candidate state with zero extra cost; this lets
    # the final backtrack carry the actual chosen arc, rather than a premature guess.
    scores, backs, breaks = [], [], []
    n_unreachable = n_gaps = 0
    for t in range(len(times)):
        if t == 0:
            scores.append(emissions[t]); backs.append(None); continue
        if states[t] is None:
            states[t] = states[t-1].copy()
            scores.append(scores[-1].copy()); backs.append(np.arange(len(states[t])))
            continue
        dt = times[t] - times[t-1]
        gap = dt > config['gap_seconds']
        n_gaps += int(gap)
        cutoff = max(config['min_cutoff_m'], config['v_max_mps'] * dt)
        beta = config['beta'] * (config['gap_beta_factor'] if gap else 1)
        prev, curr = states[t-1], states[t]
        po, co = graph.offsets(prev, xy[t-1]), graph.offsets(curr, xy[t])
        distances = np.full((len(prev), len(curr)), np.inf)
        for i, arc in enumerate(prev):
            end = graph.arcs[arc, 1]
            paths = graph.distances(end, cutoff)
            for j, dest in enumerate(curr):
                if arc == dest:
                    delta = co[j] - po[i]
                    if -config['backward_tolerance_m'] <= delta <= cutoff:
                        distances[i, j] = max(delta, 0)
                else:
                    route = graph.lengths[arc] - po[i] + paths.get(int(graph.arcs[dest, 0]), np.inf) + co[j]
                    if route <= cutoff:
                        distances[i, j] = route
        reachable = np.isfinite(distances)
        n_unreachable += int((~reachable).sum())
        if not reachable.any():
            breaks.append(t); scores.append(emissions[t]); backs.append(None); continue
        d_gps = np.linalg.norm(xy[t] - xy[t-1])
        trans = np.full(distances.shape, config['unreachable_logp'], dtype=float)
        trans[reachable] = -np.abs(distances[reachable] - d_gps) / beta
        reverse = ((graph.arcs[prev, 0, None] == graph.arcs[curr, 1][None, :]) &
                   (graph.arcs[prev, 1, None] == graph.arcs[curr, 0][None, :]) &
                   (prev[:, None] != curr[None, :]))
        trans[reverse & reachable] -= config['u_turn_penalty']
        choices = scores[-1][:, None] + trans
        parent = np.argmax(choices, axis=0)
        scores.append(choices[parent, np.arange(len(curr))] + emissions[t])
        backs.append(parent)
    if not len(times):
        return {'arc_per_point': np.array([], dtype=int), 'predicted_route': np.array([], dtype=int), 'diagnostics': {}}
    result = np.empty(len(times), dtype=int)
    idx = int(np.argmax(scores[-1]))
    for t in range(len(times)-1, -1, -1):
        result[t] = states[t][idx]
        if t:
            idx = int(np.argmax(scores[t-1])) if backs[t] is None else int(backs[t][idx])
    route = result[np.r_[True, result[1:] != result[:-1]]]
    return {'arc_per_point': result, 'predicted_route': route,
            'diagnostics': {'n_breaks': len(breaks), 'break_points': breaks,
                'n_gap_relaxed_steps': n_gaps, 'n_unreachable_pairs': n_unreachable,
                'n_points_changed': int(sum(b is not None and a != b for a, b in zip(result, baseline))),
                'n_missing_candidates': warnings, 'decode_seconds': time.perf_counter() - started}}
