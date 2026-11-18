import inspect
import itertools
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from trained_models.hmm.hmm_decoder import decode_trip


def network():
    xy = np.array([[0., 0.], [10., 0.], [20., 0.], [1000., 0.], [1010., 0.]])
    arcs = np.array([[0, 1], [1, 2], [1, 0], [3, 4]])
    a, b = xy[arcs[:, 0]], xy[arcs[:, 1]]
    return SimpleNamespace(node_xy=xy, arcs=arcs, arc_a=a, arc_b=b, arc_len=np.linalg.norm(b-a, axis=1))


@pytest.fixture
def cfg():
    return dict(alpha=1., eps=1e-12, beta=10., u_turn_penalty=2., gap_beta_factor=3.,
                gap_seconds=5., v_max_mps=50., min_cutoff_m=0., hmm_top_k=15,
                backward_tolerance_m=2., unreachable_logp=-1e6)


def call(groups, p, xy, ts, cfg):
    df = pd.DataFrame([(t, a) for t, g in enumerate(groups) for a in g], columns=['gps_row', 'arc_id'])
    return decode_trip(df, p, network(), np.array(xy), np.array(ts), cfg)


def test_no_route_argument():
    assert 'route' not in inspect.signature(decode_trip).parameters


def test_unreachable_break_and_determinism(cfg):
    args = ([[0], [3]], [1., 1.], [[1, 0], [1001, 0]], [0, 1], cfg)
    a, b = call(*args), call(*args)
    assert a['arc_per_point'].tolist() == b['arc_per_point'].tolist() == [0, 3]
    assert a['diagnostics']['n_breaks'] == 1


def test_missing_point_carries_arc(cfg):
    out = call([[0], [], [1]], [1., 1.], [[1, 0], [5, 0], [12, 0]], [0, 1, 2], cfg)
    assert out['arc_per_point'].tolist() == [0, 0, 1]
    assert out['diagnostics']['n_missing_candidates'] == 1


def test_actual_time_changes_reachability(cfg):
    args = ([[0], [3]], [1., 1.], [[1, 0], [1001, 0]])
    assert call(*args, [0, 1], cfg)['diagnostics']['n_breaks'] == 1
    # A connected but distant path becomes feasible with enough elapsed time.
    cfg['v_max_mps'] = 2
    args = ([[0], [1]], [1., 1.], [[1, 0], [19, 0]])
    assert call(*args, [0, 1], cfg)['diagnostics']['n_breaks'] == 1
    long = call(*args, [0, 10], cfg)
    assert long['diagnostics']['n_breaks'] == 0
    assert long['diagnostics']['n_gap_relaxed_steps'] == 1


def test_backward_and_uturn(cfg):
    short = call([[0], [0]], [1., 1.], [[5, 0], [4, 0]], [0, 1], cfg)
    long = call([[0], [0]], [1., 1.], [[8, 0], [4, 0]], [0, 1], cfg)
    assert short['diagnostics']['n_breaks'] == 0
    assert long['diagnostics']['n_breaks'] == 1
    # At the junction, a sufficiently strong U-turn penalty changes the choice.
    cfg['u_turn_penalty'] = 10
    out = call([[0], [1, 2]], [1., .4, .6], [[9, 0], [10, 0]], [0, 1], cfg)
    assert out['arc_per_point'].tolist() == [0, 1]


def test_viterbi_against_brute_force(cfg):
    # All candidate pairs are connected and share a zero-distance GPS point.
    n = network(); xy = np.array([[10., 0.]] * 3); ts = np.arange(3)
    groups = [[1, 2]] * 3
    rng = np.random.default_rng(42)
    for _ in range(20):
        p = rng.uniform(.1, 1, size=(3, 2)); p /= p.sum(axis=1, keepdims=True)
        out = call(groups, p.ravel(), xy, ts, cfg)['arc_per_point']
        def score(seq):
            value = sum(np.log(p[t, j]) for t, j in enumerate(seq))
            for x, y in zip(seq, seq[1:]):
                # From arc 2, traversing arc 0 reaches arc 1; the reverse
                # direction has no path out of node 2.
                value += 0 if x == y else (-2 if x == 1 else cfg['unreachable_logp'])
            return value
        expected = max(itertools.product(range(2), repeat=3), key=score)
        assert out.tolist() == [groups[t][j] for t, j in enumerate(expected)]
