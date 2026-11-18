"""Test the backend dictionary contract and frozen-setting precedence."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from trained_models.hmm.api_adapter import decode_trip as api_decode
from trained_models.hmm.hmm_decoder import decode_trip


def test_backend_network_and_frozen_config():
    node_xy = np.array([[0., 0.], [10., 0.], [20., 0.]])
    arcs = np.array([[0, 1], [1, 2]])
    network = {'node_xy': node_xy, 'arcs': arcs, 'arc_a': node_xy[arcs[:, 0]],
               'arc_b': node_xy[arcs[:, 1]], 'arc_len': np.array([10., 10.])}
    candidates = pd.DataFrame({'gps_row':[0,0,1,1], 'arc_id':[0,1,0,1]})
    probs = np.array([.8,.2,.3,.7])
    xy = np.array([[2.,0.],[12.,0.]])
    times = np.array([0., 1.])
    frozen = json.loads((Path(__file__).parents[1] / 'config.json').read_text())
    expected = decode_trip(candidates, probs, SimpleNamespace(**network), xy, times, frozen)
    actual = api_decode(candidates, probs, network, xy, times, {'alpha':999})
    assert np.array_equal(expected['arc_per_point'], actual['arc_per_point'])
    assert expected['diagnostics']['n_breaks'] == actual['diagnostics']['n_breaks']
