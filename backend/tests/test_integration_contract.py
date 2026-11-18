"""Small tests for frozen configuration, scoring and safe fallback behavior."""
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))
sys.path.insert(0, str(ROOT))
from app.errors import AppError
from app.model_loader import load_hmm_decoder, load_model_bundle
from app.pipeline import run_pipeline
from app.scoring import _compute_sequence_metrics


def test_nonfinal_decoder_cannot_be_loaded_as_the_real_hmm():
    with pytest.raises(RuntimeError, match='final decoder'):
        load_hmm_decoder('tests.stub_hmm:stub_decode_trip')


def test_sequence_f1_respects_arc_order():
    precision, recall, f1, _ = _compute_sequence_metrics([0, 1], [1, 0], np.array([1., 1.]))
    assert (precision, recall, f1) == (.5, .5, .5)


def test_failure_is_visible_or_falls_back_without_hmm_label():
    model = load_model_bundle(ROOT / 'trained_models/final_model')
    def broken(**kwargs):
        raise RuntimeError('forced HMM failure')
    args = dict(trip_id='00000052', trips_dir=ROOT/'Dataset/student_test', model_bundle=model, hmm_decoder=broken)
    fallback = run_pipeline(**args, strict_hmm=False)
    assert fallback.hmm_used is False and fallback.hmm_route is None
    assert any('fell back' in w for w in fallback.warnings)
    with pytest.raises(AppError, match='HMM decoder failed'):
        run_pipeline(**args, strict_hmm=True)


def test_decoder_receives_normalised_scores_and_explicit_config():
    model = load_model_bundle(ROOT / 'trained_models/final_model')
    settings = {'test_marker': 42}
    def check(candidates, probs, network, gps_xy, times, config):
        assert config is settings
        assert 'label' not in candidates and 'route' not in candidates
        assert np.allclose(np.bincount(candidates.gps_row, weights=probs), 1.)
        # Return the classifier decision only to exercise the call contract.
        order = np.lexsort((np.arange(len(probs)), -probs, candidates.gps_row))
        first = np.r_[True, candidates.gps_row.to_numpy()[order][1:] != candidates.gps_row.to_numpy()[order][:-1]]
        arcs = candidates.arc_id.to_numpy()[order[first]]
        return {'arc_per_point': arcs, 'diagnostics': {}}
    result = run_pipeline('00000052', ROOT/'Dataset/student_test', model, check, strict_hmm=True, hmm_config=settings)
    assert result.hmm_used and np.array_equal(result.classifier_arcs, result.hmm_arcs)
