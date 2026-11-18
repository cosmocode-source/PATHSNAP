"""Real-model API parity and privacy checks on the existing student trips."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))
sys.path.insert(0, str(ROOT))
from app.config import settings
from app.main import app
from app.model_loader import FINAL_HMM_MODULE, load_hmm_config


def wait_for(client, trip_id, refresh):
    response = client.post('/api/predict', json={'trip_id': trip_id, 'refresh': refresh})
    assert response.status_code == 202, response.text
    job_id = response.json()['job_id']
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        job = client.get(f'/api/jobs/{job_id}').json()
        if job['status'] == 'done':
            return job['result']
        if job['status'] == 'failed':
            pytest.fail(f"{trip_id}: {job['error']}")
        time.sleep(.2)
    pytest.fail(f'{trip_id}: job did not finish')


def assert_no_private_keys(value):
    if isinstance(value, dict):
        forbidden = {'reference_route', 'ref_route', 'reference_labels', 'ref_labels',
                     'label', 'label_arc_id', 'label_quality', 'label_quality_share',
                     'per_point_correct', 'is_correct', 'correct_array'}
        assert not forbidden.intersection(value), forbidden.intersection(value)
        for child in value.values():
            assert_no_private_keys(child)
    elif isinstance(value, list):
        for child in value:
            assert_no_private_keys(child)


def test_invalid_config_fails_clearly(tmp_path):
    with pytest.raises(ValueError, match='HMM_CONFIG'):
        load_hmm_config(tmp_path / 'missing.json')


def test_all_student_trips_match_recorded_report(tmp_path):
    """Compare each API result with saved per-trip report inputs, not rounded means."""
    settings.MODEL_DIR = ROOT / 'trained_models/final_model'
    settings.HMM_MODULE = FINAL_HMM_MODULE
    settings.HMM_CONFIG = ROOT / 'trained_models/hmm/config.json'
    settings.STRICT_HMM = True
    settings.TRIPS_DIR = ROOT / 'Dataset/student_test'
    table = pd.read_csv(ROOT / 'trained_models/hmm/results/per_trip_metrics.csv', dtype={'trip_id': str})
    expected = table[table.partition == 'student'].set_index(['trip_id', 'method'])
    comparisons = {'top1_all': 'top1_all', 'top1_in_cands': 'top1_in_candidates',
                   'top1_clean': 'top1_clean', 'route_continuity': 'continuity',
                   'sequence_arc_f1': 'arc_f1_vs_reference'}
    observed = []
    with TestClient(app) as client:
        health = client.get('/api/health').json()
        assert health['hmm_available'] and health['hmm_config_name'] == 'config.json'
        assert health['model_sha256'] and health['hmm_config_sha256']
        trips = client.get('/api/trips').json()
        assert len(trips) == 10
        for item in trips:
            trip_id = item['trip_id']
            result = wait_for(client, trip_id, True)
            assert result['hmm_used'] is True
            assert_no_private_keys(result)
            for api_name, method in [('classifier', 'classifier'), ('hmm', 'hmm')]:
                for api_metric, report_metric in comparisons.items():
                    actual = result['metrics'][api_name][api_metric]
                    reference = float(expected.loc[(trip_id, method), report_metric])
                    assert actual == pytest.approx(reference, abs=0.000051), (trip_id, api_name, api_metric, actual, reference)
            # The complete reference route sequence must never be present in JSON.
            route = (settings.TRIPS_DIR / trip_id / f'{trip_id}.route').read_text().split()
            assert json.dumps([int(a) for a in route]) not in json.dumps(result)
            observed.append({'trip_id':trip_id, 'runtime_s':result['total_runtime_s'],
                'timings_s':result['timings_s'], 'metrics':result['metrics'],
                'hmm_fallbacks':result['error_summary']['hmm_fallbacks'], 'warnings':result['warnings']})
        # Compare exact predictions on a cache hit and on a fresh repeat.
        trip_id = trips[0]['trip_id']
        cached = wait_for(client, trip_id, False)
        repeated = wait_for(client, trip_id, True)
        assert cached['from_cache'] is True and repeated['from_cache'] is False
        assert cached['per_point_comparison'] == repeated['per_point_comparison']
        assert cached['metrics'] == repeated['metrics']
        log_text = json.dumps(client.get('/api/logs').json())
        for forbidden in ('reference_route', 'reference_labels', 'per_point_correct', 'label_arc_id'):
            assert forbidden not in log_text
    target = os.environ.get('PATHSNAP_VERIFY_DIR')
    if target:
        out = Path(target)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'student_parity.json').write_text(json.dumps(observed, indent=2), encoding='utf-8')
