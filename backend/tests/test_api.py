"""Automated test suite for PathSnap Backend API (Section 3.10).

Verifies:
- /api/health and /api/trips response shapes.
- End-to-end predict + poll on student trips, including largest.
- Privacy guarantee: No route, labels, or per-point correctness in result JSON.
- Pipeline function signatures contain no route parameter.
- HMM changed indices calculation matches predictions difference.
- Error codes and safe error messages without file paths or tracebacks.
- Parity of route_continuity calculation against trained_models/common/metrics.py.
"""
from __future__ import annotations

import inspect
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(BACKEND_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR / "tests"))

from app.config import settings
from app.errors import ErrorCode
from app.main import app, app_state
from app.pipeline import run_pipeline
from stub_hmm import stub_decode_trip

@pytest.fixture(scope="module")
def client():
    """Module-scoped test client that runs the FastAPI lifespan."""
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    """Health endpoint returns expected keys and types."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model_name" in data
    assert "model_version" in data
    assert isinstance(data["hmm_available"], bool)
    assert data["n_trips"] == 10
    assert data["uptime_s"] >= 0.0


def test_trips_endpoint(client):
    """Trips endpoint returns 10 sorted trips with point and network counts, without route info."""
    resp = client.get("/api/trips")
    assert resp.status_code == 200
    trips = resp.json()
    assert len(trips) == 10
    trip_ids = [t["trip_id"] for t in trips]
    assert trip_ids == sorted(trip_ids)
    for t in trips:
        assert "trip_id" in t
        assert "n_gps_points" in t and t["n_gps_points"] > 0
        assert "n_nodes" in t and t["n_nodes"] > 0
        assert "n_arcs" in t and t["n_arcs"] > 0
        # Assert NO route information
        assert "route" not in t
        assert "route_pos" not in t
        assert "label" not in t


def test_prediction_signature_has_no_route_parameter():
    """Verify that run_pipeline and build_candidates_and_features have NO route parameter."""
    import pathsnap_prep

    pipeline_params = list(inspect.signature(run_pipeline).parameters.keys())
    assert "route" not in pipeline_params
    assert "ref_route" not in pipeline_params
    assert "labels" not in pipeline_params

    prep_params = list(inspect.signature(pathsnap_prep.build_candidates_and_features).parameters.keys())
    assert "route" not in prep_params
    assert "ref_route" not in prep_params
    assert "labels" not in prep_params


def test_predict_and_poll_end_to_end(client):
    """End-to-end predict and poll on a dev trip (00000052)."""
    trip_id = "00000052"
    # Initiate prediction
    post_resp = client.post("/api/predict", json={"trip_id": trip_id, "refresh": True})
    assert post_resp.status_code == 202
    job_id = post_resp.json()["job_id"]
    assert job_id

    # Poll until done
    max_wait_s = 30
    start = time.time()
    result = None
    while time.time() - start < max_wait_s:
        get_resp = client.get(f"/api/jobs/{job_id}")
        assert get_resp.status_code == 200
        job_data = get_resp.json()
        if job_data["status"] == "done":
            result = job_data["result"]
            break
        elif job_data["status"] == "failed":
            pytest.fail(f"Job failed: {job_data['error']}")
        time.sleep(0.4)

    assert result is not None, "Job did not complete within timeout."
    assert result["trip_id"] == trip_id
    assert "model" in result
    assert "routes" in result
    assert "classifier" in result["routes"]
    assert "metrics" in result
    assert "classifier" in result["metrics"]
    assert "error_summary" in result
    assert "per_point_comparison" in result
    assert "data_quality_context" in result
    assert "warnings" in result


def test_privacy_and_no_route_leakage(client):
    """Assert result JSON contains NO reference route, reference labels, or per-point correctness arrays."""
    trip_id = "00000052"
    resp = client.post("/api/predict", json={"trip_id": trip_id, "refresh": False})
    job_id = resp.json()["job_id"]

    for _ in range(30):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] == "done":
            result = job["result"]
            break
        time.sleep(0.3)

    # Read reference route file
    route_file = settings.TRIPS_DIR / trip_id / f"{trip_id}.route"
    ref_route_content = route_file.read_text(encoding="utf-8").strip().split()
    ref_arcs = [int(x) for x in ref_route_content]

    # Convert result to string and inspect
    import json
    result_str = json.dumps(result)

    # Forbidden key patterns
    forbidden_keys = [
        "reference_route",
        "reference_labels",
        "ref_route",
        "ref_labels",
        "per_point_correct",
        "is_correct",
        "correct_array",
        "label_array",
    ]
    for key in forbidden_keys:
        assert key not in result_str

    # Assert that reference route sequence as a whole is not leaked
    assert json.dumps(ref_arcs) not in result_str


def test_stub_hmm_integration_and_changed_indices(client):
    """Plug in stub HMM decoder to verify HMM code path, changed indices, and deltas."""
    # Temporarily plug in stub decoder
    app_state["hmm_decoder"] = stub_decode_trip

    try:
        trip_id = "00000052"
        resp = client.post("/api/predict", json={"trip_id": trip_id, "refresh": True})
        job_id = resp.json()["job_id"]

        result = None
        for _ in range(30):
            job = client.get(f"/api/jobs/{job_id}").json()
            if job["status"] == "done":
                result = job["result"]
                break
            time.sleep(0.3)

        assert result is not None
        assert result["hmm_used"] is True
        assert result["routes"]["hmm"] is not None
        assert result["metrics"]["hmm"] is not None
        assert result["metrics"]["delta"] is not None

        # Verify HMM changed indices equal exactly where HMM arcs differ from classifier arcs
        comparison = result["per_point_comparison"]
        clf_arcs = np.array(comparison["classifier_arc_ids"])
        hmm_arcs = np.array(comparison["hmm_arc_ids"])
        diff_indices = np.flatnonzero(clf_arcs != hmm_arcs).tolist()
        assert comparison["hmm_changed_indices"] == diff_indices

    finally:
        # Restore state
        app_state["hmm_decoder"] = None


def test_error_handling_and_no_leaks(client):
    """Test error codes and verify no filesystem paths or tracebacks in messages."""
    # 1. TRIP_NOT_FOUND (404)
    resp = client.post("/api/predict", json={"trip_id": "non_existent_trip_999"})
    assert resp.status_code == 404
    data = resp.json()
    assert data["error"]["code"] == ErrorCode.TRIP_NOT_FOUND
    assert "Traceback" not in data["error"]["message"]
    assert "\\" not in data["error"]["message"]

    # 2. INVALID_REQUEST (422) with path traversal
    resp = client.post("/api/predict", json={"trip_id": "../../etc/passwd"})
    assert resp.status_code == 422
    data = resp.json()
    assert data["error"]["code"] == ErrorCode.INVALID_REQUEST
    assert "Traceback" not in data["error"]["message"]
    assert ".." not in data["error"]["message"]

    # 3. Job not found (404)
    resp = client.get("/api/jobs/invalid_job_id_xyz")
    assert resp.status_code == 404
    data = resp.json()
    assert data["error"]["code"] == ErrorCode.INVALID_REQUEST


def test_route_continuity_parity():
    """Verify backend route continuity matches metrics.route_continuity on 2 student trips."""
    import sys
    sys.path.insert(0, str(BACKEND_DIR.parent / "trained_models"))
    from common import metrics
    from app.scoring import _compute_route_continuity

    for trip_id in ["00000052", "00000000"]:
        arcs_file = settings.TRIPS_DIR / trip_id / f"{trip_id}.arcs"
        arcs_uv = pd.read_csv(arcs_file, sep=r"\s+", header=None, names=["u", "v"], dtype=int).to_numpy()

        pred_file = BACKEND_DIR.parent / "trained_models" / "model4" / "student_predictions.parquet"
        pred_df = pd.read_parquet(pred_file)
        sub_df = pred_df[pred_df["trip_id"] == trip_id].copy()

        # metrics.route_continuity
        res_shared = metrics.route_continuity(
            sub_df[["trip_id", "point_id", "gps_row", "arc_id"]],
            sub_df["p_seed42"],
            split="student",
        )
        expected_cont = res_shared["per_trip"][trip_id]

        # Backend local continuity
        pid = metrics.point_ids(sub_df[["trip_id", "gps_row"]])
        top = metrics.top_rows(sub_df["p_seed42"].to_numpy(), pid)
        top_arcs = sub_df["arc_id"].to_numpy()[top]
        keep = np.r_[True, top_arcs[1:] != top_arcs[:-1]]
        seq = top_arcs[keep].tolist()

        local_cont, _ = _compute_route_continuity(seq, arcs_uv)
        assert abs(expected_cont - local_cont) < 1e-9


def test_largest_trip_stress(client):
    """Trip 00000076 (624k arcs / 532k nodes) runs end to end within timeout."""
    trip_id = "00000076"
    t0 = time.perf_counter()
    resp = client.post("/api/predict", json={"trip_id": trip_id, "refresh": True})
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    result = None
    for _ in range(60):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] == "done":
            result = job["result"]
            break
        elif job["status"] == "failed":
            pytest.fail(f"Largest trip failed: {job['error']}")
        time.sleep(1.0)

    elapsed = time.perf_counter() - t0
    assert result is not None
    assert elapsed < settings.TRIP_TIMEOUT_S
    print(f"\nLargest trip {trip_id} completed in {elapsed:.2f} s (backend runtime {result['total_runtime_s']:.2f} s).")
