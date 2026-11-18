"""Assertion tests for the PathSnap preprocessing pipeline (section 4.8).

Run with ``pytest data_preprocessing/data_preprocessing/test_pipeline.py`` or call ``run_all()``
(the preprocessing notebook does this). Needs the tables written by the notebook in
``data_preprocessing/cleaned_data/model_data/`` and the raw ``Dataset/`` folder.

Index spaces used below: gps_row (track row), node_id / arc_id (LOCAL to a trip), route_pos.
"""
from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pathsnap_prep as P  # noqa: E402

ROOT = P.find_repo_root(Path(__file__).resolve().parent)
DATA = ROOT / "Dataset"
MODEL = ROOT / "data_preprocessing" / "cleaned_data" / "model_data"
CACHE = Path(__file__).resolve().parent / "cache"
INFERENCE_FUNCS = [P.trip_from_arrays, P.point_flags, P.generate_candidates, P.query_radius, P.nearest_k_arcs,
                   P.pair_geometry, P.point_headings, P.build_for_trip, P.build_candidates_and_features]


def _student_trip() -> str:
    return P.list_trips(DATA, "student_test")[0]


def _raw(split: str, tid: str):
    """Read ONLY track, nodes and arcs (never the route) as plain arrays."""
    rd = lambda ext: pd.read_csv(DATA / split / tid / f"{tid}.{ext}", sep=r"\s+", header=None).to_numpy()
    return rd("track"), rd("nodes"), rd("arcs")


def test_inference_equals_batch():
    """The API entry point reproduces exactly the batch features stored for a student trip."""
    tid = _student_trip()
    feats, _, _ = P.build_candidates_and_features(*_raw("student_test", tid), config=P.CONFIG, trip_id=tid)
    stored = pd.read_parquet(MODEL / "student_candidates.parquet")
    stored = stored[stored.trip_id == tid].reset_index(drop=True)
    assert len(feats) == len(stored), "row count differs between inference and batch"
    assert (feats.gps_row.to_numpy() == stored.gps_row.to_numpy()).all() and (feats.arc_id.to_numpy() == stored.arc_id.to_numpy()).all()
    # Integer / flag features and the candidate rank must match EXACTLY on every machine (ties are broken by arc_id after
    # rounding distances to 1 mm). Continuous features may differ by float noise (different CPU / math library builds).
    exact = ["rank", "n_candidates", "is_nearest", "heading_valid", "is_reverse_twin", "after_gap", "stationary", "in_hive",
             "arc_in_deg", "arc_out_deg", "is_two_way"]
    for c in P.FEATURES:
        if c in exact:
            assert (feats[c].to_numpy() == stored[c].to_numpy()).all(), f"feature {c} differs"
        else:
            assert np.allclose(feats[c].to_numpy(float), stored[c].to_numpy(float), rtol=1e-4, atol=1e-3), f"feature {c} differs"


def test_no_route_in_inference_path():
    """No function on the prediction path takes a route argument or calls load_route / label_trip / align_to_route."""
    banned_calls = {"load_route", "label_trip", "align_to_route", "viterbi_align"}
    for fn in INFERENCE_FUNCS:
        tree = ast.parse(inspect.getsource(fn).lstrip())
        args = {a.arg for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) for a in n.args.args}
        assert "route" not in args, f"{fn.__name__} has a route argument"
        called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        called |= {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        assert not (called & banned_calls), f"{fn.__name__} calls {called & banned_calls}"


def test_runs_without_route_files():
    """The pipeline runs when only track / nodes / arcs exist (the route is never opened)."""
    tid = _student_trip()
    track, nodes, arcs = _raw("student_test", tid)
    feats, pts, trip = P.build_candidates_and_features(track, nodes, arcs, trip_id=tid)
    assert len(pts) == len(track) and len(feats) > 0


def test_index_spaces_and_projection_round_trip():
    """Index-space bounds hold and UTM -> lon/lat reproduces the raw GPS to < 1e-6 degrees."""
    for split, tid in (("train", P.list_trips(DATA, "train")[0]), ("student_test", _student_trip())):
        trip = P.load_trip(DATA, split, tid, CACHE)
        P.assert_index_spaces(trip)
        assert trip.arcs.max() < len(trip.node_xy)
        lon, lat = Transformer.from_crs(trip.epsg, 4326, always_xy=True).transform(trip.gps_xy[:, 0], trip.gps_xy[:, 1])
        assert np.abs(lon - trip.gps_lonlat[:, 0]).max() < 1e-6 and np.abs(lat - trip.gps_lonlat[:, 1]).max() < 1e-6
        route = P.load_route(DATA, split, tid, trip.n_arcs)
        assert route.max() < trip.n_arcs                                   # route_pos -> local arc_id stays in range


def test_arc_geometry_lookup():
    """Arc geometry from the loader matches the node coordinates of its two end nodes."""
    trip = P.load_trip(DATA, "student_test", _student_trip(), CACHE)
    a = np.random.default_rng(0).integers(0, trip.n_arcs, 50)
    assert np.allclose(trip.arc_a[a], trip.node_xy[trip.arcs[a, 0]]) and np.allclose(trip.arc_b[a], trip.node_xy[trip.arcs[a, 1]])
    assert np.allclose(trip.arc_len[a], np.hypot(*(trip.arc_b[a] - trip.arc_a[a]).T))


def test_every_point_has_a_candidate_and_student_rows_kept():
    """Every GPS point (train and student) has >= 1 candidate, and no student row was dropped."""
    for split, name in (("train", "points_train"), ("student_test", "points_student")):
        pts = pd.read_parquet(MODEL / f"{name}.parquet")
        assert (pts.n_candidates >= 1).all(), f"{split}: a point has no candidate"
        n_raw = sum(len(pd.read_csv(DATA / split / t / f"{t}.track", sep=r"\s+", header=None)) for t in P.list_trips(DATA, split))
        if split == "student_test":
            assert len(pts) == n_raw, f"student rows changed: {len(pts)} vs raw {n_raw}"
            assert (pts.sample_weight == 1.0).all() and pts.fit_include.all(), "student rows must have weight 1 and never be excluded"
        else:
            assert len(pts) == n_raw - int(pts.removable.sum()) or len(pts) == n_raw


def test_exclusion_rule_is_train_only_and_logged():
    """'Label not in candidates' exclusion applies to TRAIN only, and its count equals the logged number."""
    tr = pd.read_parquet(MODEL / "points_train.parquet"); st = pd.read_parquet(MODEL / "points_student.parquet")
    excluded = int((~tr.fit_include).sum())
    assert excluded == int((~tr.label_in_candidates).sum())
    assert st.fit_include.all(), "student rows must never be excluded"
    log = pd.read_csv(ROOT / "data_preprocessing" / "cleaned_data" / "data_quality_log.csv")
    logged = log[log["item"] == "fit_excluded_points_train"].value.iloc[-1]
    assert int(logged) == excluded, f"logged {logged} != computed {excluded}"
    cands = pd.read_parquet(MODEL / "train_candidates.parquet", columns=["point_id", "label", "fit_include"])
    assert (cands.groupby("point_id").label.max().astype(bool) == cands.groupby("point_id").fit_include.first()).all()


def test_scaled_table_is_reproducible_from_raw():
    """scale_features(raw student table, scaler bundle) equals the stored scaled student table."""
    import joblib
    bundle = joblib.load(MODEL / "scaler.joblib")
    raw = pd.read_parquet(MODEL / "student_candidates.parquet")
    stored = pd.read_parquet(MODEL / "student_candidates_scaled.parquet")
    rebuilt = P.scale_features(raw, bundle)
    for c in rebuilt.columns:
        assert np.allclose(rebuilt[c].to_numpy(float), stored[c].to_numpy(float), atol=1e-5), f"scaled feature {c} differs"


TESTS = [test_inference_equals_batch, test_no_route_in_inference_path, test_runs_without_route_files,
         test_index_spaces_and_projection_round_trip, test_arc_geometry_lookup,
         test_every_point_has_a_candidate_and_student_rows_kept, test_exclusion_rule_is_train_only_and_logged, test_scaled_table_is_reproducible_from_raw]


def run_all(verbose: bool = True) -> dict:
    """Run every test; returns {test_name: passed}. Raises nothing so the caller can report all results."""
    out = {}
    for t in TESTS:
        try:
            t(); out[t.__name__] = True
        except Exception as exc:  # noqa: BLE001 - report every failure
            out[t.__name__] = False
            if verbose: print("FAIL", t.__name__, "->", repr(exc))
        if verbose and out[t.__name__]: print("PASS", t.__name__)
    return out


if __name__ == "__main__":
    r = run_all()
    sys.exit(0 if all(r.values()) else 1)
