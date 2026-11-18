"""PathSnap Prediction Pipeline (Section 3.5).

Strictly in this order:
1. Load only .track, .nodes, .arcs for the trip. No .route.
2. build_candidates_and_features(track, nodes, arcs, config) (no route argument).
3. Model predict_proba; normalise within each GPS point's candidate set.
4. Classifier-only prediction: top candidate per point.
5. HMM/Viterbi decoding with the same normalised probabilities as emissions.
6. Collapse repeated arcs to get each predicted route.
7. Build map geometry for both predicted routes and downsample GPS for display if needed.
Predictions are frozen BEFORE scoring starts.
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure data_preprocessing is importable
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PREP_DIR = REPO_ROOT / "data_preprocessing" / "data_preprocessing"
if str(PREP_DIR) not in sys.path:
    sys.path.insert(0, str(PREP_DIR))

import pathsnap_prep  # noqa: E402
from pathsnap_prep import CONFIG as DEFAULT_PREP_CONFIG  # noqa: E402

from .errors import AppError, ErrorCode
from .model_loader import ModelBundle


@dataclass
class RouteData:
    arc_ids: List[int]
    n_arcs: int
    segments: List[List[List[float]]]


@dataclass
class PipelinePrediction:
    trip_id: str
    classifier_arcs: np.ndarray             # (N,) top arc id per GPS point
    hmm_arcs: Optional[np.ndarray]          # (N,) HMM arc id per GPS point (or None)
    classifier_route: RouteData
    hmm_route: Optional[RouteData]
    raw_gps: List[List[float]]              # [[lat, lon], ...]
    display_gps: List[List[float]]          # downsampled if > MAX_DISPLAY_POINTS
    display_downsample_step: int
    n_points: int
    hmm_used: bool
    hmm_fallbacks: Dict[str, int]
    per_point_comparison: Dict[str, Any]
    data_quality_context: Dict[str, Any]
    warnings: List[str]
    stage_timings: Dict[str, float]
    # Internal structures needed by scoring (no route information)
    trip: pathsnap_prep.Trip
    feats: pd.DataFrame
    points_summary: pd.DataFrame
    nodes_df: pd.DataFrame
    arcs_df: pd.DataFrame


def collapse_repeated_arcs(arc_seq: np.ndarray) -> List[int]:
    """Collapse consecutive repeated arcs in a sequence."""
    if len(arc_seq) == 0:
        return []
    keep = np.r_[True, arc_seq[1:] != arc_seq[:-1]]
    return arc_seq[keep].tolist()


def build_route_geometry(arc_seq: List[int], nodes_df: pd.DataFrame, arcs_df: pd.DataFrame) -> RouteData:
    """Build polyline segments in [lat, lon], splitting wherever consecutive arcs are disconnected."""
    if not arc_seq:
        return RouteData(arc_ids=[], n_arcs=0, segments=[])

    arcs_uv = arcs_df[["u", "v"]].to_numpy()
    lats = nodes_df["lat"].to_numpy()
    lons = nodes_df["lon"].to_numpy()

    segments: List[List[List[float]]] = []
    current_segment: List[List[float]] = []
    prev_to_node: Optional[int] = None

    for arc_id in arc_seq:
        u, v = arcs_uv[arc_id]
        pt_u = [float(lats[u]), float(lons[u])]
        pt_v = [float(lats[v]), float(lons[v])]

        if prev_to_node is None or u != prev_to_node:
            # Disconnected or first arc -> start new segment
            if current_segment:
                segments.append(current_segment)
            current_segment = [pt_u, pt_v]
        else:
            # Connected -> extend segment
            current_segment.append(pt_v)

        prev_to_node = v

    if current_segment:
        segments.append(current_segment)

    return RouteData(arc_ids=arc_seq, n_arcs=len(arc_seq), segments=segments)


def run_pipeline(
    trip_id: str,
    trips_dir: Path,
    model_bundle: ModelBundle,
    hmm_decoder: Optional[Callable],
    strict_hmm: bool = False,
    max_display_points: int = 1000,
    stage_callback: Optional[Callable[[str, float], None]] = None,
    prep_config: Optional[Dict[str, Any]] = None,
    hmm_config: Optional[Dict[str, Any]] = None,
) -> PipelinePrediction:
    """Run full route-free inference pipeline for one trip.

    Never loads or references .route.
    """
    # Preprocessing / HMM default configs
    hmm_defaults = {
        "alpha": 1.0,
        "eps": 1e-12,
        "beta": 30.0,
        "u_turn_penalty": 2.0,
        "gap_beta_factor": 3.0,
        "gap_seconds": 5.0,
        "v_max_mps": 50.0,
        "min_cutoff_m": 100.0,
        "hmm_top_k": 15,
        "backward_tolerance_m": 5.0,
        "unreachable_logp": -1000000.0,
        "cutoff_bucket_m": 25.0,
        "cache_size": 512,
    }
    cfg = {**hmm_defaults, **(prep_config or DEFAULT_PREP_CONFIG)}
    timings: Dict[str, float] = {}
    warnings: List[str] = []

    def _record_stage(stage_name: str, duration: float):
        timings[stage_name] = duration
        if stage_callback:
            stage_callback(stage_name, duration)

    # -------------------------------------------------------------------------
    # Stage 1: loading_trip
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    trip_dir = trips_dir / trip_id
    if not trip_dir.is_dir():
        raise AppError(ErrorCode.TRIP_NOT_FOUND, f"Trip {trip_id} not found in trips directory.", stage="loading_trip", trip_id=trip_id)

    track_file = trip_dir / f"{trip_id}.track"
    nodes_file = trip_dir / f"{trip_id}.nodes"
    arcs_file = trip_dir / f"{trip_id}.arcs"

    if not (track_file.exists() and nodes_file.exists() and arcs_file.exists()):
        raise AppError(ErrorCode.TRIP_NOT_FOUND, f"Trip {trip_id} is missing required data files.", stage="loading_trip", trip_id=trip_id)

    try:
        track_df = pd.read_csv(track_file, sep=r"\s+", header=None, names=["lon", "lat", "t"], dtype=float)
        nodes_df = pd.read_csv(nodes_file, sep=r"\s+", header=None, names=["lon", "lat"], dtype=float)
        arcs_df = pd.read_csv(arcs_file, sep=r"\s+", header=None, names=["u", "v"], dtype=int)
    except Exception as e:
        raise AppError(ErrorCode.PIPELINE_FAILED, f"Failed to load raw trip files: {e}", stage="loading_trip", trip_id=trip_id)

    _record_stage("loading_trip", time.perf_counter() - t0)

    # -------------------------------------------------------------------------
    # Stage 2: candidates & features (build_candidates_and_features)
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    try:
        feats, points_summary, trip = pathsnap_prep.build_candidates_and_features(
            track_df, nodes_df, arcs_df, config=cfg, trip_id=trip_id
        )
    except Exception as e:
        raise AppError(ErrorCode.PIPELINE_FAILED, f"Candidate generation failed: {e}", stage="candidates", trip_id=trip_id)
    _record_stage("candidates", time.perf_counter() - t0)

    # -------------------------------------------------------------------------
    # Stage 3: features extraction for classifier
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    feature_cols = model_bundle.features
    missing_cols = [c for c in feature_cols if c not in feats.columns]
    if missing_cols:
        raise AppError(
            ErrorCode.PIPELINE_FAILED,
            f"Candidate table missing required model features: {missing_cols}",
            stage="features",
            trip_id=trip_id,
        )
    X = feats[feature_cols].to_numpy(dtype=np.float32)
    _record_stage("features", time.perf_counter() - t0)

    # -------------------------------------------------------------------------
    # Stage 4: classifier (prediction & probability normalization)
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    try:
        raw_probs = model_bundle.booster.predict(X, num_threads=2)
    except Exception as e:
        raise AppError(ErrorCode.PIPELINE_FAILED, f"Model prediction failed: {e}", stage="classifier", trip_id=trip_id)

    # Group candidates by GPS point
    # feats['gps_row'] corresponds to the 0-based point index
    gps_rows = feats["gps_row"].to_numpy(dtype=np.int64)
    n_points = trip.n_gps

    # Normalise probabilities within each point's candidate set
    sum_prob_per_point = np.bincount(gps_rows, weights=raw_probs, minlength=n_points)
    norm_probs = raw_probs / np.maximum(sum_prob_per_point[gps_rows], 1e-12)

    # Top-1 candidate per point (ties -> earliest row / nearer candidate)
    # lexsort by gps_row ascending, then raw_probs descending, then row index ascending
    row_idx = np.arange(len(raw_probs))
    order = np.lexsort((row_idx, -raw_probs, gps_rows))
    first_in_point = np.r_[True, gps_rows[order][1:] != gps_rows[order][:-1]]
    top_indices = order[first_in_point]

    classifier_top_arcs = feats["arc_id"].to_numpy()[top_indices]
    _record_stage("classifier", time.perf_counter() - t0)

    # -------------------------------------------------------------------------
    # Stage 5: HMM decoding
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    hmm_arcs: Optional[np.ndarray] = None
    hmm_used = False
    hmm_fallbacks = {"unreachable_candidates": 0, "long_gaps": 0, "breaks": 0,
                     "unreachable_pairs": 0, "gap_relaxed_steps": 0, "missing_candidates": 0}

    if hmm_decoder is not None:
        try:
            # Network dict for HMM
            network = {
                "arcs": arcs_df[["u", "v"]].to_numpy(),
                "node_xy": trip.node_xy,
                "arc_a": trip.arc_a,
                "arc_b": trip.arc_b,
                "arc_len": trip.arc_len,
            }
            hmm_res = hmm_decoder(
                candidates=feats[["gps_row", "arc_id", "dist_m", "rank"]],
                probs=norm_probs,
                network=network,
                gps_xy=trip.gps_xy,
                times=trip.gps_t,
                config=hmm_config if hmm_config is not None else cfg,
            )
            hmm_arcs = np.asarray(hmm_res["arc_per_point"], dtype=np.int64)
            if hmm_arcs.shape != (n_points,) or np.any(hmm_arcs < 0) or np.any(hmm_arcs >= len(trip.arcs)):
                raise ValueError("HMM returned an invalid arc sequence")
            diag = hmm_res.get("diagnostics", {})
            n_unreach = int(diag.get("fallbacks_unreachable", diag.get("n_breaks", 0)))
            n_gaps = int(diag.get("fallbacks_gap", diag.get("n_gap_relaxed_steps", 0)))
            hmm_fallbacks = {
                "unreachable_candidates": n_unreach,
                "long_gaps": n_gaps,
                "breaks": int(diag.get("n_breaks", n_unreach)),
                "unreachable_pairs": int(diag.get("n_unreachable_pairs", 0)),
                "gap_relaxed_steps": n_gaps,
                "missing_candidates": int(diag.get("n_missing_candidates", 0)),
            }
            hmm_used = True
            if hmm_fallbacks["breaks"] > 0 or hmm_fallbacks["missing_candidates"] > 0:
                warnings.append(
                    f"HMM restarted {hmm_fallbacks['breaks']} times; "
                    f"{hmm_fallbacks['missing_candidates']} points lacked candidates."
                )
        except Exception as e:
            if strict_hmm:
                raise AppError(ErrorCode.PIPELINE_FAILED, f"HMM decoder failed: {e}", stage="hmm", trip_id=trip_id)
            else:
                warnings.append("HMM decoding failed; fell back to classifier-only result.")
                hmm_arcs = None
                hmm_used = False
    else:
        warnings.append("HMM decoder not configured (classifier-only mode).")

    _record_stage("hmm", time.perf_counter() - t0)

    # -------------------------------------------------------------------------
    # Stage 6 & 7: collapse repeats & building_geometry
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    classifier_collapsed = collapse_repeated_arcs(classifier_top_arcs)
    classifier_route_data = build_route_geometry(classifier_collapsed, nodes_df, arcs_df)

    hmm_route_data: Optional[RouteData] = None
    if hmm_used and hmm_arcs is not None:
        hmm_collapsed = collapse_repeated_arcs(hmm_arcs)
        hmm_route_data = build_route_geometry(hmm_collapsed, nodes_df, arcs_df)

    # Build raw GPS points [lat, lon]
    raw_gps = np.column_stack([track_df["lat"].to_numpy(), track_df["lon"].to_numpy()]).tolist()
    if len(raw_gps) > max_display_points:
        downsample_step = math.ceil(len(raw_gps) / max_display_points)
        display_gps = raw_gps[::downsample_step]
    else:
        downsample_step = 1
        display_gps = raw_gps

    # Point flags for comparison & data quality
    fl = pathsnap_prep.point_flags(trip, cfg)
    after_gap_idx = np.flatnonzero(fl["after_gap"].to_numpy()).tolist()
    stationary_idx = np.flatnonzero(fl["stationary"].to_numpy()).tolist()

    if hmm_used and hmm_arcs is not None:
        hmm_changed_idx = np.flatnonzero(hmm_arcs != classifier_top_arcs).tolist()
    else:
        hmm_changed_idx = []

    per_point_comparison = {
        "hmm_changed_indices": hmm_changed_idx,
        "after_gap_indices": after_gap_idx,
        "stationary_indices": stationary_idx,
        "classifier_arc_ids": classifier_top_arcs.tolist(),
        "hmm_arc_ids": hmm_arcs.tolist() if (hmm_used and hmm_arcs is not None) else None,
    }

    # Data quality summary
    n_cands_series = points_summary["n_candidates"]
    data_quality_context = {
        "n_candidates_mean": round(float(n_cands_series.mean()), 2),
        "n_candidates_max": int(n_cands_series.max()),
        "share_after_long_gap": round(float(fl["after_gap"].mean()), 4),
        "share_stationary": round(float(fl["stationary"].mean()), 4),
    }

    _record_stage("building_geometry", time.perf_counter() - t0)

    return PipelinePrediction(
        trip_id=trip_id,
        classifier_arcs=classifier_top_arcs,
        hmm_arcs=hmm_arcs,
        classifier_route=classifier_route_data,
        hmm_route=hmm_route_data,
        raw_gps=raw_gps,
        display_gps=display_gps,
        display_downsample_step=downsample_step,
        n_points=n_points,
        hmm_used=hmm_used,
        hmm_fallbacks=hmm_fallbacks,
        per_point_comparison=per_point_comparison,
        data_quality_context=data_quality_context,
        warnings=warnings,
        stage_timings=timings,
        trip=trip,
        feats=feats,
        points_summary=points_summary,
        nodes_df=nodes_df,
        arcs_df=arcs_df,
    )
