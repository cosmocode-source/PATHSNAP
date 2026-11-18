"""PathSnap Server-Side Scoring (Section 3.6).

This is the ONLY module in the backend that reads the .route file and reference labels.
Never exposes the reference route, per-point reference labels, or per-point correctness flags
in any response, log line, or returned object. Only aggregate numbers are returned.
"""
from __future__ import annotations

import collections
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PREP_DIR = REPO_ROOT / "data_preprocessing" / "data_preprocessing"
if str(PREP_DIR) not in sys.path:
    sys.path.insert(0, str(PREP_DIR))

import pathsnap_prep  # noqa: E402
from pathsnap_prep import CONFIG as DEFAULT_PREP_CONFIG  # noqa: E402
from trained_models.common.metrics import lcs_length  # noqa: E402

from .errors import AppError, ErrorCode
from .pipeline import PipelinePrediction


def _compute_sequence_metrics(
    pred_seq: List[int],
    ref_seq: List[int],
    arc_len: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Compute arc precision, recall, F1, and length-weighted route mismatch fraction."""
    c_pred = collections.Counter(pred_seq)
    c_ref = collections.Counter(ref_seq)
    # The final comparison uses the ordered longest common subsequence, not
    # unordered multiset overlap, for precision, recall and F1.
    overlap = lcs_length(np.asarray(pred_seq, dtype=np.int64), np.asarray(ref_seq, dtype=np.int64))
    n_p = max(len(pred_seq), 1)
    n_r = max(len(ref_seq), 1)

    precision = float(overlap / n_p)
    recall = float(overlap / n_r)
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    all_arcs = set(c_pred.keys()).union(c_ref.keys())
    mismatch_m = sum(abs(c_pred[a] - c_ref[a]) * float(arc_len[a]) for a in all_arcs)
    ref_m = sum(c_ref[a] * float(arc_len[a]) for a in c_ref)
    mismatch_frac = float(mismatch_m / max(ref_m, 1e-6))

    return precision, recall, f1, mismatch_frac


def _compute_route_continuity(collapsed_arcs: List[int], arcs_uv: np.ndarray) -> Tuple[float, int]:
    """Compute continuity (fraction of connected transitions) and count of broken transitions."""
    if len(collapsed_arcs) < 2:
        return 1.0, 0
    seq = np.asarray(collapsed_arcs, dtype=np.int64)
    # Consecutive arc connection: to_node(a) == from_node(b)
    connected = arcs_uv[seq[:-1], 1] == arcs_uv[seq[1:], 0]
    n_broken = int(np.sum(~connected))
    continuity = float(np.mean(connected))
    return continuity, n_broken


def score_prediction(
    prediction: PipelinePrediction,
    trips_dir: Path,
    prep_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Server-side scoring against the reference route.

    Loads .route only now, after predictions are frozen.
    Returns aggregate metrics and error summary only.
    """
    t0 = time.perf_counter()
    trip_id = prediction.trip_id
    trip = prediction.trip
    cfg = prep_config or DEFAULT_PREP_CONFIG

    # Load reference route file
    route_file = trips_dir / trip_id / f"{trip_id}.route"
    if not route_file.exists():
        raise AppError(ErrorCode.PIPELINE_FAILED, f"Reference route file missing for trip {trip_id}.", stage="scoring", trip_id=trip_id)

    try:
        ref_route = pd.read_csv(route_file, sep=r"\s+", header=None, names=["arc_id"], dtype=int)["arc_id"].to_numpy()
    except Exception as e:
        raise AppError(ErrorCode.PIPELINE_FAILED, f"Failed to load reference route: {e}", stage="scoring", trip_id=trip_id)

    # Compute reference labels via route-constrained Viterbi alignment
    try:
        ref_labels_df = pathsnap_prep.label_trip(trip, ref_route, config=cfg)
    except Exception as e:
        raise AppError(ErrorCode.PIPELINE_FAILED, f"Server-side labelling failed: {e}", stage="scoring", trip_id=trip_id)

    ref_arcs = ref_labels_df["arc_id"].to_numpy()
    ref_quality = ref_labels_df["label_quality"].to_numpy()
    n_pts = len(ref_arcs)

    # Determine which points had their reference arc in their candidate set
    feats = prediction.feats
    # Candidates per point: create set lookup
    cand_by_point = feats.groupby("gps_row")["arc_id"].apply(set).to_dict()
    in_candidates = np.array([ref_arcs[i] in cand_by_point.get(i, set()) for i in range(n_pts)], dtype=bool)
    miss_rate = float(1.0 - np.mean(in_candidates))

    # Point-level quality masks
    clean_mask = ref_quality == "clean"
    ambig_mask = ref_quality == "ambiguous"
    far_mask = ref_quality == "far"

    label_quality_share = {
        "clean": round(float(np.mean(clean_mask)), 4),
        "ambiguous": round(float(np.mean(ambig_mask)), 4),
        "far": round(float(np.mean(far_mask)), 4),
    }

    # Add warning about far labels if justified
    if label_quality_share["far"] >= 0.05:
        pct = round(label_quality_share["far"] * 100, 1)
        prediction.warnings.append(
            f"{pct}% of observations have poor reference alignment; accuracy may understate prediction quality."
        )

    arcs_uv = prediction.arcs_df[["u", "v"]].to_numpy()
    arc_len = trip.arc_len
    ref_route_list = ref_route.tolist()

    # --- Classifier metrics ---
    clf_arcs = prediction.classifier_arcs
    clf_hit = clf_arcs == ref_arcs
    clf_top1_all = float(np.mean(clf_hit))
    clf_top1_in_cands = float(np.mean(clf_hit[in_candidates])) if in_candidates.any() else float("nan")
    clf_top1_clean = float(np.mean(clf_hit[clean_mask])) if clean_mask.any() else float("nan")

    clf_continuity, clf_broken = _compute_route_continuity(prediction.classifier_route.arc_ids, arcs_uv)
    clf_prec, clf_rec, clf_f1, clf_mismatch = _compute_sequence_metrics(
        prediction.classifier_route.arc_ids, ref_route_list, arc_len
    )

    clf_metrics = {
        "top1_all": round(clf_top1_all, 4),
        "top1_in_cands": round(clf_top1_in_cands, 4),
        "top1_clean": round(clf_top1_clean, 4),
        "candidate_miss_rate": round(miss_rate, 4),
        "route_continuity": round(clf_continuity, 4),
        "sequence_arc_precision": round(clf_prec, 4),
        "sequence_arc_recall": round(clf_rec, 4),
        "sequence_arc_f1": round(clf_f1, 4),
        "sequence_length_weighted_mismatch": round(clf_mismatch, 4),
    }

    # --- HMM metrics (if available) ---
    hmm_metrics: Optional[Dict[str, float]] = None
    delta_metrics: Optional[Dict[str, float]] = None
    hmm_n_correct: Optional[int] = None
    hmm_n_incorrect: Optional[int] = None
    hmm_broken: Optional[int] = None

    if prediction.hmm_used and prediction.hmm_arcs is not None and prediction.hmm_route is not None:
        h_arcs = prediction.hmm_arcs
        h_hit = h_arcs == ref_arcs
        h_top1_all = float(np.mean(h_hit))
        h_top1_in_cands = float(np.mean(h_hit[in_candidates])) if in_candidates.any() else float("nan")
        h_top1_clean = float(np.mean(h_hit[clean_mask])) if clean_mask.any() else float("nan")

        h_continuity, hmm_broken = _compute_route_continuity(prediction.hmm_route.arc_ids, arcs_uv)
        h_prec, h_rec, h_f1, h_mismatch = _compute_sequence_metrics(
            prediction.hmm_route.arc_ids, ref_route_list, arc_len
        )

        hmm_metrics = {
            "top1_all": round(h_top1_all, 4),
            "top1_in_cands": round(h_top1_in_cands, 4),
            "top1_clean": round(h_top1_clean, 4),
            "candidate_miss_rate": round(miss_rate, 4),
            "route_continuity": round(h_continuity, 4),
            "sequence_arc_precision": round(h_prec, 4),
            "sequence_arc_recall": round(h_rec, 4),
            "sequence_arc_f1": round(h_f1, 4),
            "sequence_length_weighted_mismatch": round(h_mismatch, 4),
        }

        delta_metrics = {
            k: round(hmm_metrics[k] - clf_metrics[k], 4) for k in clf_metrics
        }
        hmm_n_correct = int(np.sum(h_hit))
        hmm_n_incorrect = int(np.sum(~h_hit))

    # Error summary (aggregates only)
    error_summary = {
        "n_points": n_pts,
        "n_correct": {
            "classifier": int(np.sum(clf_hit)),
            "hmm": hmm_n_correct,
        },
        "n_incorrect": {
            "classifier": int(np.sum(~clf_hit)),
            "hmm": hmm_n_incorrect,
        },
        "broken_transitions": {
            "classifier": clf_broken,
            "hmm": hmm_broken,
        },
        "hmm_fallbacks": prediction.hmm_fallbacks,
        "n_points_hmm_changed": len(prediction.per_point_comparison["hmm_changed_indices"]),
    }

    prediction.stage_timings["scoring"] = time.perf_counter() - t0

    return {
        "metrics": {
            "classifier": clf_metrics,
            "hmm": hmm_metrics,
            "delta": delta_metrics,
        },
        "error_summary": error_summary,
    }
