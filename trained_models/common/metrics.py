"""Evaluation metrics shared by all five PathSnap models (identical code = identical evaluation).

Every model outputs one score per (GPS point, candidate arc) row: the predicted probability that this candidate is the
point's true road arc. All metrics below are computed from those scores, the 0/1 label of each row and the point keys.
Sample weights are used for FITTING only; every metric here is UNWEIGHTED (each GPS point / row counts once).

Definitions
-----------
* Top-1 accuracy (observation level, per point). For each point (trip_id, point_id) take the candidate with the highest
  score (ties -> the nearer candidate, i.e. the earlier stored row) and check whether it is the aligned label arc.
    - top1_all      : over ALL points. A point whose label arc is not among its candidates can never be right -> a miss.
    - top1_in_cands : only over points whose label arc IS among the candidates (the model's own ranking quality).
* Macro F1 (OUR OWN definition; the guidelines do not define it). Candidate level, two classes: the top-scoring
  candidate of each point is predicted 1, all other candidates 0. Macro F1 = mean of F1(class 1) and F1(class 0).
* Log loss, two variants:
    - logloss_raw  : binary cross-entropy of the raw candidate probabilities against the 0/1 row labels (all rows).
    - logloss_norm : probabilities normalised to sum to 1 inside each point (what the HMM emission will use); the
                     categorical cross-entropy -log p_norm(true arc), averaged over points whose label is among the
                     candidates (for the other points it is undefined: the true arc has no probability at all).
* Candidate miss rate: share of points whose label arc is not among the candidates. The same for every model; it is the
  ceiling on Top-1 (top1_all <= 1 - miss_rate).
* Route continuity (classifier-only route): per trip, take the top-1 arc of every point in GPS order, collapse
  consecutive repeats, and count the consecutive pairs (a, b) where arc a ends at the node where arc b starts
  (to_node(a) == from_node(b) in that trip's own road network). Reported pooled over all pairs and as a per-trip mean.
* Calibration: Brier score and expected calibration error (ECE, 10 equal-width bins) of the raw probabilities.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, log_loss

from .data_loading import load_arc_endpoints

EPS = 1e-15   # probability clip for log loss (avoids log(0))


# ---------------------------------------------------------------------------------------------------------------------
# point grouping helpers
# ---------------------------------------------------------------------------------------------------------------------
def point_ids(keys: pd.DataFrame) -> np.ndarray:
    """Integer id per row, one id per GPS point. Rows of a point are contiguous (data_loading sorts by trip, gps_row)."""
    trip = keys["trip_id"].to_numpy()
    row = keys["gps_row"].to_numpy()
    new = np.ones(len(keys), bool)
    new[1:] = (trip[1:] != trip[:-1]) | (row[1:] != row[:-1])
    return np.cumsum(new) - 1


def top_rows(score: np.ndarray, pid: np.ndarray) -> np.ndarray:
    """Row index of the highest-scoring candidate of every point (ties -> earliest row = nearer candidate)."""
    order = np.lexsort((np.arange(len(score)), -score, pid))     # by point, then score desc, then row asc
    first = np.r_[True, pid[order][1:] != pid[order][:-1]]
    return order[first]                                          # one row per point, in point-id order


# ---------------------------------------------------------------------------------------------------------------------
# core metrics on one set of rows
# ---------------------------------------------------------------------------------------------------------------------
def evaluate(keys: pd.DataFrame, y: np.ndarray, score: np.ndarray) -> dict:
    """All per-row / per-point metrics for one set of candidate rows (all rows of every point included)."""
    y = np.asarray(y).astype(np.int8)
    p = np.clip(np.asarray(score, dtype=np.float64), EPS, 1 - EPS)
    pid = point_ids(keys)
    n_pts = pid[-1] + 1
    pos_per_point = np.bincount(pid, weights=y, minlength=n_pts)
    assert pos_per_point.max() <= 1, "a point has more than one positive candidate"
    in_cands = pos_per_point > 0

    top = top_rows(p, pid)
    hit = y[top] == 1                                           # per point: is the top-1 candidate the label arc?
    pred = np.zeros_like(y)
    pred[top] = 1                                               # candidate-level decision for Macro F1 / confusion matrix

    # per-point normalised probabilities (HMM emission view)
    s = np.bincount(pid, weights=p, minlength=n_pts)
    p_norm_true = np.bincount(pid, weights=p * y, minlength=n_pts) / s
    ll_norm = float(-np.mean(np.log(np.clip(p_norm_true[in_cands], EPS, 1)))) if in_cands.any() else float("nan")

    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "n_points": int(n_pts),
        "n_rows": int(len(y)),
        "top1_all": float(hit.mean()),
        "top1_in_cands": float(hit[in_cands].mean()) if in_cands.any() else float("nan"),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1])),
        "logloss_raw": float(log_loss(y, p, labels=[0, 1])),
        "logloss_norm": ll_norm,
        "miss_rate": float(1 - in_cands.mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": expected_calibration_error(y, p),
        "confusion_matrix": cm.tolist(),                        # [[TN, FP], [FN, TP]] at the top-1 decision
    }


def evaluate_with_clean(keys: pd.DataFrame, y, score, label_quality) -> dict:
    """``evaluate`` on ALL points (headline) and on CLEAN points only (diagnostic)."""
    out = {"all": evaluate(keys, y, score)}
    m = np.asarray(label_quality) == "clean"
    out["clean"] = evaluate(keys[m].reset_index(drop=True), np.asarray(y)[m], np.asarray(score)[m]) if m.any() else None
    return out


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """ECE: row-weighted mean |mean predicted prob - observed positive rate| over 10 equal-width probability bins."""
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    cnt = np.bincount(bins, minlength=n_bins)
    conf = np.bincount(bins, weights=p, minlength=n_bins)
    acc = np.bincount(bins, weights=y, minlength=n_bins)
    ok = cnt > 0
    return float(np.sum(np.abs(conf[ok] - acc[ok])) / len(p))


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Per-bin mean predicted probability vs observed positive rate (for the calibration plot)."""
    p = np.asarray(p, float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    df = pd.DataFrame({"bin": bins, "p": p, "y": np.asarray(y)})
    g = df.groupby("bin").agg(mean_pred=("p", "mean"), frac_pos=("y", "mean"), count=("y", "size")).reset_index()
    return g


# ---------------------------------------------------------------------------------------------------------------------
# per-trip and sequence metrics
# ---------------------------------------------------------------------------------------------------------------------
def per_trip_top1(keys: pd.DataFrame, y, score, label_quality) -> pd.DataFrame:
    """Top-1 (all points) per trip, plus the share of far labels and the miss rate of each trip."""
    y = np.asarray(y)
    pid = point_ids(keys)
    top = top_rows(np.asarray(score, float), pid)
    n_pts = pid[-1] + 1
    in_c = np.bincount(pid, weights=y, minlength=n_pts) > 0
    df = pd.DataFrame({
        "trip_id": keys["trip_id"].to_numpy()[top],
        "hit": y[top] == 1,
        "in_cands": in_c,
        "far": np.asarray(label_quality)[top] == "far",
    })
    g = df.groupby("trip_id").agg(n_points=("hit", "size"), top1_all=("hit", "mean"),
                                  miss_rate=("in_cands", lambda s: 1 - s.mean()), far_share=("far", "mean"))
    hit_in = df[df.in_cands].groupby("trip_id")["hit"].mean().rename("top1_in_cands")
    return g.join(hit_in).reset_index()


@lru_cache(maxsize=None)
def _arcs(split: str, trip_id: str) -> np.ndarray:
    return load_arc_endpoints(split, trip_id)


def route_continuity(keys: pd.DataFrame, score, split: str) -> dict:
    """Connectivity of the classifier-only route (top-1 arc per point, GPS order, repeats collapsed)."""
    pid = point_ids(keys)
    top = top_rows(np.asarray(score, float), pid)
    sel = keys.iloc[top][["trip_id", "gps_row", "arc_id"]].sort_values(["trip_id", "gps_row"], kind="stable")
    connected = total = 0
    per_trip = {}
    for trip_id, g in sel.groupby("trip_id", sort=True):
        arcs_seq = g["arc_id"].to_numpy()
        keep = np.r_[True, arcs_seq[1:] != arcs_seq[:-1]]        # collapse consecutive repeats
        seq = arcs_seq[keep]
        if len(seq) < 2:
            per_trip[trip_id] = float("nan")
            continue
        uv = _arcs(split, trip_id)
        ok = uv[seq[:-1], 1] == uv[seq[1:], 0]                   # to_node(a) == from_node(b)
        connected += int(ok.sum())
        total += len(ok)
        per_trip[trip_id] = float(ok.mean())
    vals = [v for v in per_trip.values() if not np.isnan(v)]
    return {
        "pooled": connected / total if total else float("nan"),
        "trip_mean": float(np.mean(vals)) if vals else float("nan"),
        "n_pairs": total,
        "per_trip": per_trip,
    }
