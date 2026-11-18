"""Common experiment harness: the SAME cross-validation, final fit, student evaluation, timing and output files for
every model. A model script only supplies how to fit, predict and save its model (a ``ModelSpec``).

Procedure (identical for all five models)
-----------------------------------------
1. Load the train and student candidate tables (raw or scaled, as the model needs) with ``data_loading``.
2. 5-fold group CV on the 80 training trips using the stored ``fold`` ids (GroupKFold by trip_id, preprocessing seed).
   For fold k: FIT on rows of the other four folds with fit_include == True (with sample weights where the model
   supports them); PREDICT every row of fold k (all of its points, including those whose label is not a candidate).
3. Metrics per fold on ALL points and on CLEAN points only -> mean and std over the 5 folds.
   Out-of-fold predictions are also pooled for route continuity and per-trip Top-1.
4. Final model: fit once on all training rows with fit_include == True. Its metrics on the training rows themselves
   ("train" score) are compared with CV and student to judge over- / under-fitting.
5. Student-test (10 trips): evaluation only (headline = ALL points, clean-only = diagnostic). Nothing is fitted,
   selected or tuned on it.
6. Timing: total fit time of the final model, and inference time (model prediction only, features already built)
   per 1,000 GPS points on the student partition.

Models trained with several seeds (MLP) repeat steps 2-6 per seed; headline numbers are the mean over seeds and the
spread over seeds is stored as ``seed_std``.
"""
from __future__ import annotations

import json
import logging
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import plotting
from .data_loading import (SEED, CandidateData, feature_list, fit_mask, load_candidates, load_points)
from .metrics import evaluate_with_clean, per_trip_top1, reliability_table, route_continuity, sequence_metrics

METRIC_KEYS = ["top1_all", "top1_in_cands", "macro_f1", "logloss_raw", "logloss_norm", "brier", "ece",
               "accuracy", "precision_1", "recall_1", "f1_1", "precision_0", "recall_0", "f1_0"]
SEQ_KEYS = ["seq_precision", "seq_recall", "seq_f1", "seq_f1_trip_mean", "pred_to_ref_length_ratio"]
WEAK_TRIP_TOP1 = 0.80     # per-trip Top-1 below this is flagged as a weak trip in the per-trip plots/tables


@dataclass
class ModelSpec:
    """Everything the harness needs to know about one model."""
    name: str                                  # display name, e.g. "Logistic Regression"
    family: str                                # algorithm family, e.g. "Linear / probabilistic"
    folder: Path                               # output folder (trained_models/modelN or a sub-folder)
    scaled: bool                               # True -> scaled table (LR, KNN, MLP); False -> raw (RF, LightGBM)
    config: dict                               # hyperparameters + choices; written to config.json
    fit: Callable[..., tuple[Any, dict | None]]   # fit(X, y, w, trip_ids, seed) -> (model, history or None)
    predict: Callable[[Any, np.ndarray], np.ndarray]   # predict(model, X) -> P(candidate is the true arc)
    save: Callable[[Any, Path, int], list[str]]        # save(model, folder, seed) -> saved file names
    loss_name: str                             # loss function / objective, written to results
    seeds: list[int] = field(default_factory=lambda: [SEED])
    history_ylabel: str = "loss"


# ---------------------------------------------------------------------------------------------------------------------
def _logger(folder: Path) -> logging.Logger:
    folder.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(f"pathsnap.{folder.name}")
    log.handlers.clear()
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.FileHandler(folder / "training_log.txt", mode="w", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.propagate = False
    return log


def _metric_row(res: dict) -> dict:
    return {k: res[k] for k in METRIC_KEYS}


def _strip(res: dict | None) -> dict | None:
    """Metrics without the bulky confusion matrix (for compact tables)."""
    if res is None:
        return None
    return {k: v for k, v in res.items() if k != "confusion_matrix"}


def _mean_tree(trees: list):
    """Element-wise mean over a list of identically shaped nested dicts/lists (numbers averaged, others from the first)."""
    t0 = trees[0]
    if isinstance(t0, dict):
        return {k: _mean_tree([t[k] for t in trees]) for k in t0}
    if isinstance(t0, list) and t0 and isinstance(t0[0], (list, int, float)):
        return np.mean(np.asarray(trees, dtype=float), axis=0).tolist()
    if isinstance(t0, (int, float)) and not isinstance(t0, bool):
        return float(np.nanmean([float(t) for t in trees]))
    return t0


def _std_tree(trees: list):
    """Element-wise std over seeds for the numeric leaves (spread between seeds)."""
    t0 = trees[0]
    if isinstance(t0, dict):
        return {k: _std_tree([t[k] for t in trees]) for k in t0}
    if isinstance(t0, (int, float)) and not isinstance(t0, bool):
        return float(np.nanstd([float(t) for t in trees]))
    return None


def _check_points(data: CandidateData, log: logging.Logger) -> None:
    """The candidate table must cover every GPS point of the points table, with the same label_in_candidates flag."""
    pts = load_points(data.split)
    g = pd.DataFrame({"trip_id": data.keys.trip_id, "gps_row": data.keys.gps_row, "y": data.y})
    per_pt = g.groupby(["trip_id", "gps_row"])["y"].max().rename("in_c").reset_index()
    m = pts.merge(per_pt, on=["trip_id", "gps_row"], how="left")
    assert m["in_c"].notna().all(), "some points have no candidate rows"
    assert (m["in_c"].astype(bool) == m["label_in_candidates"].astype(bool)).all(), "label_in_candidates mismatch"
    log.info(f"[{data.split}] {len(pts):,} points, {data.n:,} candidate rows, "
             f"miss rate {1 - pts.label_in_candidates.mean():.4f}, fit_include rows {data.fit_include.sum():,}")


# ---------------------------------------------------------------------------------------------------------------------
def _run_one_seed(spec: ModelSpec, train: CandidateData, student: CandidateData, seed: int, log: logging.Logger):
    folds = sorted(set(train.fold.tolist()))
    trip_ids = train.keys["trip_id"].to_numpy()
    oof = np.full(train.n, np.nan)
    fold_rows, histories, cv_fit_s = [], {}, []

    # ---- 1) 5-fold group CV --------------------------------------------------------------------------------------
    for k in folds:
        fit_idx = fit_mask(train, [f for f in folds if f != k])
        val_idx = train.fold == k
        t0 = time.perf_counter()
        model, hist = spec.fit(train.X[fit_idx], train.y[fit_idx], train.w[fit_idx], trip_ids[fit_idx], seed)
        fit_s = time.perf_counter() - t0
        cv_fit_s.append(fit_s)
        p = spec.predict(model, train.X[val_idx])
        oof[val_idx] = p
        res = evaluate_with_clean(train.keys[val_idx].reset_index(drop=True), train.y[val_idx], p, train.label_quality[val_idx])
        for subset in ("all", "clean"):
            fold_rows.append({"seed": seed, "fold": k, "subset": subset, "n_fit_rows": int(fit_idx.sum()),
                              "n_val_points": res[subset]["n_points"], "fit_seconds": round(fit_s, 2), **_metric_row(res[subset])})
        if hist is not None:
            histories[f"fold{k}"] = hist
        log.info(f"seed {seed} fold {k}: fit {fit_s:7.1f}s  rows {fit_idx.sum():,}  "
                 f"top1_all {res['all']['top1_all']:.4f}  top1_in_cands {res['all']['top1_in_cands']:.4f}  "
                 f"macroF1 {res['all']['macro_f1']:.4f}  ll_raw {res['all']['logloss_raw']:.4f}  ll_norm {res['all']['logloss_norm']:.4f}")
        del model
    assert not np.isnan(oof).any(), "every training row must get an out-of-fold prediction"

    fold_df = pd.DataFrame(fold_rows)
    cv = {}
    for subset in ("all", "clean"):
        sub = fold_df[fold_df.subset == subset]
        cv[subset] = {m: {"mean": float(sub[m].mean()), "std": float(sub[m].std(ddof=1))} for m in METRIC_KEYS}
    oof_res = evaluate_with_clean(train.keys, train.y, oof, train.label_quality)
    oof_cont = route_continuity(train.keys, oof, "train")
    oof_trip = per_trip_top1(train.keys, train.y, oof, train.label_quality)
    oof_seq = sequence_metrics(train.keys, oof, load_points("train"))

    # ---- 2) final fit on all training rows (fit_include) ---------------------------------------------------------
    all_idx = fit_mask(train, folds)
    t0 = time.perf_counter()
    model, hist = spec.fit(train.X[all_idx], train.y[all_idx], train.w[all_idx], trip_ids[all_idx], seed)
    final_fit_s = time.perf_counter() - t0
    if hist is not None:
        histories["final"] = hist
    log.info(f"seed {seed} final fit: {final_fit_s:.1f}s on {all_idx.sum():,} rows")

    p_train = spec.predict(model, train.X)
    train_res = evaluate_with_clean(train.keys, train.y, p_train, train.label_quality)

    # ---- 3) student-test: evaluation only ----------------------------------------------------------------------
    t0 = time.perf_counter()
    p_student = spec.predict(model, student.X)
    infer_s = time.perf_counter() - t0
    student_res = evaluate_with_clean(student.keys, student.y, p_student, student.label_quality)
    student_cont = route_continuity(student.keys, p_student, "student")
    student_trip = per_trip_top1(student.keys, student.y, p_student, student.label_quality)
    student_seq = sequence_metrics(student.keys, p_student, load_points("student"))
    n_student_points = student_res["all"]["n_points"]
    log.info(f"seed {seed} student: top1_all {student_res['all']['top1_all']:.4f}  macroF1 {student_res['all']['macro_f1']:.4f}  "
             f"ll_norm {student_res['all']['logloss_norm']:.4f}  continuity {student_cont['pooled']:.4f}  "
             f"inference {1000 * infer_s / n_student_points * 1000:.1f} ms / 1,000 points")

    saved = spec.save(model, spec.folder, seed)

    result = {
        "cv": cv,
        "cv_oof_pooled": {"all": _strip(oof_res["all"]), "clean": _strip(oof_res["clean"])},
        "train": {"all": _strip(train_res["all"]), "clean": _strip(train_res["clean"])},
        "student": {"all": _strip(student_res["all"]), "clean": _strip(student_res["clean"])},
        "confusion_matrix": {"cv_oof_all": oof_res["all"]["confusion_matrix"], "student_all": student_res["all"]["confusion_matrix"]},
        "route_continuity": {
            "cv_oof": {k: oof_cont[k] for k in ("pooled", "trip_mean", "n_pairs")},
            "student": {k: student_cont[k] for k in ("pooled", "trip_mean", "n_pairs")},
        },
        "sequence_level": {
            "cv_oof": {k: oof_seq[k] for k in SEQ_KEYS},
            "student": {k: student_seq[k] for k in SEQ_KEYS},
        },
        "timing": {
            "cv_fit_seconds_per_fold": [round(s, 2) for s in cv_fit_s],
            "final_fit_seconds": round(final_fit_s, 2),
            "inference_seconds_student": round(infer_s, 4),
            "inference_ms_per_1000_points": round(1000 * infer_s / n_student_points * 1000, 3),
            "n_student_points": n_student_points,
        },
    }
    extras = {
        "fold_df": fold_df, "histories": histories, "saved": saved,
        "oof": oof, "p_student": p_student, "oof_trip": oof_trip, "student_trip": student_trip,
        "oof_cont_trip": oof_cont["per_trip"], "student_cont_trip": student_cont["per_trip"],
        "oof_seq_trip": {t: v["seq_f1"] for t, v in oof_seq["per_trip"].items()},
        "student_seq_trip": {t: v["seq_f1"] for t, v in student_seq["per_trip"].items()},
    }
    return result, extras


# ---------------------------------------------------------------------------------------------------------------------
def run_experiment(spec: ModelSpec) -> dict:
    """Run CV + final fit + student evaluation for ``spec`` and write every output file into ``spec.folder``."""
    folder = spec.folder
    fig_dir = folder / "figures"
    log = _logger(folder)
    log.info(f"=== {spec.name} ({spec.family}) | seeds {spec.seeds} | table {'scaled' if spec.scaled else 'raw'} ===")
    log.info(f"python {platform.python_version()} on {platform.platform()}")

    # config.json: every setting of this run (hyperparameters, seeds, features, data rules)
    config = {
        "model": spec.name, "family": spec.family, "seeds": spec.seeds,
        "table": "scaled" if spec.scaled else "raw", "features": feature_list(), "n_features": len(feature_list()),
        "loss_function": spec.loss_name,
        "fit_rows": "train rows with fit_include == True (other CV folds during CV, all folds for the final model)",
        "cv": "5-fold GroupKFold by trip_id, fold ids stored by preprocessing (cleaned_data/model_data/folds.csv)",
        "evaluation": "unweighted; ALL points headline, CLEAN points diagnostic; student-test evaluation only",
        "hyperparameters": spec.config,
    }
    (folder / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    t_all = time.perf_counter()
    train = load_candidates("train", scaled=spec.scaled)
    student = load_candidates("student", scaled=spec.scaled)
    _check_points(train, log)
    _check_points(student, log)

    per_seed, extras_by_seed = {}, {}
    for seed in spec.seeds:
        per_seed[seed], extras_by_seed[seed] = _run_one_seed(spec, train, student, seed, log)

    seeds = spec.seeds
    headline = per_seed[seeds[0]] if len(seeds) == 1 else _mean_tree([per_seed[s] for s in seeds])
    first = extras_by_seed[seeds[0]]

    # train / CV / student gap (over- or under-fitting evidence)
    gap = {}
    for m in ("top1_all", "macro_f1", "logloss_norm"):
        tr, cvm, st = headline["train"]["all"][m], headline["cv"]["all"][m]["mean"], headline["student"]["all"][m]
        gap[m] = {"train": tr, "cv_mean": cvm, "student": st, "train_minus_cv": tr - cvm, "cv_minus_student": cvm - st}

    results = {
        "model": spec.name, "family": spec.family, "loss_function": spec.loss_name, "seeds": seeds,
        "n_features": len(train.features), "candidate_miss_rate": {
            "train": headline["cv_oof_pooled"]["all"]["miss_rate"], "student": headline["student"]["all"]["miss_rate"]},
        **headline,
        "train_cv_student_gap": gap,
        "saved_files": sorted({f for s in seeds for f in extras_by_seed[s]["saved"]}),
        "total_wall_seconds": round(time.perf_counter() - t_all, 1),
    }
    if len(seeds) > 1:
        results["seed_std"] = _std_tree([per_seed[s] for s in seeds])
        results["per_seed"] = per_seed
    (folder / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # fold_results.csv (every seed x fold x subset) and per-trip Top-1
    pd.concat([extras_by_seed[s]["fold_df"] for s in seeds]).to_csv(folder / "fold_results.csv", index=False)
    trips = []
    for s in seeds:
        e = extras_by_seed[s]
        a = e["oof_trip"].assign(partition="train_cv_oof", seed=s)
        a["route_continuity"] = a.trip_id.map(e["oof_cont_trip"])
        a["seq_f1"] = a.trip_id.map(e["oof_seq_trip"])
        b = e["student_trip"].assign(partition="student", seed=s)
        b["route_continuity"] = b.trip_id.map(e["student_cont_trip"])
        b["seq_f1"] = b.trip_id.map(e["student_seq_trip"])
        trips += [a, b]
    trips_df = pd.concat(trips, ignore_index=True)
    trips_df["weak_trip"] = trips_df["top1_all"] < WEAK_TRIP_TOP1
    trips_df.to_csv(folder / "per_trip_top1.csv", index=False)

    # student predictions (small) for the comparison notebook and the later HMM stage; one score column per seed
    pred = student.keys.copy()
    pred["label"] = student.y
    pred["label_quality"] = student.label_quality
    for s in seeds:
        pred[f"p_seed{s}"] = extras_by_seed[s]["p_student"].astype(np.float32)
    pred.to_parquet(folder / "student_predictions.parquet", index=False)

    # figures (first seed for the per-row plots; all from real outputs)
    plotting.plot_fold_metrics(first["fold_df"], spec.name, fig_dir / "cv_fold_metrics.png")
    plotting.plot_confusion(per_seed[seeds[0]]["confusion_matrix"]["student_all"],
                            f"{spec.name}: student-test, top-1 decision", fig_dir / "confusion_student.png")
    plotting.plot_confusion(per_seed[seeds[0]]["confusion_matrix"]["cv_oof_all"],
                            f"{spec.name}: train CV out-of-fold, top-1 decision", fig_dir / "confusion_cv_oof.png")
    plotting.plot_reliability(
        {"train CV out-of-fold": reliability_table(train.y, first["oof"]),
         "student-test": reliability_table(student.y, first["p_student"])},
        f"{spec.name}: calibration of raw candidate probabilities", fig_dir / "calibration.png")
    plotting.plot_per_trip(first["student_trip"], f"{spec.name}: Top-1 per student-test trip (red < {WEAK_TRIP_TOP1})",
                           fig_dir / "per_trip_top1_student.png", WEAK_TRIP_TOP1)
    plotting.plot_per_trip(first["oof_trip"], f"{spec.name}: Top-1 per train trip, CV out-of-fold (red < {WEAK_TRIP_TOP1})",
                           fig_dir / "per_trip_top1_train_cv.png", WEAK_TRIP_TOP1)
    hist_all = {s: extras_by_seed[s]["histories"] for s in seeds if extras_by_seed[s]["histories"]}
    if hist_all:   # only models with a real per-iteration / per-epoch history (LightGBM, MLP)
        (folder / "loss_history.json").write_text(json.dumps({str(k): v for k, v in hist_all.items()}), encoding="utf-8")
        for s, hs in hist_all.items():
            plotting.plot_loss_curves(hs, f"{spec.name} (seed {s}): real loss history per CV fold and final fit",
                                      spec.history_ylabel, fig_dir / f"loss_curves_seed{s}.png")

    log.info(f"CV top1_all {headline['cv']['all']['top1_all']['mean']:.4f} +/- {headline['cv']['all']['top1_all']['std']:.4f} | "
             f"student top1_all {headline['student']['all']['top1_all']:.4f} | total {results['total_wall_seconds']}s")
    log.info(f"outputs written to {folder}")
    return results
