r"""Add the guideline metrics that the FIRST training run did not write out, without retraining any model.

Added to each model's results.json / fold_results.csv / per_trip_top1.csv (first-run outputs):

1. Accuracy, precision, recall and F1 of both classes (candidate level, top-1 decision).
   These are exact functions of numbers the first run already saved. Each point gets exactly one predicted "1", so for
   a set of candidate rows with N rows, P points and Q points whose label is a candidate:
       TP = top1_all * P,  FP = P - TP,  FN = Q - TP,  TN = N - TP - FP - FN.
   N, P and Q are counted from the candidate table (labels and folds only, no predictions). Every derived confusion
   matrix is checked against the saved one (pooled out-of-fold and student) and every derived Macro F1 against the
   saved Macro F1, so the additions are verified, not assumed.
2. Sequence-level performance on the STUDENT partition, computed from the first run's saved
   student_predictions.parquet with common.metrics.sequence_metrics.
   On the train CV partition it is NOT available for the first run: out-of-fold predictions were not saved. It is
   recorded as such (future runs compute it, see common/experiment.py).

Run from the trained_models folder:  ..\.venv\Scripts\python -m common.add_missing_metrics
Safe to run twice (it recomputes the same values).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.data_loading import MODEL_DATA, load_points
from common.metrics import classification_from_counts, sequence_metrics

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["model1", "model2", "model3", "model4", "model5"]
CLS_KEYS = ["accuracy", "precision_1", "recall_1", "f1_1", "precision_0", "recall_0", "f1_0"]
NOTE = ("accuracy / precision / recall / per-class F1 and student sequence-level metrics were added after the first run "
        "by common/add_missing_metrics.py from that run's saved outputs (no retraining); CV sequence-level metrics are "
        "not available for the first run because out-of-fold predictions were not saved")


def point_counts(split: str) -> pd.DataFrame:
    """Per (fold, subset): candidate rows N, points P, points with label among candidates Q. Labels only."""
    df = pd.read_parquet(MODEL_DATA / f"{split}_candidates.parquet",
                         columns=["trip_id", "gps_row", "label", "label_quality", "fold"])
    pts = df.groupby(["trip_id", "gps_row"]).agg(fold=("fold", "first"), q=("label", "max"),
                                                 quality=("label_quality", "first"), n=("label", "size")).reset_index()
    out = []
    for subset, sel in (("all", pts), ("clean", pts[pts.quality == "clean"])):
        g = sel.groupby("fold").agg(N=("n", "sum"), P=("q", "size"), Q=("q", "sum")).reset_index()
        g["subset"] = subset
        out.append(g)
        out.append(pd.DataFrame({"fold": ["ALL"], "N": [sel.n.sum()], "P": [len(sel)], "Q": [sel.q.sum()], "subset": [subset]}))
    return pd.concat(out, ignore_index=True)


def derive(top1_all: float, top1_in: float, N: int, P: int, Q: int, macro_f1: float | None = None) -> tuple[dict, list]:
    """Confusion counts + classification metrics from Top-1 values and row/point counts (with consistency checks)."""
    tp = round(top1_all * P)
    assert abs(tp - top1_all * P) < 1e-6 * P + 1e-6, "top1_all * points is not an integer count"
    if Q:
        assert abs(tp - top1_in * Q) < 1e-6 * Q + 1e-6, "top1_all and top1_in_cands disagree on TP"
    fp, fn = P - tp, Q - tp
    tn = N - tp - fp - fn
    m = classification_from_counts(tn, fp, fn, tp)
    if macro_f1 is not None:
        assert abs((m["f1_0"] + m["f1_1"]) / 2 - macro_f1) < 1e-9, "derived Macro F1 differs from the saved one"
    return m, [[int(tn), int(fp)], [int(fn), int(tp)]]


def cv_block(fold_df: pd.DataFrame, counts: pd.DataFrame) -> tuple[dict, pd.DataFrame, dict]:
    """Per-fold classification metrics -> fold table columns, CV mean/std per subset, pooled OOF confusion counts."""
    rows, pooled = [], {}
    for _, r in fold_df.iterrows():
        c = counts[(counts.subset == r.subset) & (counts.fold == r.fold)].iloc[0]
        m, cm = derive(r.top1_all, r.top1_in_cands, int(c.N), int(c.P), int(c.Q), r.macro_f1)
        rows.append({**m})
        acc = pooled.setdefault(r.subset, np.zeros((2, 2), dtype=np.int64))
        acc += np.asarray(cm)
    add = pd.DataFrame(rows, index=fold_df.index)
    full = pd.concat([fold_df.drop(columns=[k for k in CLS_KEYS if k in fold_df]), add], axis=1)
    cv = {}
    for subset in ("all", "clean"):
        sub = full[full.subset == subset]
        cv[subset] = {k: {"mean": float(sub[k].mean()), "std": float(sub[k].std(ddof=1))} for k in CLS_KEYS}
    return cv, full, pooled


def process_seed(res: dict, fold_df: pd.DataFrame, counts_tr: pd.DataFrame, counts_st: pd.DataFrame,
                 preds: pd.DataFrame, pcol: str, pts_st: pd.DataFrame) -> tuple[dict, pd.DataFrame, dict]:
    """Add the metrics to one seed's result block (in place); return the extended fold table and per-trip seq F1."""
    cv, full, pooled = cv_block(fold_df, counts_tr)
    for subset in ("all", "clean"):
        res["cv"][subset].update(cv[subset])

    # pooled out-of-fold: the summed per-fold confusion counts must equal the saved pooled matrix
    assert pooled["all"].tolist() == res["confusion_matrix"]["cv_oof_all"], "derived OOF confusion matrix differs"
    for subset in ("all", "clean"):
        res["cv_oof_pooled"][subset].update(classification_from_counts(*pooled[subset].ravel()))

    # train (in-sample) and student
    for part, counts in (("train", counts_tr), ("student", counts_st)):
        for subset in ("all", "clean"):
            c = counts[(counts.subset == subset) & (counts.fold == "ALL")].iloc[0]
            r = res[part][subset]
            m, cm = derive(r["top1_all"], r["top1_in_cands"], int(c.N), int(c.P), int(c.Q), r["macro_f1"])
            if part == "student" and subset == "all":
                assert cm == res["confusion_matrix"]["student_all"], "derived student confusion matrix differs"
            r.update(m)

    # sequence level: student from the saved first-run predictions; CV not available (no saved OOF predictions)
    keys = preds[["trip_id", "point_id", "gps_row", "arc_id"]]
    seq = sequence_metrics(keys, preds[pcol].to_numpy(np.float64), pts_st)
    res["sequence_level"] = {
        "cv_oof": "not available for the first run (out-of-fold predictions were not saved)",
        "student": {k: seq[k] for k in ("seq_precision", "seq_recall", "seq_f1", "seq_f1_trip_mean", "pred_to_ref_length_ratio")},
    }
    return res, full, {t: v["seq_f1"] for t, v in seq["per_trip"].items()}


def mean_over_seeds(blocks: list[dict]):
    """Same averaging as the harness (_mean_tree): numbers averaged element-wise, strings kept."""
    from common.experiment import _mean_tree, _std_tree
    return _mean_tree(blocks), _std_tree(blocks)


def main() -> None:
    import argparse
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT, help="folder holding model1..model5 (default: trained_models)")
    ROOT = ap.parse_args().root
    counts_tr, counts_st = point_counts("train"), point_counts("student")
    counts_st["fold"] = counts_st["fold"].astype(str).replace({"-1": "ALL"})
    counts_st = counts_st.drop_duplicates(["subset", "fold"], keep="last")
    pts_st = load_points("student")
    for m in MODELS:
        d = ROOT / m
        res = json.loads((d / "results.json").read_text(encoding="utf-8"))
        folds = pd.read_csv(d / "fold_results.csv")
        preds = pd.read_parquet(d / "student_predictions.parquet").sort_values(["trip_id", "gps_row"], kind="stable").reset_index(drop=True)
        seeds = res["seeds"]
        fold_tables, seq_trip = [], {}
        if len(seeds) == 1:
            res, full, seq_trip[seeds[0]] = process_seed(res, folds, counts_tr, counts_st, preds, f"p_seed{seeds[0]}", pts_st)
            fold_tables.append(full)
        else:
            blocks = []
            for s in seeds:
                b = res["per_seed"][str(s)]
                b, full, seq_trip[s] = process_seed(b, folds[folds.seed == s].copy(), counts_tr, counts_st, preds, f"p_seed{s}", pts_st)
                res["per_seed"][str(s)] = b
                blocks.append(b)
                fold_tables.append(full)
            mean, std = mean_over_seeds(blocks)
            for part in ("cv", "cv_oof_pooled", "train", "student"):
                res[part] = mean[part]
            res["seed_std"] = {**res.get("seed_std", {}), **{part: std[part] for part in ("cv", "cv_oof_pooled", "train", "student")}}
            res["sequence_level"] = {"cv_oof": blocks[0]["sequence_level"]["cv_oof"],
                                     "student": mean["sequence_level"]["student"]}
            res["seed_std"]["sequence_level"] = {"student": std["sequence_level"]["student"]}
        res["post_hoc_additions"] = NOTE
        (d / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        pd.concat(fold_tables).to_csv(d / "fold_results.csv", index=False)

        pt = pd.read_csv(d / "per_trip_top1.csv", dtype={"trip_id": str})
        pt["seq_f1"] = [seq_trip[s].get(t, np.nan) if p == "student" else np.nan
                        for p, s, t in zip(pt.partition, pt.seed, pt.trip_id)]
        pt.to_csv(d / "per_trip_top1.csv", index=False)

        st = res["student"]["all"]
        print(f"{m}: student accuracy {st['accuracy']:.4f}  precision_1 {st['precision_1']:.4f}  recall_1 {st['recall_1']:.4f}  "
              f"f1_1 {st['f1_1']:.4f}  f1_0 {st['f1_0']:.4f} | seq_f1 {res['sequence_level']['student']['seq_f1']:.4f}  "
              f"seq_P {res['sequence_level']['student']['seq_precision']:.4f}  seq_R {res['sequence_level']['student']['seq_recall']:.4f}")


if __name__ == "__main__":
    main()
