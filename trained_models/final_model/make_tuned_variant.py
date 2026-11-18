"""Build the TUNED LightGBM variant for the HMM stage: trained_models/final_model/variants/tuned/.

The primary final model stays the UNTUNED LightGBM (decision rule in trained_models/model4/tuning/). The team asked to
offer the tuned configuration to the HMM stage as well, so it can be compared on route-level metrics. This script writes:
  * lightgbm_model.txt          - tuned final model (copied from model4/tuning/tuned_model/, fitted on all fit_include rows)
  * student_predictions.parquet - its student-test candidate scores (copied, same file the harness wrote)
  * oof_predictions.parquet     - 5-fold out-of-fold probabilities on the 80 training trips, in the SAME format as
                                  trained_models/hmm/results/oof_predictions.parquet (keys + 'probability'),
                                  fitted with the tuned parameters and the same procedure as the untuned model
                                  (trip-grouped early stopping on 20% of the training trips, then refit on all fold rows)
  * config.json                 - parameters, source and the CV / student numbers of the variant
Uses only training rows to fit; student rows are only predicted. The evaluator partition is never read.
Run from anywhere:  python trained_models/final_model/make_tuned_variant.py
"""
import json, shutil, sys
from pathlib import Path
import numpy as np, lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit

HERE = Path(__file__).resolve().parent
TRAINED = HERE.parent
TUNING = TRAINED / "model4" / "tuning"
sys.path.insert(0, str(TRAINED)); sys.path.insert(0, str(TUNING))
from common.data_loading import SEED, feature_list, fit_mask, load_candidates   # noqa: E402
import tune_lib as T                                                            # noqa: E402

OUT = HERE / "variants" / "tuned"; OUT.mkdir(parents=True, exist_ok=True)
frozen = json.loads((TUNING / "best_params.json").read_text())
PARAMS = {**{k: v for k, v in T.CONFIG["fixed_params"].items() if k != "num_threads"}, "num_threads": 0, **frozen["params"]}


def fit(X, y, w, trip_ids, seed=SEED):
    """Same procedure as the untuned model 4: early stopping on a trip-grouped 20% of the training trips, then refit."""
    p = {**PARAMS, "seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed}
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=T.CONFIG["early_stop_val_share"], random_state=seed).split(X, y, groups=trip_ids))
    feats = feature_list()
    dtr = lgb.Dataset(X[tr], y[tr], weight=w[tr], feature_name=feats)
    dva = lgb.Dataset(X[va], y[va], weight=w[va], reference=dtr, feature_name=feats)
    b = lgb.train(p, dtr, num_boost_round=T.CONFIG["max_rounds"], valid_sets=[dva],
                  callbacks=[lgb.early_stopping(T.CONFIG["early_stopping_rounds"], verbose=False)])
    return lgb.train(p, lgb.Dataset(X, y, weight=w, feature_name=feats), num_boost_round=int(b.best_iteration)), int(b.best_iteration)


data = load_candidates("train", scaled=False)
folds = sorted(set(data.fold.tolist())); trips = data.keys.trip_id.to_numpy()
oof = np.full(data.n, np.nan); best_its = {}
for k in folds:
    m = fit_mask(data, [f for f in folds if f != k]); v = data.fold == k
    model, best = fit(data.X[m], data.y[m], data.w[m], trips[m]); best_its[int(k)] = best
    oof[v] = model.predict(data.X[v]); print(f"fold {k}: best_iteration {best}", flush=True)
assert np.isfinite(oof).all()
out = data.keys.copy(); out["probability"] = oof
out.to_parquet(OUT / "oof_predictions.parquet", index=False)

for f in ("lightgbm_model.txt", "student_predictions.parquet"):
    shutil.copy2(TUNING / "tuned_model" / f, OUT / f)
res = json.loads((TUNING / "tuned_model" / "results.json").read_text())
cfg = {"variant": "tuned", "role": "alternative for the HMM stage (primary final model = untuned, see ../../config.json)",
       "params": PARAMS, "source_trial": frozen["chosen_trial"], "features": feature_list(), "input_table": "raw candidate table",
       "oof_best_iteration_per_fold": best_its, "oof_format": "same columns as trained_models/hmm/results/oof_predictions.parquet",
       "student_prediction_column": "p_seed42",
       "metrics": {"cv_top1_all": res["cv"]["all"]["top1_all"]["mean"], "cv_top1_in_cands": res["cv"]["all"]["top1_in_cands"]["mean"],
                   "cv_logloss_norm": res["cv"]["all"]["logloss_norm"]["mean"], "student_top1_all": res["student"]["all"]["top1_all"],
                   "student_logloss_norm": res["student"]["all"]["logloss_norm"], "route_continuity_student": res["route_continuity"]["student"]["pooled"]},
       "seed": SEED}
(OUT / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print("tuned variant written to", OUT)
