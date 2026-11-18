"""Model 4 - LightGBM (gradient boosting family) for PathSnap candidate scoring.

What: an additive sequence of small decision trees, each fitted to the remaining errors (gradients of the log loss) of
the trees before it. Why: usually the strongest model family on tabular features like ours, fast to train on ~1.8M rows,
and it optimises log loss directly, so its probabilities suit the HMM emissions.

Data: RAW table (tree splits do not need scaling). Fit rows: fit_include == True with sample_weight.

Early stopping (trip-grouped): inside whatever rows ``fit`` receives (the 4 training folds during CV, all 80 trips for
the final model), EARLY_STOP_VAL_SHARE of the TRIPS are held out as a validation set (GroupShuffleSplit by trip_id, so
no trip is on both sides). Boosting stops when the validation log loss has not improved for EARLY_STOP_ROUNDS rounds.
The real per-iteration train / validation log loss of that run is saved (loss_history.json, loss curve figures).
The model is then refit on ALL the rows it received for exactly best_iteration rounds, so no training trip is wasted.
The CV validation fold is never used for early stopping.

Run from the trained_models folder:   ..\.venv\Scripts\python model4\train_lightgbm.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))           # make trained_models/common importable

from common.data_loading import SEED, feature_list  # noqa: E402
from common.experiment import ModelSpec, run_experiment  # noqa: E402

# ---- configuration -----------------------------------------------------------------------------------------------
PARAMS = {
    "objective": "binary",          # log loss on the 0/1 candidate label
    "metric": "binary_logloss",
    "learning_rate": 0.05,          # small steps; early stopping picks the number of rounds
    "num_leaves": 63,               # moderate tree complexity
    "min_child_samples": 100,       # >= 100 rows per leaf: guards against memorising single trips
    "feature_fraction": 0.8,        # 80% of features per tree (decorrelation)
    "bagging_fraction": 0.8,        # 80% of rows per iteration
    "bagging_freq": 1,
    "lambda_l2": 1.0,               # mild L2 on leaf values
    "verbose": -1,
    "num_threads": 0,               # all cores
}
CONFIG = {
    **PARAMS,
    "max_rounds": 3000,             # upper bound; early stopping decides
    "early_stopping_rounds": 100,
    "early_stop_val_share": 0.2,    # share of TRIPS held out for early stopping (trip-grouped)
    "refit_on_all_rows": True,      # refit with best_iteration rounds on every row fit() received
    "sample_weight": "clean 1.0 / ambiguous 0.5 / far 0.3 (also on the early-stopping validation rows)",
    "seed": SEED,
}


def fit(X, y, w, trip_ids, seed):
    """Early-stopped boosting on a trip-grouped inner split, then refit on all rows for best_iteration rounds."""
    params = {**PARAMS, "seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed}
    gss = GroupShuffleSplit(n_splits=1, test_size=CONFIG["early_stop_val_share"], random_state=seed)
    tr, va = next(gss.split(X, y, groups=trip_ids))
    feats = feature_list()
    dtr = lgb.Dataset(X[tr], y[tr], weight=w[tr], feature_name=feats, free_raw_data=False)
    dva = lgb.Dataset(X[va], y[va], weight=w[va], reference=dtr, feature_name=feats, free_raw_data=False)
    record: dict = {}
    booster = lgb.train(params, dtr, num_boost_round=CONFIG["max_rounds"], valid_sets=[dtr, dva],
                        valid_names=["train", "val"],
                        callbacks=[lgb.early_stopping(CONFIG["early_stopping_rounds"], verbose=False),
                                   lgb.record_evaluation(record)])
    best = int(booster.best_iteration)
    history = {"train": record["train"]["binary_logloss"], "val": record["val"]["binary_logloss"],
               "best_iteration": best, "n_val_trips": int(len(np.unique(trip_ids[va])))}
    print(f"    early stopping: best_iteration {best}, val logloss {history['val'][best - 1]:.5f}")
    if CONFIG["refit_on_all_rows"]:
        booster = lgb.train(params, lgb.Dataset(X, y, weight=w, feature_name=feats), num_boost_round=best)
    return booster, history


def predict(model, X):
    """Boosted probability of class 1."""
    return model.predict(X)


def save(model, folder: Path, seed: int):
    """Save the final booster as LightGBM text plus gain-based feature importances."""
    model.save_model(str(folder / "lightgbm_model.txt"))
    imp = dict(sorted(zip(model.feature_name(), model.feature_importance("gain").round(1).tolist()), key=lambda kv: -kv[1]))
    (folder / "feature_importance_gain.json").write_text(__import__("json").dumps(imp, indent=2), encoding="utf-8")
    return ["lightgbm_model.txt", "feature_importance_gain.json"]


if __name__ == "__main__":
    run_experiment(ModelSpec(
        name="LightGBM", family="Boosting", folder=HERE, scaled=False, config=CONFIG,
        fit=fit, predict=predict, save=save,
        loss_name="sample-weighted binary log loss (objective='binary')",
        history_ylabel="binary log loss (sample-weighted)",
    ))
