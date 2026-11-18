"""Bayesian hyperparameter optimisation of the chosen model (LightGBM, trained_models/model4) - shared library.

The notebook ``tune_lightgbm.ipynb`` in this folder drives everything; this module holds the code it calls, so the
long-running study can also be resumed from a plain ``python`` process (``python tune_lib.py``) with identical code.

What it does
------------
* Loads the RAW candidate table with the shared loader (``common.data_loading``): the 22 selected features, labels,
  sample weights, ``fit_include`` and the 5 stored GroupKFold folds (by trip_id). ``arc_id`` is never a feature.
* Objective of one Optuna trial = mean over the 5 CV folds of Top-1 on label-in-candidates points of the CV validation
  fold, using the shared metric code (``common.metrics.evaluate``), so the score is defined exactly like the comparison.
* NESTED early stopping: inside each fold, 20% of the fold's TRAINING trips are held out (GroupShuffleSplit by trip_id,
  seed 42, the same split the untuned model uses) and boosting stops when their weighted log loss stops improving
  (100 rounds patience, cap 2000 rounds). The CV validation fold and the student partition are never used for stopping.
* Every trial also records CV Top-1 on ALL points, normalised log loss, Macro F1 (mean and std), the per-fold values,
  the best iteration of every fold and the wall time, as Optuna user attributes.
* The study is stored in SQLite (``optuna_study.db``), so it can be resumed after an interruption.

Evaluator data is never read. Student-test is never read by anything in this module.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
from sklearn.model_selection import GroupShuffleSplit

HERE = Path(__file__).resolve().parent                     # trained_models/model4/tuning
TRAINED = HERE.parents[1]                                  # trained_models
sys.path.insert(0, str(TRAINED))                           # make trained_models/common importable

from common.data_loading import SEED, feature_list, fit_mask, load_candidates  # noqa: E402
from common.metrics import evaluate  # noqa: E402

# ---------------------------------------------------------------------------------------------------------------------
# CONFIG: every setting of the study in one place (also written to config.json by the notebook)
# ---------------------------------------------------------------------------------------------------------------------
CONFIG = {
    "seed": SEED,                                  # 42: sampler, inner split, LightGBM seeds, bootstrap
    "n_trials": 46,                                # COMPLETE trials = 40 unique configurations + 6 duplicate evaluations
                                                   # (task: 60, at least 40; cut to 40 unique for the time budget)
    "n_workers": 2,                                # two processes share the SQLite study (8 LightGBM threads each)
    "study_name": "lightgbm_top1_in_cands",
    "storage": f"sqlite:///{(HERE / 'optuna_study.db').as_posix()}",
    "sampler": "TPE (optuna.samplers.TPESampler, seed 42, 10 random start-up trials = Optuna default)",
    "pruner": "none (every trial is run on all 5 folds so its record is complete; the budget allows it)",
    "objective": "maximise mean over the 5 stored CV folds of Top-1 on label-in-candidates points (top1_in_cands)",
    "tie_breaker": "trials within one fold standard error (std / sqrt(5)) of the best mean: lowest mean normalised log loss",
    "max_rounds": 2000,                            # cap on boosting rounds; early stopping chooses the rest
    "early_stopping_rounds": 100,                  # same patience as the untuned model
    "early_stop_val_share": 0.2,                   # share of the fold's TRAINING trips held out for early stopping
    "refit_inside_tuning": False,                  # tuning scores the early-stopped model (see notebook for why)
    "dataset_params": {"feature_pre_filter": False, "verbose": -1},   # lets min_child_samples vary on one Dataset
    "fixed_params": {                              # NOT searched: identical to the untuned model
        "objective": "binary", "metric": "binary_logloss", "bagging_freq": 1, "verbose": -1,
        "num_threads": int(__import__("os").environ.get("PATHSNAP_THREADS", "0")),   # 0 = all cores; 8 per worker in the study
    },
    "class_weighting": "none, as in the untuned model (it uses sample weights only: clean 1.0 / ambiguous 0.5 / far 0.3)",
    "search_space": {
        "learning_rate": "float 0.01 - 0.2, log",
        "num_leaves": "int 15 - 255, log",
        "max_depth": "categorical {-1, 4, 5, ..., 12}",
        "min_child_samples": "int 10 - 300",
        "feature_fraction": "float 0.5 - 1.0",
        "bagging_fraction": "float 0.5 - 1.0 (bagging_freq = 1)",
        "lambda_l1": "float 1e-3 - 10, log",
        "lambda_l2": "float 1e-3 - 10, log",
        "min_split_gain": "float 0 - 1",
    },
}
MAX_DEPTH_CHOICES = [-1, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# Parallel workers MUST use different sampler seeds: with the same seed two workers propose identical parameters
# (this happened in the first parallel run; those repeated evaluations are kept in the log and flagged as duplicates).
WORKER = int(__import__("os").environ.get("PATHSNAP_WORKER", "0"))   # 0, 1, ... -> sampler seed 42, 43, ...

# Untuned model 4 parameters, enqueued as trial 0 so the baseline is inside the search. lambda_l1 = 0 and
# min_split_gain = 0 in the untuned model; 0 is outside the log range of lambda_l1, so its lowest value 1e-3 is used
# (an L1 penalty of 0.001 on leaf values is negligible next to lambda_l2 = 1).
BASELINE_PARAMS = {
    "learning_rate": 0.05, "num_leaves": 63, "max_depth": -1, "min_child_samples": 100,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "lambda_l1": 1e-3, "lambda_l2": 1.0, "min_split_gain": 0.0,
}
TRIAL_METRICS = ["top1_in_cands", "top1_all", "logloss_norm", "macro_f1", "logloss_raw"]


def suggest(trial: optuna.Trial) -> dict:
    """Search space (reasons for every range are in the notebook)."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "max_depth": trial.suggest_categorical("max_depth", MAX_DEPTH_CHOICES),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 300),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
    }


def lgb_params(p: dict, seed: int = SEED) -> dict:
    """Full LightGBM parameter dict = fixed part + searched part + seeds (same seed keys as the untuned model)."""
    return {**CONFIG["fixed_params"], **p, "seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed}


# ---------------------------------------------------------------------------------------------------------------------
# Fold data, built once and reused by every trial
# ---------------------------------------------------------------------------------------------------------------------
class FoldData:
    """For each CV fold k: inner train / early-stopping Datasets (from the fold's training trips) and the CV
    validation rows. LightGBM Datasets are constructed once (binning) and reused by all trials."""

    def __init__(self):
        self.train = load_candidates("train", scaled=False)
        tr = self.train
        trip = tr.keys["trip_id"].to_numpy()
        self.folds = sorted(set(tr.fold.tolist()))
        feats = feature_list()
        self.items = []
        for k in self.folds:
            fit_idx = np.flatnonzero(fit_mask(tr, [f for f in self.folds if f != k]))   # 4 folds, fit_include only
            gss = GroupShuffleSplit(n_splits=1, test_size=CONFIG["early_stop_val_share"], random_state=SEED)
            a, b = next(gss.split(fit_idx, groups=trip[fit_idx]))                        # trip-grouped inner split
            i_tr, i_es = fit_idx[a], fit_idx[b]
            dtr = lgb.Dataset(tr.X[i_tr], tr.y[i_tr], weight=tr.w[i_tr], feature_name=feats,
                              params=CONFIG["dataset_params"], free_raw_data=False).construct()
            des = lgb.Dataset(tr.X[i_es], tr.y[i_es], weight=tr.w[i_es], reference=dtr, feature_name=feats,
                              params=CONFIG["dataset_params"], free_raw_data=False).construct()
            val = np.flatnonzero(tr.fold == k)                                           # ALL rows of fold k
            self.items.append({
                "fold": k, "dtr": dtr, "des": des, "X_val": tr.X[val], "y_val": tr.y[val],
                "keys_val": tr.keys.iloc[val].reset_index(drop=True),
                "n_fit_rows": len(fit_idx), "n_es_trips": int(len(np.unique(trip[i_es]))),
            })


def cv_score(params: dict, fd: FoldData) -> dict:
    """Run one configuration on the 5 folds (nested early stopping, no refit). Returns per-fold and mean metrics."""
    per_fold, best_its = [], []
    for it in fd.items:
        booster = lgb.train(params, it["dtr"], num_boost_round=CONFIG["max_rounds"], valid_sets=[it["des"]],
                            valid_names=["early_stop"],
                            callbacks=[lgb.early_stopping(CONFIG["early_stopping_rounds"], verbose=False)])
        best = int(booster.best_iteration)
        p = booster.predict(it["X_val"], num_iteration=best)
        r = evaluate(it["keys_val"], it["y_val"], p)
        per_fold.append({m: float(r[m]) for m in TRIAL_METRICS})
        best_its.append(best)
    out = {"per_fold": per_fold, "best_iterations": best_its}
    for m in TRIAL_METRICS:
        v = np.array([f[m] for f in per_fold])
        out[m] = {"mean": float(v.mean()), "std": float(v.std(ddof=1))}
    return out


def make_objective(fd: FoldData, log_path: Path):
    """Optuna objective: mean CV top1_in_cands; every other metric stored as a user attribute."""
    def objective(trial: optuna.Trial) -> float:
        t0 = time.perf_counter()
        p = suggest(trial)
        res = cv_score(lgb_params(p), fd)
        secs = time.perf_counter() - t0
        for m in TRIAL_METRICS:
            trial.set_user_attr(f"cv_{m}_mean", res[m]["mean"])
            trial.set_user_attr(f"cv_{m}_std", res[m]["std"])
        trial.set_user_attr("per_fold", res["per_fold"])
        trial.set_user_attr("best_iterations", res["best_iterations"])
        trial.set_user_attr("mean_best_iteration", float(np.mean(res["best_iterations"])))
        trial.set_user_attr("seconds", round(secs, 1))
        line = (f"trial {trial.number:3d}  top1_in_cands {res['top1_in_cands']['mean']:.5f} +/- {res['top1_in_cands']['std']:.5f}  "
                f"top1_all {res['top1_all']['mean']:.5f}  ll_norm {res['logloss_norm']['mean']:.5f}  "
                f"best_it {res['best_iterations']}  {secs:6.1f}s  params {json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in p.items()})}")
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S  ") + line + "\n")
        return res["top1_in_cands"]["mean"]
    return objective


def get_study() -> optuna.Study:
    """Create or load the SQLite-backed study (TPE, seed 42, maximise)."""
    return optuna.create_study(study_name=CONFIG["study_name"], storage=CONFIG["storage"], direction="maximize",
                               sampler=optuna.samplers.TPESampler(seed=SEED + WORKER), load_if_exists=True)


def run_study(n_trials: int | None = None, fd: FoldData | None = None) -> optuna.Study:
    """Run (or resume) the study until it has ``n_trials`` complete trials. Trial 0 = the untuned parameters."""
    n_trials = n_trials or CONFIG["n_trials"]
    study = get_study()
    log_path = HERE / "tuning_log.txt"
    if len(study.trials) == 0:
        study.enqueue_trial(BASELINE_PARAMS, user_attrs={"note": "untuned model 4 parameters (baseline inside the search)"})
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S  ") + f"study created: {json.dumps({k: v for k, v in CONFIG.items() if k != 'search_space'})}\n")
    done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    if done < n_trials:
        fd = fd or FoldData()
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        # MaxTrialsCallback stops EVERY worker once the shared study holds n_trials complete trials
        stop = optuna.study.MaxTrialsCallback(n_trials, states=(optuna.trial.TrialState.COMPLETE,))
        study.optimize(make_objective(fd, log_path), n_trials=n_trials, gc_after_trial=True, callbacks=[stop])
    return study


def select_trial(study: optuna.Study):
    """Best trial with the tie-break rule: among trials whose mean CV top1_in_cands is within one fold standard error
    (std / sqrt(5)) of the best mean, take the one with the lowest mean normalised log loss."""
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    best = max(trials, key=lambda t: t.value)
    se = best.user_attrs["cv_top1_in_cands_std"] / np.sqrt(5)
    close = [t for t in trials if t.value >= best.value - se]
    chosen = min(close, key=lambda t: t.user_attrs["cv_logloss_norm_mean"])
    return chosen, best, se, close


if __name__ == "__main__":       # resume / run the study from a terminal with the same code as the notebook
    s = run_study(int(sys.argv[1]) if len(sys.argv) > 1 else None)
    print("complete trials:", sum(t.state == optuna.trial.TrialState.COMPLETE for t in s.trials), "best:", s.best_value)
