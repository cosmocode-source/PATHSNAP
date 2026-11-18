"""Model 3 - Random Forest (bagging ensemble of decision trees) for PathSnap candidate scoring.

What: an average of many decision trees, each grown on a bootstrap sample of the training rows with a random subset of
features at every split. Why: captures non-linear rules and feature interactions (e.g. "small distance AND matching
heading") without scaling, and averaging many trees is far more stable than a single tree.

Data: RAW table (trees split on thresholds, so scaling changes nothing). Fit rows: fit_include == True, weighted by
sample_weight (clean 1, ambiguous 0.5, far 0.3).

Anti-memorisation settings: max_depth 12 and min_samples_leaf 100 stop trees from growing down to single rows, and
max_samples 0.3 (each tree sees a 30% bootstrap) decorrelates the trees and keeps training time and model size down.

Run from the trained_models folder:   ..\.venv\Scripts\python model3\train_random_forest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))           # make trained_models/common importable

from common.data_loading import SEED  # noqa: E402
from common.experiment import ModelSpec, run_experiment  # noqa: E402

# ---- configuration -----------------------------------------------------------------------------------------------
CONFIG = {
    "n_estimators": 100,        # enough trees for a stable average; more trees mainly cost time and disk
    "max_depth": 12,            # caps tree size (<= 4,096 leaves) so trees cannot memorise individual rows
    "min_samples_leaf": 100,    # each leaf averages >= 100 rows -> smoother, less overconfident probabilities
    "max_features": "sqrt",     # ~5 of 22 features per split: decorrelates trees
    "bootstrap": True,
    "max_samples": 0.3,         # each tree sees a 30% bootstrap sample of the fit rows
    "criterion": "gini",
    "class_weight": None,       # imbalance is not re-weighted: Top-1 is a per-point ranking, raw probabilities stay interpretable
    "sample_weight": "clean 1.0 / ambiguous 0.5 / far 0.3",
    "random_state": SEED,
    "n_jobs": -1,
}


def fit(X, y, w, trip_ids, seed):
    """Fit the forest with sample weights. No per-iteration loss history exists for a forest -> None."""
    params = {k: v for k, v in CONFIG.items() if k != "sample_weight"}
    params["random_state"] = seed
    model = RandomForestClassifier(**params)
    model.fit(X, y, sample_weight=w)
    return model, None


def predict(model, X):
    """Mean of the trees' leaf class shares = probability of class 1."""
    return model.predict_proba(X)[:, 1]


def save(model, folder: Path, seed: int):
    """Save the final forest (compressed) and its impurity-based feature importances."""
    joblib.dump(model, folder / "random_forest_model.joblib", compress=3)
    feats = __import__("common.data_loading", fromlist=["feature_list"]).feature_list()
    imp = dict(sorted(zip(feats, model.feature_importances_.round(5).tolist()), key=lambda kv: -kv[1]))
    (folder / "feature_importance.json").write_text(__import__("json").dumps(imp, indent=2), encoding="utf-8")
    return ["random_forest_model.joblib", "feature_importance.json"]


if __name__ == "__main__":
    run_experiment(ModelSpec(
        name="Random Forest", family="Ensemble (bagging)", folder=HERE, scaled=False, config=CONFIG,
        fit=fit, predict=predict, save=save,
        loss_name="Gini impurity (per split, sample-weighted); no global loss is optimised",
    ))
