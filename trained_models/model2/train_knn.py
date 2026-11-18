"""Model 2 - K-Nearest Neighbours, distance-weighted (instance-based family) for PathSnap candidate scoring.

What: P(candidate is the true arc) = distance-weighted share of true arcs among the K most similar training
(point, candidate) rows in the 22-feature space. Why: a non-parametric model with no assumption about the shape of the
decision boundary, so it shows whether "candidates that look alike are labelled alike" holds for this task.

Data: SCALED table (KNN uses Euclidean distance, so features must be on one scale; the StandardScaler is train-fitted).

Subsample: KNN keeps the whole training set in memory and every prediction compares against all of it, so it is fitted
on a fixed random subsample of SUBSAMPLE_ROWS training rows (drawn with the seed, from the fit rows of the current
fold). 150,000 rows is ~8-10% of the ~1.4-1.8M fit rows: large enough to cover the feature space (about 15,000 true
arcs) while keeping the 5-fold CV + final fit + scoring of ~4M query rows to minutes.

Sample weights: sklearn's KNeighborsClassifier does not accept sample weights, so the far / ambiguous down-weighting
cannot be applied. Only the exclusion rule (fit_include == True) is applied, as for every model.

Run from the trained_models folder:   ..\.venv\Scripts\python model2\train_knn.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))           # make trained_models/common importable

from common.data_loading import MODEL_DATA, SEED  # noqa: E402
from common.experiment import ModelSpec, run_experiment  # noqa: E402

# ---- configuration -----------------------------------------------------------------------------------------------
CONFIG = {
    "n_neighbors": 50,          # ~5 true arcs expected among 50 neighbours at the 10% positive rate -> usable probabilities
    "weights": "distance",      # closer neighbours count more (1 / distance)
    "metric": "euclidean",      # on standardised features
    "algorithm": "auto",        # sklearn picks the search structure
    "subsample_rows": 150_000,  # see module docstring for the reason
    "subsample_seed": SEED,
    "sample_weight": "not supported by sklearn KNN -> only the fit_include exclusion rule is applied",
    "n_jobs": -1,
}


def fit(X, y, w, trip_ids, seed):
    """Fit KNN on a seeded random subsample of the fit rows (w is intentionally unused, see docstring)."""
    rng = np.random.default_rng(seed)
    n = min(CONFIG["subsample_rows"], len(y))
    idx = np.sort(rng.choice(len(y), size=n, replace=False))
    model = KNeighborsClassifier(n_neighbors=CONFIG["n_neighbors"], weights=CONFIG["weights"],
                                 metric=CONFIG["metric"], algorithm=CONFIG["algorithm"], n_jobs=CONFIG["n_jobs"])
    model.fit(X[idx], y[idx])
    print(f"    subsample {n:,} rows, positives {int(y[idx].sum()):,}")
    return model, None


def predict(model, X):
    """Distance-weighted share of true-arc neighbours = probability of class 1 (predicted in chunks to bound memory)."""
    out = np.empty(len(X))
    step = 200_000
    for i in range(0, len(X), step):
        out[i:i + step] = model.predict_proba(X[i:i + step])[:, 1]
    return out


def save(model, folder: Path, seed: int):
    """Save the fitted KNN (it stores its 150k-row subsample) and a copy of the train-fitted scaler."""
    joblib.dump(model, folder / "knn_model.joblib", compress=3)
    shutil.copy(MODEL_DATA / "scaler.joblib", folder / "scaler.joblib")
    return ["knn_model.joblib", "scaler.joblib"]


if __name__ == "__main__":
    run_experiment(ModelSpec(
        name="KNN (distance-weighted)", family="Instance-based", folder=HERE, scaled=True, config=CONFIG,
        fit=fit, predict=predict, save=save,
        loss_name="none (lazy learner, no training objective); probabilities = distance-weighted neighbour vote",
    ))
