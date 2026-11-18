"""Model 1 - Logistic Regression (linear / probabilistic family) for PathSnap candidate scoring.

What: a class-weighted, L2-regularised logistic regression that scores every (GPS point, candidate arc) pair with
P(candidate is the true arc). Why: the simplest probabilistic baseline. It gives calibrated-ish probabilities that the
HMM can use as emissions, trains in seconds and shows how far a purely linear decision boundary over the 22 features gets.

Data: SCALED table (winsorised, log1p on the two skewed distances, StandardScaler fitted on train) because L2 and
lbfgs need features on comparable scales. Fit rows: fit_include == True, weighted by sample_weight (clean 1,
ambiguous 0.5, far 0.3) times the train-learned class weight. Evaluation: common/experiment.py (identical for all models).

Run from the trained_models folder:   ..\.venv\Scripts\python model1\train_logreg.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))           # make trained_models/common importable

from common.data_loading import MODEL_DATA, SEED, load_prep_config  # noqa: E402
from common.experiment import ModelSpec, run_experiment             # noqa: E402

# ---- configuration: every setting of this model lives here (copied into config.json) ---------------------------------
PREP = load_prep_config()
CLASS_WEIGHT = {int(k): float(v) for k, v in PREP["class_weights"].items()}   # learned on train by preprocessing
CONFIG = {
    "penalty": "l2",           # ridge penalty: shrinks the correlated distance features (VIF ~11) instead of dropping them
    "C": 1.0,                  # inverse regularisation strength; sklearn default, no tuning at this stage
    "solver": "lbfgs",         # standard for L2 + ~1.8M rows x 22 dense features
    "max_iter": 2000,          # enough for convergence on standardised features (checked in the log: n_iter_)
    "tol": 1e-4,
    "class_weight": CLASS_WEIGHT,   # ~0.56 / 4.86: compensates the 1-in-10 positive rate
    "sample_weight": "clean 1.0 / ambiguous 0.5 / far 0.3 (multiplied by class weight)",
    "random_state": SEED,
}


def fit(X, y, w, trip_ids, seed):
    """Fit one logistic regression. trip_ids are not needed here (no inner validation split). No history -> None."""
    model = LogisticRegression(
        C=CONFIG["C"], solver=CONFIG["solver"], max_iter=CONFIG["max_iter"], tol=CONFIG["tol"],
        class_weight=CONFIG["class_weight"], random_state=seed,
    )
    model.fit(X, y, sample_weight=w)      # sklearn multiplies sample_weight by class_weight internally
    print(f"    lbfgs iterations: {int(model.n_iter_[0])}")
    return model, None


def predict(model, X):
    """P(candidate is the true arc) = probability of class 1."""
    return model.predict_proba(X)[:, 1]


def save(model, folder: Path, seed: int):
    """Save the final model and a copy of the train-fitted scaler (needed to score new trips identically)."""
    joblib.dump(model, folder / "logreg_model.joblib")
    shutil.copy(MODEL_DATA / "scaler.joblib", folder / "scaler.joblib")
    coef = dict(zip(PREP["selected_features"], np.round(model.coef_[0], 5).tolist()))
    (folder / "coefficients.json").write_text(__import__("json").dumps(
        {"intercept": float(model.intercept_[0]), "coef_on_scaled_features": coef}, indent=2), encoding="utf-8")
    return ["logreg_model.joblib", "scaler.joblib", "coefficients.json"]


if __name__ == "__main__":
    run_experiment(ModelSpec(
        name="Logistic Regression", family="Linear / probabilistic", folder=HERE, scaled=True, config=CONFIG,
        fit=fit, predict=predict, save=save,
        loss_name="weighted binary cross-entropy (log loss) + L2 penalty",
    ))
