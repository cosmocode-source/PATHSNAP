"""Load the model-ready tables written by the preprocessing stage (data_preprocessing/cleaned_data/).

What this module does
---------------------
* Reads the frozen preprocessing config and takes the feature list from it (``selected_features``, 22 features).
  The list is never hard-coded here, so the models always use exactly what preprocessing decided.
* Loads the raw candidate table (for tree models) or the scaled candidate table (for linear / KNN / MLP models).
* The scaled TRAIN table is not stored in git (77 MB). It is rebuilt exactly from the raw train table and the
  train-fitted ``scaler.joblib`` with ``pathsnap_prep.scale_features`` (the same function preprocessing used), then cached
  locally (git-ignored path).
* Never reads Dataset/ for modelling. The only exception is ``load_arc_endpoints``, used by the route-continuity METRIC
  (team decision: road connectivity is needed to say whether two predicted arcs connect; it is never a feature or a label).
* Never touches the evaluator partition.

Row keys: (trip_id, point_id). arc_id is LOCAL to a trip and is only carried along as an identifier, never as a feature.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---- fixed paths (relative to the repository root, found from this file's location) ---------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
PREP_DIR = REPO_ROOT / "data_preprocessing" / "data_preprocessing"          # pathsnap_prep.py + frozen config
CLEAN_DIR = REPO_ROOT / "data_preprocessing" / "cleaned_data"
MODEL_DATA = CLEAN_DIR / "model_data"                                       # candidate tables, points, scaler, folds
DATASET_DIR = REPO_ROOT / "Dataset"                                         # ONLY for .arcs in the continuity metric
TRAINED_DIR = REPO_ROOT / "trained_models"

SEED = 42                       # one seed for everything random in the modelling stage
SPLITS = ("train", "student")   # file-name tags used by preprocessing; the evaluator partition is never listed
DATASET_SPLIT_DIR = {"train": "train", "student": "student_test"}           # folder names under Dataset/

# Columns carried with every candidate row besides the features (identifiers + label + training bookkeeping).
KEY_COLS = ["trip_id", "point_id", "gps_row", "arc_id"]
META_COLS = ["label", "label_quality", "sample_weight", "fit_include", "fold"]


def load_prep_config() -> dict:
    """Frozen preprocessing configuration (thresholds, winsor bounds, weights, feature decisions)."""
    with open(PREP_DIR / "preprocessing_config.json", encoding="utf-8") as f:
        return json.load(f)


def feature_list() -> list[str]:
    """The selected features, in the order stored by preprocessing (22 features)."""
    return list(load_prep_config()["selected_features"])


def _scale_with_prep(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the train-fitted scaling bundle with the preprocessing library's own function (no re-implementation)."""
    if str(PREP_DIR) not in sys.path:
        sys.path.insert(0, str(PREP_DIR))
    import pathsnap_prep  # noqa: E402  (imported lazily: only needed when the scaled train table must be rebuilt)

    bundle = joblib.load(MODEL_DATA / "scaler.joblib")
    return pathsnap_prep.scale_features(raw, bundle)


def _raw_table(split: str) -> pd.DataFrame:
    assert split in SPLITS, f"unknown split {split!r}"
    cols = KEY_COLS + META_COLS + feature_list()
    return pd.read_parquet(MODEL_DATA / f"{split}_candidates.parquet", columns=cols)


def _scaled_table(split: str) -> pd.DataFrame:
    """Scaled candidate table. Student is stored; train is rebuilt from raw + scaler.joblib and cached locally."""
    path = MODEL_DATA / f"{split}_candidates_scaled.parquet"
    cols = KEY_COLS + META_COLS + feature_list()
    if path.exists():
        return pd.read_parquet(path, columns=cols)
    assert split == "train", "only the train scaled table is expected to be rebuilt"
    raw = pd.read_parquet(MODEL_DATA / "train_candidates.parquet")
    scaled = _scale_with_prep(raw)
    out = pd.concat([raw[KEY_COLS + META_COLS].reset_index(drop=True), scaled.reset_index(drop=True)], axis=1)
    out.to_parquet(path, index=False)          # git-ignored cache (see .gitignore); exact rebuild every time
    return out[cols]


@dataclass
class CandidateData:
    """One split's candidate rows in a model-friendly form. Row i of every array is the same (point, candidate) pair."""
    split: str
    keys: pd.DataFrame          # trip_id, point_id, gps_row, arc_id (arc_id local, identifier only)
    X: np.ndarray               # (n, 22) float32 features in feature_list() order
    y: np.ndarray               # (n,) int8 1 = candidate is the point's aligned label arc
    w: np.ndarray               # (n,) float32 sample weight (train: clean 1 / ambiguous 0.5 / far 0.3; student: 1)
    fit_include: np.ndarray     # (n,) bool  False = the point's label arc is not among its candidates (never fit on)
    fold: np.ndarray            # (n,) int8  GroupKFold id by trip (0-4 train, -1 student)
    label_quality: np.ndarray   # (n,) str   clean / ambiguous / far (point-level, repeated on each candidate row)
    features: list[str]

    @property
    def n(self) -> int:
        return len(self.y)


def load_candidates(split: str, scaled: bool) -> CandidateData:
    """Load a split. ``scaled=False`` -> raw table (Random Forest, LightGBM); ``True`` -> scaled (LR, KNN, MLP).

    Rows are sorted by (trip_id, gps_row, rank-order as stored) so every model sees the same row order.
    """
    df = _scaled_table(split) if scaled else _raw_table(split)
    df = df.sort_values(["trip_id", "gps_row"], kind="stable").reset_index(drop=True)
    feats = feature_list()
    data = CandidateData(
        split=split,
        keys=df[KEY_COLS].copy(),
        X=df[feats].to_numpy(np.float32),
        y=df["label"].to_numpy(np.int8),
        w=df["sample_weight"].to_numpy(np.float32),
        fit_include=df["fit_include"].to_numpy(bool),
        fold=df["fold"].to_numpy(np.int8),
        label_quality=df["label_quality"].to_numpy(),
        features=feats,
    )
    # Guards that the preprocessing contract still holds (cheap, catches a stale or mismatched table early).
    assert np.isfinite(data.X).all(), "NaN/inf in features"
    if split == "student":
        assert (data.fold == -1).all() and data.fit_include.all() and (data.w == 1).all(), "student rows must be untouched"
    return data


def fit_mask(data: CandidateData, train_folds) -> np.ndarray:
    """Rows a model may be FITTED on: the given training folds AND fit_include == True (exclusion rule)."""
    return np.isin(data.fold, list(train_folds)) & data.fit_include


def load_points(split: str) -> pd.DataFrame:
    """One row per GPS point: label arc, label quality, label_in_candidates (for miss rate / Top-1 denominators)."""
    cols = ["trip_id", "point_id", "gps_row", "label_arc_id", "label_quality", "label_in_candidates"]
    return pd.read_parquet(MODEL_DATA / f"points_{split}.parquet", columns=cols)


def load_arc_endpoints(split: str, trip_id: str) -> np.ndarray:
    """(A, 2) from_node, to_node of every arc of one trip; row index = local arc_id.

    Used ONLY by the route-continuity metric (are two consecutive predicted arcs connected?). Never a feature.
    """
    path = DATASET_DIR / DATASET_SPLIT_DIR[split] / trip_id / f"{trip_id}.arcs"
    return pd.read_csv(path, sep=r"\s+", header=None, names=["u", "v"], dtype="int64").to_numpy()
