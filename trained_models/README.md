# PathSnap: model training and comparison (process only)

This folder trains five candidate-scoring models, evaluates them all in exactly the same way, and compares them.
This README describes the **process only**. All measured numbers are in each model's `results.json` and
`fold_results.csv` and in `comparison/model_comparison.csv`.

## Task formulation

For every GPS point, preprocessing generated candidate road arcs and 22 features per (point, candidate) pair. A model
outputs, for every pair, the probability that the candidate is the point's true (aligned) arc. The highest-scoring
candidate of a point is that point's predicted road arc. The later HMM / Viterbi stage will use the probabilities,
normalised within each point, as emissions.

## Environment

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r trained_models/requirements.txt
```

## How to run

**All five models in one terminal (demo).** From the repository root:

```powershell
.venv\Scripts\python trained_models\run_all_models.py
```

This runs model1 to model5 one after another with live output. When they finish, it prints one summary table of
all five models, read from their `results.json` files, and the model the selection rule picks. Options:
`--models 1 4` trains only those models; `--summary-only` prints the table of saved results without training.
Re-running overwrites each model's outputs.

**One model at a time.** Run each script from the `trained_models` folder. Each script runs CV, the final fit and the student evaluation, then
writes all of its outputs into its own folder.

```powershell
cd trained_models
..\.venv\Scripts\python model1\train_logreg.py
..\.venv\Scripts\python model2\train_knn.py
..\.venv\Scripts\python model3\train_random_forest.py
..\.venv\Scripts\python model4\train_lightgbm.py
..\.venv\Scripts\python model5\train_mlp.py
..\.venv\Scripts\jupyter nbconvert --to notebook --execute --inplace comparison\model_comparison.ipynb
```

The first run rebuilds `data_preprocessing/cleaned_data/model_data/train_candidates_scaled.parquet`. It is git-ignored
and rebuilt exactly from the raw train table and `scaler.joblib` with `pathsnap_prep.scale_features`.

## Folder layout

| Path | Content |
|---|---|
| `common/data_loading.py` | Loads the raw or scaled candidate tables, the 22 selected features (read from `preprocessing_config.json`, never hard-coded), labels, sample weights, `fit_include` and fold ids. Also reads `.arcs` files, **only** for the route-continuity metric. |
| `common/metrics.py` | Top-1, Macro F1, log loss (raw and normalised per point), candidate miss rate, route continuity, Brier score / ECE, per-trip Top-1. |
| `common/plotting.py` | Shared plots (fold bars, confusion matrices, calibration, real loss curves, per-trip bars, comparison bars). |
| `common/experiment.py` | The common harness: CV, final fit, student evaluation, timing and every output file. |
| `common/redraw_loss_curves.py` | Redraws the LightGBM / MLP loss-curve figures from the saved real histories (style changes only, no retraining). |
| `model1/` | Logistic Regression (class-weighted, L2), scaled table |
| `model2/` | KNN (distance-weighted, fitted on a seeded 150k-row subsample), scaled table |
| `model3/` | Random Forest (depth and leaf size capped), raw table |
| `model4/` | LightGBM (early stopping on a trip-grouped split, then refit), raw table |
| `model5/` | Shallow MLP in PyTorch (3 seeds, early stopping on a trip-grouped split), scaled table |
| `comparison/` | `model_comparison.csv`, `model_comparison.ipynb` and the cross-model plots |
| `requirements.txt` | Pinned packages for this stage |

Each `modelN/` folder holds:

| File | Content |
|---|---|
| `train_<model>.py` | Training script: configuration, fit / predict / save functions, commented |
| `config.json` | Hyperparameters, seeds, feature list, loss function, data rules of the run |
| model file(s) | `*.joblib`, `lightgbm_model.txt` or `mlp_seed*.pt`. Plus `scaler.joblib` for the scaled-table models. |
| `results.json` | All metrics: CV mean/std (all and clean points), pooled out-of-fold, train, student (all and clean), route continuity, calibration, confusion matrices, timing, train / CV / student gap |
| `fold_results.csv` | One row per fold (and seed) and subset (all / clean) |
| `per_trip_top1.csv` | Per-trip Top-1, miss rate, far-label share, route continuity, weak-trip flag (train CV out-of-fold and student) |
| `student_predictions.parquet` | Student-test candidate scores (for the comparison notebook and the HMM stage) |
| `training_log.txt` | Log of the run |
| `loss_history.json` | Only for LightGBM and the MLP: real per-iteration / per-epoch train and validation loss |
| `figures/` | CV fold metrics, confusion matrices, calibration curves, per-trip Top-1, loss curves (LightGBM / MLP only) |

## Data rules (applied identically to all five models)

- Features: the 22 `selected_features` of the frozen preprocessing config. `arc_id` is local to a trip and is never a
  feature or a class label. Rows are keyed by `trip_id`, `point_id`.
- The raw table goes to Random Forest and LightGBM. The scaled table goes to Logistic Regression, KNN and the MLP.
  Same rows, folds and features everywhere.
- Fitting uses only training rows with `fit_include == True` (points whose label arc is among their candidates),
  weighted by `sample_weight` (clean 1.0, ambiguous 0.5, far 0.3) where the algorithm supports it. sklearn KNN does
  not support sample weights, so it gets only the exclusion rule.
- Cross-validation uses the 5 stored GroupKFold folds by trip. Early stopping (LightGBM, MLP) uses a trip-grouped split
  *inside* the training folds, never the CV validation fold.
- Student-test is used for evaluation only. Nothing is fitted, selected or tuned on it. The evaluator partition is
  never read.
- The seed is 42 everywhere (the MLP uses 42, 43 and 44).

## Metric definitions

All metrics are unweighted: each point or row counts once. Sample weights only affect fitting.

- **Top-1 (observation level).** For each point (`trip_id`, `point_id`), take the highest-scoring candidate (ties go
  to the nearer candidate) and compare it with the aligned label arc.
  - `top1_all`: over all points. A point whose label is not among its candidates counts as a miss.
  - `top1_in_cands`: only over points whose label is among the candidates.
- **Macro F1 (our own definition, because the guidelines do not define it).** This is candidate level, with two
  classes. The top-scoring candidate of each point is predicted 1 and every other candidate 0. Macro F1 is the mean of
  F1(class 1) and F1(class 0).
- **Log loss.**
  - `logloss_raw`: binary cross-entropy of the raw candidate probabilities over all rows.
  - `logloss_norm`: probabilities are normalised to sum to 1 within each point, as the HMM will use them. The score is
    −log p(true arc), averaged over the points whose label is among the candidates.
- **Candidate miss rate.** The share of points whose label arc is not among the candidates. It is the same for every
  model and sets the ceiling on `top1_all`.
- **Route continuity (classifier only).** For each trip, take each point's top-1 arc in GPS order and collapse
  consecutive repeats. Continuity is the share of consecutive pairs where the first arc ends at the node where the next
  begins in that trip's network. It is reported pooled and as a per-trip mean. Reading `Dataset/<split>/<trip>/<trip>.arcs`
  for this metric is the one, agreed exception to "no Dataset/ for modelling".
- **Calibration.** Brier score and ECE (10 equal-width bins) of the raw probabilities, plus a reliability plot.
- **CV.** Mean and standard deviation over the 5 folds, on all points (headline) and on clean points only
  (`label_quality == clean`, diagnostic).
- **Student-test.** All points are the headline and clean-only points are a diagnostic.
- **Timing.** The final-fit wall time, and inference time per 1,000 GPS points (model prediction only, on the
  student partition, with features already built).

## Model selection rule

The primary metric is CV Top-1 on all points, with log loss as the tie-breaker. Fold stability, inference speed,
probability quality for the HMM (normalised log loss, calibration) and model size are also weighed. The justification
is in `comparison/model_comparison.ipynb`. A CV Top-1 difference smaller than the seed-to-seed spread measured for the MLP (std of its CV Top-1 over 3 seeds) counts as a tie.
