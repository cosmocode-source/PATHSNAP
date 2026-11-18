# PathSnap preprocessing: process description

This folder turns the raw trips into a model-ready candidate-ranking table. It describes **process only**; measured results are in the
notebook outputs (`PathSnap_Preprocessing.ipynb`), `../cleaned_data/validation/` and `../cleaned_data/data_quality_log.csv`.

## Files

| File | Purpose |
|---|---|
| `pathsnap_prep.py` | Library: loader, point flags, labelling, candidate generation, features, inference entry point. Every threshold is in `CONFIG` with its reason. |
| `PathSnap_Preprocessing.ipynb` | Runs the whole pipeline on train and student_test (Restart and Run All) and writes the outputs. |
| `test_pipeline.py` | Assertion tests (inference == batch, no route on the prediction path, index spaces, candidate coverage, exclusion rule). |
| `preprocessing_config.json` | Frozen configuration written by the notebook (thresholds, train-learned winsor bounds, weights, feature decisions). |
| `cache/` | Per-trip projected arrays (git-ignored, regenerated). |

Run from the notebook folder: `jupyter notebook PathSnap_Preprocessing.ipynb`, then Restart and Run All. Tests: `pytest test_pipeline.py`.

## Problem formulation

For every GPS point, generate candidate road arcs, compute features for every (point, candidate arc) pair, and label a pair 1 if the
candidate is the reference arc of that point. Road networks are **per trip**: `arc_id`, `node_id` and `route_pos` are local identifiers and
are never used as global class labels or joined by row number across files (`gps_row`, `node_id`, `arc_id`, `route_pos` are four separate
index spaces).

## Data used and leakage rules

- Training (80 trips): fitting, cross-validation, every learned statistic.
- Student-test (10 trips): never dropped or altered; weight 1; used to verify the pipeline and later to compare models. Nothing is tuned on it.
- Evaluator-test: never opened.
- All thresholds, clipping bounds, scalers and transforms are learned from **training data only**, saved to the config and applied unchanged
  to student-test (and later to evaluator requests).
- Rows of `.nodes`, `.arcs` and `.route` are never deleted or reordered.

## Pipeline

1. **Loader.** Reads `.track`, `.nodes`, `.arcs` with explicit schemas, projects to a per-trip UTM zone chosen from the trip centroid
   (metres, not degrees), derives arc geometry, length, bearing, node degrees, and the presence of an opposite-direction arc. Builds a spatial
   index (STRtree) over the arcs. The route is **not** read here.
2. **Point flags.** Per point: stationary, GPS hive, point after a long gap (with gap length), speed jump, robust speed outlier (per-trip
   median / MAD, uses the real time step), and three removable flags (exact duplicate, invalid coordinate, non-monotonic timestamp). Only
   training trips may drop rows, and only for the three removable flags; everything else is kept and flagged.
3. **Labelling (training and server-side scoring only).** `.route` is a sequence of arcs, not a per-point label, so each fix is aligned to the
   route with a monotone Viterbi search. States are route positions (so loops that revisit an arc are different states); emission is a capped
   Gaussian on the point-to-arc distance; transitions allow staying, advancing and a small forward skip, with larger skips allowed as the time
   step grows or after a long gap; going backwards is not allowed. Each label gets its distance and a quality tag (clean / ambiguous / far).
   Nearest-arc labelling and a greedy monotonic rule were rejected: they fail on loops and where the reference route does not cover the track.
4. **Candidate generation (route-free).** Primary radius query; if the point has too few candidates, a larger fallback radius; if still
   none, the K nearest arcs. The number of candidates per point is capped (nearest kept) and the source (primary / fallback / nearest-K) is
   recorded. Candidate generation never sees the label or its distance. The labels are used afterwards only to *measure* the candidate miss rate.
5. **Features (route-free).** Per (point, candidate):
   - geometry: perpendicular distance, projection fraction, distance to the arc start and end, log arc length (an exact geometry of the match);
   - context inside the point's candidate set: rank by distance, distance minus the nearest candidate's distance, number of candidates, nearest flag;
   - direction: arc bearing and the GPS heading from a smoothed window (invalid and neutral when the displacement is too small or a gap is inside the window),
     absolute heading-bearing difference with sine and cosine, and a reverse-twin flag (the opposite-direction arc is also a candidate);
   - motion: speed from the real time step, after-gap flag and gap length, stationary, hive;
   - temporal context: distance of the previous and next fixes to the same arc, and the mean distance of a +/- window of fixes to it;
   - topology: arc in-degree and out-degree, one-way versus two-way.
   Network distance between consecutive candidates is deliberately **not** computed here; it belongs to the HMM transition model.
6. **Feature selection.** On usable training rows only: skew, kurtosis, share of zeros and missing values, per-feature AUC, Spearman
   correlation, VIF, near-constant check. Rule-based keep / transform / drop decisions are written to `feature_decisions.csv` and the config.
7. **Scaling and weighting (training only).** Two tables: raw (trees) and scaled (linear, MLP, distance models). Scaled = winsorise with
   train-learned percentile bounds, `log1p` for the features decided as skewed, StandardScaler fitted on usable train rows, binary flags
   passed through. Sample weights down-weight far and ambiguous labels in training; student rows always have weight 1. Class weights come from
   training only. Fixed GroupKFold fold ids by `trip_id` with a stored seed.

## Far-point policy

Far points (label too far from the GPS point, typically because the reference route does not cover the track) influence **labels, weights and
reporting only**. They never influence candidate generation. Training keeps them with a reduced weight; training rows whose label arc is not in
the candidate set are excluded from fitting (they stay in the table with `fit_include = False`, and the count is logged). A far label is never
moved to the nearest arc, because that would invent labels. Student-test rows are never dropped or altered.

## Inference-safe design

- `build_candidates_and_features(track, nodes, arcs, config)` has no route argument and never drops a row. The API calls it.
- `label_trip(trip, route, config)` is separate and used only for training labels and server-side scoring. Ground truth stays on the server.
- `test_pipeline.py` checks that the inference output equals the batch output, that nothing on the prediction path takes or opens a route, and that
  every point (including student-test) has at least one candidate.

## Outputs

`../cleaned_data/` (small items and validation, committed) and `../cleaned_data/model_data/`:

- `train_candidates.parquet`, `student_candidates.parquet`: raw features, keys (`trip_id`, `point_id`, `gps_row`, local `arc_id`), `label`,
  `label_quality`, flags, `sample_weight`, `fit_include`, `fold`.
- `train_candidates_scaled.parquet`, `student_candidates_scaled.parquet`: selected features, scaled.
- `points_train.parquet`, `points_student.parquet`: one row per GPS point (candidate count and source, whether the label arc is among the
  candidates, flags). The candidate miss rate is reproducible from these.
- `folds.csv`, `scaler.joblib`.
- `../preprocessing_config.json` is `preprocessing_config.json` in this folder; `../cleaned_data/data_quality_log.csv` records every decision.
- `../cleaned_data/validation/`: tables, figures and Folium maps behind each decision.

`train_candidates_scaled.parquet` is git-ignored because of its size; rebuild it from the raw train table with
`pathsnap_prep.scale_features(raw_df, joblib.load('scaler.joblib'))` (or by running the notebook). Any other table that is missing can be
regenerated by running the notebook.
