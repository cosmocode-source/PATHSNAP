# PathSnap: data analysis and preprocessing report

Status report for the data-analysis and preprocessing phase. Every number below was produced by the code in this repository
(`data_analytics/PathSnap_EDA.ipynb`, `data_preprocessing/PathSnap_Preprocessing.ipynb`) and can be checked in
`cleaned_data/validation/`, `cleaned_data/data_quality_log.csv` and the executed notebooks. Partitions used: **train (80 trips)** and
**student_test (10 trips)**. The evaluator partition was never opened.

*AI-assistance disclosure:* the preprocessing code, notebook and this report were produced with AI assistance (Claude Code) and were
executed by the team's machine; the team must read, understand and be able to explain them. The EDA notebook is the team lead's work.

---

## 1. Where we are

| Stage | Status |
|---|---|
| Raw data exploration (EDA, team lead) | Done, pushed |
| Loader, projection, per-point flags, drop rule (4.1-4.2) | Done, pushed |
| Route-constrained labelling and validation (4.3) | Done, pushed |
| Candidate generation and miss-rate study (4.4) | Done, pushed |
| Feature engineering and feature decisions (4.5, 4.5b) | Done, pushed |
| Scaling, sample weights, folds, model-ready tables (4.6-4.7) | Done, pushed |
| Inference-safe pipeline + 8 assertion tests (4.8) | Done, all pass |
| Folium sanity maps (4.9), process README (4.10) | Done, pushed |
| Model training, Bayesian optimisation, HMM / Viterbi, API, UI, final report PDF | **Not started** |

Preprocessing is complete **for the candidate-ranking formulation**. Two things are intentionally not part of it: network distance between
consecutive candidates (belongs to the HMM transition model) and anything involving the evaluator partition.

## 2. Dataset

- 90 trips with files `.track` (lon, lat, t), `.nodes` (lon, lat), `.arcs` (from_node, to_node), `.route` (arc_id sequence).
- 208,417 GPS points in train, 19,160 in student-test. Per trip: 361-19,313 points (train), 262-3,958 (student).
- **Road networks are per trip** (299-155,003 nodes, up to 307,392 arcs). `arc_id` is local and is never a global class label.
- Trips are spread worldwide (25 distinct UTM zones, longitude -100 to 137), so each trip is projected to its own UTM zone (metres, not degrees).
- GPS rows, node ids, arc ids and route positions are four separate index spaces and are never joined by row number.

## 3. Data-quality findings

| Check | Result |
|---|---|
| Missing / invalid values, `(0,0)`, bad arc or route ids | None found (train and student) |
| Duplicate rows, duplicate or negative timestamps | None; the training drop rule removed **0 rows** (student: 0 by construction) |
| Sampling | About 1 Hz: 99.1% of steps are 1 s (train), 99.9% (student) |
| Long gaps (> 5 s) | 1,216 events (train), 44 (student); longest 153,064 s (train), 210 s (student) |
| Stationary / hive points | 1,427 / 410 (train); 80 / 43 (student) |
| Speed jumps (> 50 m/s) | 1,530 (train), 8 (student); **none** coincide with gap points |
| Isolated nodes | 0 in all 90 trips |
| Connectivity | Largest weak component holds a median 98% of nodes (worst 69%); largest strong component a median 96% (worst 64%) |
| Route continuity | 11 trips have a few non-chained consecutive arcs (worst 98.1%); the route always stays inside one component |
| Two-way arcs | 92.2% of arcs have an opposite-direction partner; about 2.02 arcs per node |
| Zero-length arcs | Up to 14 per trip (kept; guarded in the distance code) |

Observations that shaped decisions:
- **Speed jumps are mostly fast roads, not GPS errors.** 1,232 of the 1,530 train jumps are in two motorway trips (`00000032`, `00000001`, median speed about 41 m/s). The flag is kept for reporting only; it is not used to drop rows.
- **The reference route does not always match the GPS track.** 18 trips have a route shorter than 60% of the GPS path; the worst, `00000054`, has a 31 km route for a 170 km track, and its far fixes are tens to hundreds of metres from every route arc.

## 4. Preprocessing pipeline

### 4.1 Loader
Explicit schemas, per-trip UTM projection (round trip checked to under 1e-6 degrees), arc geometry / length / bearing, node in/out degree,
opposite-direction flag, STRtree over arcs. The loader never reads `.route`.

### 4.2 Point flags
Per point: stationary, in-hive, after-gap (with gap length), speed jump, robust speed outlier (per-trip median/MAD, real time step), and three
removable flags (exact duplicate, invalid coordinate, non-monotonic time). **Only training trips may drop rows, and only for the removable flags**;
student-test is never altered.

### 4.3 Labelling by route-constrained alignment
`.route` is a sequence, not a per-point label, so each fix is aligned with a monotone Viterbi search over **route positions** (so loops are handled).

- **The EDA's 1.3% nearest-vs-monotonic disagreement is explained.** It occurs in train trip `00000046`: the route is a loop that starts and ends at the same place, so the greedy monotonic rule jumps to the last arc at the first fix and can never go back. It is not an indexing or direction bug. The same effect appears in 8 of 80 train trips (agreement below 50%). The Viterbi aligner solves it; the greedy rule is not used anywhere.
- Order is correct for all 90 trips. Pooled route recovery: **96.5% (train), 96.6% (student)**. The worst trips (70-75%) have a step between fixes longer than their typical arc and 22-47% of steps longer than 1 s, so short arcs are physically skipped; this is a sampling effect.
- Label distance: train median 5.7 m (p95 146 m, inflated by uncovered trips; p95 14.4 m for a typical trip), student median 5.3 m (p95 16.9 m).
- Label quality (point-weighted): train clean 80.9% / ambiguous 6.7% / far 12.4%; student clean 81.9% / ambiguous 17.3% / far 0.8%.
- Agreement with the nearest route arc on clean points: 95.3% (train), 93.8% (student).
- Sensitivity of the aligner settings (sigma 5/10/15 m, advance cost 0.5/1/2) on 10 train trips changed the median label distance by under 0.15 m.

**Thresholds justified from training data:** `far_label_m = 30 m` is the p99 label distance (29.4 m) of the 24 well-covered train trips;
`ambiguity_margin_m = 5 m` is about the median label distance (4.8 m), one typical GPS error.

**Far points.** 25,796 train points (12.4%), 56 trips, but 3 trips hold 87% of them (`00000054` alone 67%); 91% are more than 50 m from any route arc (in `00000054`, `00000033` and `00000002`, over 99% of far fixes are more than 30 m from every arc of the
reference route), so the reference route does not describe where the vehicle actually drove in those stretches. They are not concentrated on gap, jump or hive points. Policy (team lead): far points affect labels, weights and reporting only,
never candidate generation; a far label is never moved to the nearest arc.

### 4.4 Candidate generation (route-free)
50 m primary radius; fewer than 2 candidates -> 100 m fallback; still none -> 5 nearest arcs; cap 50 (about the p99 of the uncapped count, 45.8).
Sources (train): primary 90.8%, fallback 8.6%, nearest-K 0.7%; student: 99.95% / 0.05% / 0%. Every point has at least one candidate.
Mean candidates per point: 9.3 (train), 10.8 (student).

**Candidate miss rate, train (208,417 points):**

| Scheme | Mean candidates | All | Clean | Ambiguous | Far |
|---|---|---|---|---|---|
| 15 m | 2.9 | 17.5% | 5.6% | 9.3% | 100% |
| 20 m | 3.6 | 14.3% | 2.1% | 3.5% | 100% |
| 30 m | 5.3 | 12.4% | 0% | 0% | 100% |
| 50 m | 9.0 | 11.1% | 0% | 0% | 89.9% |
| 100 m | 22.4 | 10.2% | 0% | 0% | 82.1% |
| **Final scheme** | 9.3 | 10.4% | 0% | 0% | 83.8% |

Caveat: clean and ambiguous labels are, by definition, at most 30 m from their point, so their 0% at radii of 30 m and above is not an independent result.
Far points cause the whole remaining miss rate (21,620 far points have no candidate that is their label arc; the cap on candidates per point costs about 400 far points compared with 100 m alone). Student-test miss rate (descriptive only): 0.15% overall,
clean 0%, ambiguous 0%, far 19.5% (29 of 149 far points). Decision: keep 50 m / 100 m fallback / K = 5 / cap 50.

### 4.5 Features (route-free), 26 candidates
Geometry (distance, projection fraction, distance to arc start / end, log arc length); context in the candidate set (rank, distance minus nearest, number of candidates, nearest flag);
direction (arc bearing, smoothed GPS heading with a validity flag, absolute difference, sine, cosine, reverse-twin flag); motion (speed from real dt, after-gap, gap length, stationary, hive);
temporal context (distance of previous / next fixes and a +/- 3 window to the same arc); topology (in/out degree, two-way flag).
No NaN or inf. Network distance between consecutive candidates is deliberately not computed (HMM stage).
The reverse-twin flag fires on 76.3% (train) / 74.9% (student) of candidate pairs, confirming that most roads appear as opposite-direction pairs.

### 4.5b Feature decisions (usable train rows: 186,797 points; 1,943,052 candidate pairs in total)
Class balance on usable rows: 10.3% positive (one per 9.7 rows); class weights from train 0.557 / 4.856.

| Decision | Features | Evidence |
|---|---|---|
| Drop (4) | `heading_diff_abs` | |Spearman| 1.00 with `heading_cos` (monotone function), equal AUC edge, so the positively oriented `heading_cos` is kept |
| | `is_two_way` | |Spearman| 0.999 with `is_reverse_twin` |
| | `gap_len_s` | |Spearman| 1.00 with `after_gap`; skew 229 |
| | `in_hive` | 99.8% one value, AUC edge below 0.02 |
| Transform log1p (2) | `dist_to_start_m`, `dist_to_end_m` | skew about 12, tail to 4.7 km |
| Keep (20) | everything else | includes weak single features that may help in interactions |

Strongest single features (AUC): `rank` 0.075, `dist_minus_nearest` 0.071, `dist_m` 0.113 (low = true arc), `heading_cos` 0.831, `log_arc_len` 0.759, `is_nearest` 0.747.
Mild collinearity (VIF above 10): `dist_m` 10.6, `mean_dist_win_m` 11.9 (matters for linear models only).

### 4.6 Scaling, weights, folds (training only)
- Raw table (trees) and scaled table (linear / MLP / distance): winsorise at train 1st / 99th percentile, log1p on 2 features, StandardScaler fitted on usable train rows, 5 binary flags pass through (22 selected features: 17 continuous + 5 binary).
- Sample weights (train points): clean 1.0 (168,566), ambiguous 0.5 (14,055), far 0.3 (25,796); student always 1.
- 21,620 train points (all far) have no positive row; they stay in the table with `fit_include = False` and must be filtered when fitting (count logged and tested). Student rows are never excluded.
- Folds: GroupKFold by `trip_id`, 5 folds, seed 42, 16 trips each; student rows have fold -1.

## 5. Outputs

Report figures (every caption states its partition): `cleaned_data/validation/report_figures/` with `CAPTIONS.md`, generated by `data_preprocessing/PathSnap_Report_Plots.ipynb`.


| Path | Content |
|---|---|
| `data_preprocessing/pathsnap_prep.py` | Library (loader, flags, labelling, candidates, features, `build_candidates_and_features`, `label_trip`, `scale_features`); every threshold in `CONFIG` |
| `data_preprocessing/PathSnap_Preprocessing.ipynb` | Executed notebook, Restart and Run All |
| `data_preprocessing/test_pipeline.py` | 8 assertion tests (all pass) |
| `data_preprocessing/preprocessing_config.json` | Frozen config: thresholds, winsor bounds, weights, feature lists |
| `cleaned_data/model_data/train_candidates.parquet` (81 MB), `student_candidates.parquet` | Raw model-ready candidate tables (keys, label, label quality, features, flags, `sample_weight`, `fit_include`, `fold`) |
| `cleaned_data/model_data/student_candidates_scaled.parquet`, `scaler.joblib` | Scaled student table and the train-fitted scaler |
| `cleaned_data/model_data/points_*.parquet`, `folds.csv` | One row per GPS point (miss rate reproducible); fold table |
| `cleaned_data/point_flags_*.parquet`, `point_labels_*.parquet` | Per-point flags and aligned labels |
| `cleaned_data/validation/` | Tables, figures and Folium maps behind every decision |
| `cleaned_data/data_quality_log.csv` | Log of measurements and decisions |

Not in git: the scaled **train** table (77.7 MB); rebuild with `pathsnap_prep.scale_features(raw_train, joblib.load('scaler.joblib'))` or by running the notebook.

## 6. Tests (all pass)
Inference output equals the stored batch features for a student trip; nothing on the prediction path takes a route argument or calls the labelling code (AST check); pipeline runs from track / nodes / arcs only;
index-space and projection round-trip checks; arc geometry lookup; every point (train and student) has at least one candidate and no student row is dropped; the exclusion rule is train-only and its logged count matches;
the scaled table is reproducible from the raw one.

## 7. Limitations and open issues
0. **Cross-machine determinism (fixed after a teammate's check):** two arcs of the same street at the same distance used to swap rank on a different machine because of float noise. Candidates are now ordered by distance rounded to 1 mm, then arc_id; the rank and flag features are asserted to match exactly across machines, other features to a small tolerance. Run `pytest test_pipeline.py` on the API laptop too.
1. **Reference-route coverage.** 12.4% of train labels are far, mostly from three trips. Models are trained with those points down-weighted or excluded; evaluation on student-test is on all points and clean-only (descriptive). Headline numbers may be pulled down on any trip whose reference route is incomplete.
2. **Student ambiguity is higher than train** (17.3% vs 6.7% ambiguous), probably because of more road-dense or looping routes; it affects how clean the student labels are, not the pipeline.
3. **Speed-jump flag** fires on legitimate motorway speeds; do not use it as an error filter.
4. **Heading features are redundant by construction** (`heading_diff_abs` and its cosine); only `heading_cos` is kept (the tie is broken by orientation, deterministically).
5. **Sampling effect:** trips with a step longer than the typical arc cannot have every route arc observed; labels cannot be recovered for skipped arcs, which is expected.
6. **Repository size:** `train_candidates.parquet` is 81 MB (above GitHub's 50 MB recommendation, below the 100 MB limit). If it becomes a problem, move it to Git LFS or split it by fold.
7. **The EDA outputs** (`outputs/`) are git-ignored; a committed copy (tables, figures, Folium map) regenerated from the EDA notebook is in `data_analytics/eda_outputs/`.
8. The EDA README's numbers come from the lead's pilot trips; where they differ from ours (for example 5.5 candidates per point at 50 m, pilot of 3 trips, vs 9.0 over all 80 trips), ours are measured on all 80.

## 8. What is next
1. **Model experiments:** train at least five models from different families on `train_candidates*.parquet` (filter `fit_include`, use `sample_weight`, group CV by `fold`), compare on student-test (Top-1 accuracy, macro F1, log loss, miss rate, route continuity, timing).
2. **Bayesian hyperparameter optimisation** for the chosen model with group-aware CV and a stored trial record.
3. **HMM / Viterbi decoding:** hidden states = candidate arcs, emissions = classifier probabilities, transitions from network connectivity and path distance (this is where network distance is computed), gap and missing-candidate handling; compare classifier-only vs classifier + HMM.
4. **FastAPI backend** (calls `build_candidates_and_features`, scores against server-side labels via `label_trip`, logs requests) and the **UI** on the second laptop.
5. **Technical report PDF** with the sections required by the guidelines, using this report as the data-and-preprocessing source, plus final Folium maps and screenshots.
