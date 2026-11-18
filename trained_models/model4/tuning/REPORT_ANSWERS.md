# Bayesian hyperparameter optimisation: report answers

All numbers come from `tune_lightgbm.ipynb` (this folder): `optuna_trials.csv`, `results.json`, `untuned_vs_tuned.csv`.

**1. Which hyperparameters were searched?** Learning rate, number of leaves, maximum depth, minimum rows per leaf (`min_child_samples`), feature
fraction, bagging fraction, L1 and L2 regularisation, and minimum split gain. Class weighting and sample weights were kept exactly as in the untuned model.

**2. What ranges, and why?** Learning rate 0.01-0.2 (log), leaves 15-255 (log), depth -1 or 4-12, rows per leaf 10-300, feature and bagging
fraction 0.5-1.0, L1 and L2 0.001-10 (log), split gain 0-1. Each range brackets the untuned value and stays where training is practical: below 0.01
the learning rate needs more than the 2,000-round cap, below 0.5 a tree would often miss the main distance and heading features. Full reasons are in the notebook.

**3. What objective?** The mean 5-fold cross-validation Top-1 accuracy on points whose true arc is among the candidates, on the 80 training trips
with the stored trip-grouped folds. Points without their true arc among the candidates are wrong for every configuration, so leaving them out ranks
configurations the same way with less noise. Ties within one fold standard error go to the lower normalised log loss. Early stopping is nested inside
each training fold (20% of the training trips held out); the validation fold and student-test are never used to stop or choose.

**4. How many trials?** 41 unique complete configurations (46 complete evaluations, 6 of them accidental repeats) with Optuna's TPE sampler (seed 42), trial 0 being the untuned configuration. Total compute
76 minutes (112 s per trial on average).

**5. Best values?** Trial 23: learning rate 0.0239, leaves 232, max depth 12,
min rows per leaf 205, feature fraction 0.543, bagging fraction 0.582,
L1 0.002778, L2 0.2308, min split gain 0.3347. CV Top-1 (in-candidates) 0.8575;
the untuned configuration scored 0.8552 in the same search. The top 5 trials span 0.0007, against a fold standard error of 0.0149.

**6. Did the tuned model beat the untuned one?** With the full procedure (refit) on the same folds: CV Top-1 on all points 0.7733 untuned
vs 0.7737 tuned; in-candidates 0.8514 vs 0.8520; normalised log loss
0.4461 vs 0.4408. Paired over the 80 training trips, the pooled gain is +0.0005 with a 95% bootstrap
interval of [-0.0015, +0.0027]. This does not meet the rule (gain beyond the noise AND log loss not worse), so the untuned model is kept.
On student-test (used once, after freezing): Top-1 0.8636 untuned vs 0.8635 tuned.

## Untuned vs tuned

| Metric | Untuned | Tuned | Tuned - untuned |
|---|---|---|---|
| CV Top-1 all (mean) | 0.7733 | 0.7737 | +0.0004 |
| CV Top-1 all (std) | 0.1291 | 0.1289 | -0.0002 |
| CV Top-1 in-cands (mean) | 0.8514 | 0.8520 | +0.0005 |
| CV Top-1 in-cands (std) | 0.0361 | 0.0375 | +0.0013 |
| CV Macro F1 | 0.8926 | 0.8929 | +0.0002 |
| CV log loss raw | 0.0870 | 0.0856 | -0.0014 |
| CV log loss norm | 0.4461 | 0.4408 | -0.0053 |
| CV ECE | 0.0114 | 0.0101 | -0.0013 |
| Student Top-1 all | 0.8636 | 0.8635 | -0.0001 |
| Student Top-1 clean | 0.8848 | 0.8848 | +0.0000 |
| Student Macro F1 | 0.9252 | 0.9252 | -0.0000 |
| Student log loss raw | 0.0679 | 0.0656 | -0.0023 |
| Student log loss norm | 0.4018 | 0.3829 | -0.0189 |
| Route continuity (CV OOF) | 0.7285 | 0.7389 | +0.0104 |
| Route continuity (student) | 0.7389 | 0.7549 | +0.0160 |
| Train Top-1 all | 0.7971 | 0.8156 | +0.0185 |
| Final fit seconds | 12.5 | 31.1 | +18.5200 |
| Inference ms / 1,000 points | 3.3950 | 16.6 | +13.2140 |
| Model size MB | 0.5545 | 5.5607 | +5.0062 |
| Trees | 79.0 | 219.0 | +140.0000 |

Final model for the HMM and the API: **untuned** LightGBM, saved in `trained_models/final_model/`.
