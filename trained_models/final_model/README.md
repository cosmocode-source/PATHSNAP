# Final model for the HMM and the API

Two LightGBM variants are offered. Both take the **raw** candidate table with the 22 features in `feature_list.json` and output, for every
(GPS point, candidate arc) row, the probability that the candidate is the true arc. Normalise the probabilities within each point to use them
as HMM emissions.

| Variant | Files | Role |
|---|---|---|
| **Untuned** (primary) | `lightgbm_model.txt`, `config.json` | Chosen by the tuning decision rule: the tuned model's CV gain was inside the noise |
| **Tuned** (alternative) | `variants/tuned/` | Offered to the HMM stage so both can be compared on route-level results |

`variants/tuned/` also holds `oof_predictions.parquet` (5-fold out-of-fold probabilities on the 80 training trips, same columns as
`trained_models/hmm/results/oof_predictions.parquet`) and `student_predictions.parquet` (column `p_seed42`). `make_tuned_variant.py` rebuilds them.

| | Untuned | Tuned |
|---|---|---|
| CV Top-1, all points | 0.7733 | 0.7737 |
| CV normalised log loss | 0.4461 | 0.4408 |
| Student Top-1 | 0.8636 | 0.8635 |
| Classifier-only route continuity (student) | 0.739 | 0.755 |
| Inference per 1,000 points | 3.4 ms | 16.6 ms |
| Model size | 0.55 MB | 5.6 MB |

Choose between them on the HMM's own results using training (out-of-fold) and student-test data only. Never use the evaluator trips.
Full tuning record: `trained_models/model4/tuning/`.
