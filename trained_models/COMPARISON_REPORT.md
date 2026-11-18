# Final model and route decoder comparison

## Executive summary

1. The final classifier is the untuned LightGBM.
2. The original training-only decision found that the tuned gain was within noise.
3. The final route decoder is the untuned-model HMM with `trained_models/hmm/config.json`.
4. On student-test, pooled all-point top-1 was 86.36% for the classifier and 84.98% with HMM.
5. Student mean-trip continuity was 75.68% and 85.68%, respectively.
6. Student mean-trip arc F1 against the reference route was 0.7305 and 0.8367.
7. The HMM improves continuity but lowers pooled point accuracy.
8. The paired mean-trip student top-1 difference has a confidence interval that includes zero.
9. The tuned classifier is shown for comparison; its HMM was not run.
10. The student set has only ten trips; its results did not select the model or decoder.

## How we got here

The tuning study selected an untuned final classifier because its prespecified training-only decision required a positive bootstrap interval for the in-candidate CV gain and no worse normalised log loss. The recorded gain interval was [−0.00149, +0.00269], so it included zero. The existing final-model directory already contains the untuned model. The HMM was selected using out-of-fold untuned-model probabilities and training trips; it was then evaluated on all training trips and the held-out student trips. No tuned-model HMM or new decoder tuning was run here.

## Master comparison

CV columns are means and sample standard deviations across the five stored folds. Student top-1, macro F1 and log loss are pooled over points or candidate rows as appropriate; continuity, route F1 and mismatch are mean trip values. Route metrics are **vs reference route**. Times are measured in the original runs, not estimates of end-to-end latency.

| System | CV top-1 all | CV top-1 in candidates | Student top-1 all / in candidates / clean | Student candidate macro F1 | Student normalised log loss | Student continuity | Student reference route F1 | Student length mismatch | Time per 1,000 points | Final fit time |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 Untuned classifier | 0.7733 ± 0.1291 | 0.8514 ± 0.0361 | 0.8636 / 0.8649 / 0.8848 | 0.9252 | 0.4018 | 0.7568 | 0.7305 | 0.8677 | 3.395 ms model | 12.53 s classifier |
| S2 Tuned classifier | 0.7737 ± 0.1289 | 0.8520 ± 0.0375 | 0.8635 / 0.8648 / 0.8848 | 0.9252 | 0.3829 | 0.7672 | 0.7380 | 0.8070 | 16.609 ms model | 31.05 s classifier |
| S3 Untuned + HMM **FINAL** | 0.7379 ± 0.1348 | 0.8105 ± 0.0419 | 0.8498 / 0.8511 / 0.8593 | 0.9176 | 5.1435 (hard) | 0.8568 | 0.8367 | 0.2485 | 714.36 ms decode only | 12.53 s classifier |
| S4 Tuned + HMM | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |

S4 was **NOT RUN: the tuned model did not beat the untuned model on cross-validation under the original decision rule**. The HMM returns a hard path, not posterior probabilities; a genuine posterior log loss is **NOT DEFINED**. The value marked “hard” is reconstructed for one-hot path decisions using the common metric’s probability clipping and normalisation. It is dominated by the clipping floor and is not comparable to a calibrated classifier loss. Candidate macro F1 is computed from the resulting one-arc-per-point confusion counts. The HMM decode timing excludes classifier inference and preprocessing. The reference length mismatch uses the multiset symmetric difference of arc lengths divided by reference-route length; it ignores order and can exceed one.

## Detailed pooled results

| Partition / system | All top-1 | In-candidate top-1 | Clean top-1 | Candidate miss | Candidate macro F1 | Normalised log loss | Continuity (mean trip) | Broken transitions | Reference precision / recall / F1 (mean trip) | Reference length mismatch (mean trip) |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| train S1 | 0.7673 | 0.8562 | 0.8812 | 0.1037 | 0.8939 | 0.4326 | 0.7405 | 14608 | 0.6390 / 0.8860 / 0.7062 | 1.7463 |
| train S2 | 0.7678 | 0.8567 | 0.8820 | 0.1037 | 0.8941 | 0.4264 | 0.7506 | 13789 | 0.6465 / 0.8884 / 0.7127 | 1.6865 |
| train S3 | 0.7313 | 0.8160 | 0.8375 | 0.1037 | 0.8727 | 6.3556 (hard) | 0.8443 | 6553 | 0.7739 / 0.9049 / 0.8111 | 0.3974 |
| student S1 | 0.8636 | 0.8649 | 0.8848 | 0.0015 | 0.9252 | 0.4018 | 0.7568 | 1334 | 0.6570 / 0.8870 / 0.7305 | 0.8677 |
| student S2 | 0.8635 | 0.8648 | 0.8848 | 0.0015 | 0.9252 | 0.3829 | 0.7672 | 1233 | 0.6674 / 0.8890 / 0.7380 | 0.8070 |
| student S3 | 0.8498 | 0.8511 | 0.8593 | 0.0015 | 0.9176 | 5.1435 (hard) | 0.8568 | 656 | 0.8136 / 0.8997 / 0.8367 | 0.2485 |

train S3 changed 35344 points, recorded 189 breaks and relaxed 1216 gap steps. Decode time was 135.99 s total; the largest trip 00000054 had 19313 points and took 7.960 s.

student S3 changed 2246 points, recorded 0 breaks and relaxed 44 gap steps. Decode time was 13.69 s total; the largest trip 00000056 had 3958 points and took 2.391 s.

## What each step added

The table uses a paired bootstrap of the **mean per-trip** top-1 difference; it is not the original decision-rule estimate, which used a different pooled estimand. A positive mean-trip interval in this descriptive table does not reopen the frozen classifier decision.

| Partition | Comparison | Mean trip top-1 difference | 95% paired trip bootstrap CI | Verdict |
|---|---|---:|---|---|
| train | Tuning (S2 − S1) | +0.0018 | [+0.0002, +0.0034] | real gain |
| train | HMM on untuned (S3 − S1) | -0.0226 | [-0.0414, -0.0056] | worse |
| train | HMM on tuned (S4 − S2) | NOT RUN | NOT RUN | NOT RUN |
| train | Tuning + HMM (S4 − S1) | NOT RUN | NOT RUN | NOT RUN |
| student | Tuning (S2 − S1) | +0.0018 | [-0.0015, +0.0049] | within noise |
| student | HMM on untuned (S3 − S1) | +0.0014 | [-0.0272, +0.0204] | within noise |
| student | HMM on tuned (S4 − S2) | NOT RUN | NOT RUN | NOT RUN |
| student | Tuning + HMM (S4 − S1) | NOT RUN | NOT RUN | NOT RUN |

### Additional paired bootstrap intervals

| Partition | Comparison | Metric | Mean trip difference | 95% CI |
|---|---|---|---:|---|
| train | S2 − S1 | top1_in_candidates | +0.0019 | [+0.0003, +0.0036] |
| train | S2 − S1 | continuity | +0.0101 | [+0.0062, +0.0139] |
| train | S3 − S1 | top1_in_candidates | -0.0242 | [-0.0438, -0.0056] |
| train | S3 − S1 | continuity | +0.1038 | [+0.0828, +0.1281] |
| student | S2 − S1 | top1_in_candidates | +0.0018 | [-0.0013, +0.0048] |
| student | S2 − S1 | continuity | +0.0104 | [-0.0022, +0.0202] |
| student | S3 − S1 | top1_in_candidates | +0.0014 | [-0.0275, +0.0207] |
| student | S3 − S1 | continuity | +0.1000 | [+0.0612, +0.1359] |

For continuity, the paired mean-trip HMM difference and its bootstrap interval are:
- train: +0.1038, 95% CI [+0.0828, +0.1281] (real gain).
- student: +0.1000, 95% CI [+0.0612, +0.1359] (real gain).

## Decoder settings

| Parameter | Untuned final config | Tuned-model config |
|---|---:|---|
| alpha | 2 | NOT RUN |
| beta | 30 m | NOT RUN |
| U-turn penalty | 2 | NOT RUN |
| gap beta factor | 3 | NOT RUN |
| minimum cutoff | 100 m | NOT RUN |
| top-k | 15 | NOT RUN |

These settings won the final seeded training-only decoder search on the selected trip subset. Earlier exploratory runs are preserved and excluded. There is no tuned-model config because Branch B applies.

## Where the HMM helped and did not

Positive and negative differences below use all-point top-1 and are not causal diagnoses. The full trip table and weak-component diagnostics are linked below.

train: 36 trips improved, 44 were equal or worse. Largest gains: 00000026 +0.1413, 00000004 +0.1410, 00000022 +0.1224.
Equal or worse: 00000082 -0.3015, 00000079 -0.2903, 00000083 -0.2455, 00000010 -0.2398, 00000003 -0.1998, 00000080 -0.1873, 00000051 -0.1748, 00000081 -0.1210, 00000072 -0.1022, 00000031 -0.0966, 00000033 -0.0883, 00000036 -0.0782, 00000021 -0.0730, 00000075 -0.0725, 00000063 -0.0694, 00000070 -0.0679, 00000062 -0.0657, 00000048 -0.0645, 00000002 -0.0636, 00000034 -0.0551, 00000093 -0.0512, 00000086 -0.0490, 00000008 -0.0453, 00000084 -0.0420, 00000035 -0.0396, 00000061 -0.0385, 00000095 -0.0361, 00000038 -0.0334, 00000088 -0.0308, 00000069 -0.0203, 00000046 -0.0178, 00000068 -0.0174, 00000016 -0.0138, 00000066 -0.0111, 00000042 -0.0107, 00000027 -0.0101, 00000011 -0.0081, 00000092 -0.0073, 00000025 -0.0025, 00000097 -0.0023, 00000057 -0.0018, 00000064 -0.0014, 00000054 -0.0005, 00000085 +0.0000.

student: 7 trips improved, 3 were equal or worse. Largest gains: 00000009 +0.0324, 00000076 +0.0303, 00000052 +0.0229.
Equal or worse: 00000077 -0.1117, 00000040 -0.0160, 00000091 -0.0023.

For each non-improving trip, `trained_models/hmm/results/REPORT_ANSWERS.md` records observed gap, far-label, stationary, sparse-step, repeated-arc and road-component indicators where applicable. These are possible associations; aggregate indicators cannot prove a cause. Far-heavy trips can have reference-route coverage gaps. Sparse observations can skip arcs, so the collapsed per-point sequence can omit traversed arcs.

## Caveats

- Candidate misses set the shared all-point ceiling. The observed miss rates were 10.37% on train and 0.15% on student-test.
- Ambiguous labels are more common in student-test (17.30%) than train (6.74%); far labels are concentrated in particular training trips.
- Only LightGBM received this tuning; the other four model families were not tuned in the same way.
- Student-test contains ten trips. The trip bootstrap is correspondingly uncertain, and continuity and point accuracy answer different questions.
- The study export has 47 COMPLETE and three FAIL records, while the tuning narrative describes 46 complete evaluations and 41 unique configurations. This record-count discrepancy has not been resolved; the previously recorded decision was not recomputed.
- The final and model4 untuned model files differ in line endings; their content matches after CRLF normalisation, which the backend model loader already performs.
- The decoder requires a candidate at the first GPS point; it cannot carry an arc before one has been selected.

## Figures

- [train: Top-1 across systems](comparison/figures/final_train_top1_all.png)
- [train: Continuity across systems](comparison/figures/final_train_continuity.png)
- [train: Paired trip differences](comparison/figures/final_train_paired.png)
- [train: HMM differences and trip characteristics](comparison/figures/final_train_characteristics.png)
- [train: Decode time versus trip size](comparison/figures/final_train_runtime.png)
- [student: Top-1 across systems](comparison/figures/final_student_top1_all.png)
- [student: Continuity across systems](comparison/figures/final_student_continuity.png)
- [student: Paired trip differences](comparison/figures/final_student_paired.png)
- [student: HMM differences and trip characteristics](comparison/figures/final_student_characteristics.png)
- [student: Decode time versus trip size](comparison/figures/final_student_runtime.png)

[Per-trip comparison](comparison/final_comparison/comparison_per_trip.csv) · [Comparison summary and bootstrap intervals](comparison/final_comparison/comparison_summary.json) · [Earlier HMM trip diagnostics](hmm/results/REPORT_ANSWERS.md)

## Bayesian optimisation questions

1. **What was searched?** Learning rate, leaves, depth, minimum leaf size, feature and bagging fractions, L1/L2 and split gain.
2. **What ranges and why?** The search bracketed the untuned settings with practical values; full ranges are recorded in the tuning report.
3. **What was the objective?** Mean in-candidate top-1 across the stored trip-grouped training folds, with normalised log loss used for near ties.
4. **How many trials?** The narrative reports 41 unique completed configurations and 46 completed evaluations; the trial-export discrepancy is noted above.
5. **What were the best values?** Trial 23 used learning rate 0.0239, 232 leaves, depth 12, minimum leaf size 205, feature fraction 0.543, bagging fraction 0.582, L1 0.002778, L2 0.2308 and split gain 0.3347. These were not the deployed settings.
6. **Did tuning win?** No: the original paired interval included zero, so the untuned classifier remained final.

## HMM questions

1. **Hidden states:** directed candidate road arcs at each GPS observation.
2. **Emissions:** classifier probabilities normalised within each point, floored, logged and weighted.
3. **Transitions:** the discrepancy between directed road distance and GPS displacement, plus a reverse-turn penalty.
4. **Missing candidates and gaps:** carry the previous arc when possible; use real elapsed time and a relaxed transition scale for a gap.
5. **Disconnected segments:** if every candidate pair is unreachable, record a break and restart; otherwise unreachable pairs get a very low score.
6. **Did it improve the route?** Continuity and reference-route F1 increased, but pooled point accuracy fell; the student mean-trip top-1 difference was within the bootstrap interval around zero.

## API hand-off

Set `MODEL_DIR` to the absolute path of `trained_models/final_model` (or `../trained_models/final_model` relative to the backend directory). Set `HMM_MODULE=trained_models.hmm.api_adapter:decode_trip` and put the repository root on `PYTHONPATH` before starting the backend. The adapter loads `trained_models/hmm/config.json`, the final untuned decoder configuration, and converts the backend network dictionary to the representation used during evaluation. It does not load reference routes or labels. No backend or frontend files were changed.

The previous direct `trained_models.hmm.hmm_decoder:decode_trip` setting alone is not compatible with the backend network dictionary, and the backend defaults do not load the frozen decoder settings. Use the adapter above. Re-check this hand-off with a live API smoke test in the deployment environment; this report verifies the adapter contract with unit tests but does not claim a full live-server test.
