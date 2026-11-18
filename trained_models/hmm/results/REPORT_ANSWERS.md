# Report answers

1. **Hidden states.** Each state is a candidate directed road arc for one GPS point. The result chooses one arc for every point.

2. **Emissions.** The classifier scores candidate arcs; the scores are normalised within each point. The decoder takes the log of each probability after flooring it and multiplies it by alpha.

3. **Transitions.** The decoder compares directed network distance between projected candidate positions with straight-line GPS distance. It penalises disagreement and reverse-direction turns. Short backward motion on the same arc is treated as stationary jitter.

4. **Missing candidates and gaps.** A point without candidates carries the previous state and is counted. A first point without candidates cannot be carried and raises an error. Gaps use the actual elapsed time, a larger search bound, and a relaxed distance penalty.

5. **Disconnected observations.** If no candidate pair is reachable, the decoder records a break and restarts from the next emissions. If only some pairs are unreachable, those pairs receive a very low score.

6. **Did it improve the route?** The results below compare classifier-only and HMM predictions. Training uses out-of-fold scores; student-test was reserved until parameters were frozen. Confidence intervals resample whole trips, so they describe uncertainty across trips.

## student
- top1_all: classifier 0.8693, HMM 0.8707, difference +0.0014; 95% paired trip bootstrap interval [-0.0265, +0.0211] (interval includes zero).
  Pooled over points: classifier 0.8636, HMM 0.8498.
- top1_in_candidates: classifier 0.8700, HMM 0.8715, difference +0.0014; 95% paired trip bootstrap interval [-0.0283, +0.0205] (interval includes zero).
  Pooled over points: classifier 0.8649, HMM 0.8511.
- continuity: classifier 0.7568, HMM 0.8568, difference +0.1000; 95% paired trip bootstrap interval [+0.0621, +0.1344] (interval excludes zero).
- top1_clean, pooled over points: classifier 0.8848, HMM 0.8593.

Ablations (mean trip top-1): classifier 0.8693, hmm 0.8707, no_gap_relaxation 0.8706, no_uturn_penalty 0.8721. The classifier is also the emission-only result.
Mean trip arc_precision_vs_reference: classifier 0.6570, HMM 0.8136.
Mean trip arc_recall_vs_reference: classifier 0.8870, HMM 0.8997.
Mean trip arc_f1_vs_reference: classifier 0.7305, HMM 0.8367.
Mean trip length_mismatch_vs_reference: classifier 0.8677, HMM 0.2485.
Pooled candidate miss rate: 0.1514%. HMM changed 2246 points, recorded 0 breaks and relaxed 44 gap steps.
Broken consecutive transitions after collapsing repeats: classifier 1334, HMM 656.
Largest trip: 00000056, 3958 points, 2.391 seconds to decode.
Pruning removed the label for 3 of 19131 points whose label was initially in the candidate set (0.0157%).

## train
- top1_all: classifier 0.8361, HMM 0.8136, difference -0.0226; 95% paired trip bootstrap interval [-0.0406, -0.0045] (interval excludes zero).
  Pooled over points: classifier 0.7673, HMM 0.7313.
- top1_in_candidates: classifier 0.8501, HMM 0.8259, difference -0.0242; 95% paired trip bootstrap interval [-0.0439, -0.0061] (interval excludes zero).
  Pooled over points: classifier 0.8562, HMM 0.8160.
- continuity: classifier 0.7405, HMM 0.8443, difference +0.1038; 95% paired trip bootstrap interval [+0.0822, +0.1277] (interval excludes zero).
- top1_clean, pooled over points: classifier 0.8812, HMM 0.8375.

Ablations (mean trip top-1): classifier 0.8361, hmm 0.8136, no_gap_relaxation 0.8139, no_uturn_penalty 0.8120. The classifier is also the emission-only result.
Mean trip arc_precision_vs_reference: classifier 0.6390, HMM 0.7739.
Mean trip arc_recall_vs_reference: classifier 0.8860, HMM 0.9049.
Mean trip arc_f1_vs_reference: classifier 0.7062, HMM 0.8111.
Mean trip length_mismatch_vs_reference: classifier 1.7463, HMM 0.3974.
Pooled candidate miss rate: 10.3734%. HMM changed 35344 points, recorded 189 breaks and relaxed 1216 gap steps.
Broken consecutive transitions after collapsing repeats: classifier 14608, HMM 6553.
Largest trip: 00000054, 19313 points, 7.960 seconds to decode.
Pruning removed the label for 206 of 186797 points whose label was initially in the candidate set (0.1103%).

## Where HMM did not help

The notes below are characteristics associated with each trip, not proven causes. Differences are in all-point top-1.

- student 00000040: -0.0160; no clear cause from aggregate trip indicators. Reference repeats 0 arcs; largest weak road component contains 99.0% of indexed nodes.
- student 00000077: -0.1117; no clear cause from aggregate trip indicators. Reference repeats 0 arcs; largest weak road component contains 97.1% of indexed nodes.
- student 00000091: -0.0023; no clear cause from aggregate trip indicators. Reference repeats 0 arcs; largest weak road component contains 98.5% of indexed nodes.
- train 00000002: -0.0636; candidate misses 9.6%. Reference repeats 0 arcs; largest weak road component contains 97.7% of indexed nodes.
- train 00000003: -0.1998; no clear cause from aggregate trip indicators. Reference repeats 0 arcs; largest weak road component contains 94.4% of indexed nodes.
- train 00000008: -0.0453; no clear cause from aggregate trip indicators. Reference repeats 1 arcs; largest weak road component contains 98.4% of indexed nodes.
- train 00000010: -0.2398; no clear cause from aggregate trip indicators. Reference repeats 0 arcs; largest weak road component contains 93.4% of indexed nodes.
- train 00000011: -0.0081; no clear cause from aggregate trip indicators. Reference repeats 0 arcs; largest weak road component contains 99.7% of indexed nodes.
- train 00000016: -0.0138; no clear cause from aggregate trip indicators. Reference repeats 0 arcs; largest weak road component contains 93.3% of indexed nodes.
- train 00000021: -0.0730; no clear cause from aggregate trip indicators. Reference repeats 0 arcs; largest weak road component contains 95.1% of indexed nodes.
- train 00000025: -0.0025; no clear cause from aggregate trip indicators. Reference repeats 0 arcs; largest weak road component contains 95.0% of indexed nodes.
- train 00000027: -0.0101; no clear cause from aggregate trip indicators. Reference repeats 0 arcs; largest weak road component contains 97.4% of indexed nodes.
- train 00000031: -0.0966; no clear cause from aggregate trip indicators. Reference repeats 0 arcs; largest weak road component contains 98.3% of indexed nodes.
- train 00000033: -0.0883; far-label share 71.2%, candidate misses 58.2%. Reference repeats 0 arcs; largest weak road component contains 98.6% of indexed nodes.
- train 00000034: -0.0551; no clear cause from aggregate trip indicators. Reference repeats 0 arcs; largest weak road component contains 98.4% of indexed nodes.
- train 00000035: -0.0396; no clear cause from aggregate trip indicators. Reference repeats 0 arcs; largest weak road component contains 91.4% of indexed nodes.
- train 00000036: -0.0782; no clear cause from aggregate trip indicators. Reference repeats 0 arcs; largest weak road component contains 97.8% of indexed nodes.
- train 00000038: -0.0334; no clear cause from aggregate trip indicators. Reference repeats 0 arcs; largest weak road component contains 97.5% of indexed nodes.
- train 00000042: -0.0107; no clear cause from aggregate trip indicators. Reference repeats 0 arcs; largest weak road component contains 94.3% of indexed nodes.
- train 00000046: -0.0178; no clear cause from aggregate trip indicators. Reference repeats 0 arcs; largest weak road component contains 99.0% of indexed nodes.
- train 00000048: -0.0645; no clear cause from aggregate trip indicators. Reference repeats 0 arcs; largest weak road component contains 98.2% of indexed nodes.
- train 00000051: -0.1748; no clear cause from aggregate trip indicators. Reference repeats 0 arcs; largest weak road component contains 98.5% of indexed nodes.
- train 00000054: -0.0005; far-label share 89.4%, candidate misses 86.7%, steps over 50 m 85.6%. Reference repeats 0 arcs; largest weak road component contains 97.0% of indexed nodes.
- train 00000057: -0.0018; no clear cause from aggregate trip indicators. Reference repeats 0 arcs; largest weak road component contains 99.3% of indexed nodes.
- train 00000061: -0.0385; no clear cause from aggregate trip indicators. Reference repeats 0 arcs; largest weak road component contains 98.5% of indexed nodes.
- train 00000062: -0.0657; no clear cause from aggregate trip indicators. Reference repeats 0 arcs; largest weak road component contains 97.0% of indexed nodes.
- train 00000063: -0.0694; no clear cause from aggregate trip indicators. Reference repeats 0 arcs; largest weak road component contains 100.0% of indexed nodes.
- train 00000064: -0.0014; no clear cause from aggregate trip indicators. Reference repeats 0 arcs; largest weak road component contains 98.3% of indexed nodes.
- train 00000066: -0.0111; candidate misses 8.0%. Reference repeats 0 arcs; largest weak road component contains 95.2% of indexed nodes.
- train 00000068: -0.0174; no clear cause from aggregate trip indicators. Reference repeats 0 arcs; largest weak road component contains 94.5% of indexed nodes.
- train 00000069: -0.0203; no clear cause from aggregate trip indicators. Reference repeats 29 arcs; largest weak road component contains 98.8% of indexed nodes.
- train 00000070: -0.0679; no clear cause from aggregate trip indicators. Reference repeats 0 arcs; largest weak road component contains 99.6% of indexed nodes.
- train 00000072: -0.1022; no clear cause from aggregate trip indicators. Reference repeats 0 arcs; largest weak road component contains 89.0% of indexed nodes.
- train 00000075: -0.0725; no clear cause from aggregate trip indicators. Reference repeats 2 arcs; largest weak road component contains 94.4% of indexed nodes.
- train 00000079: -0.2903; no clear cause from aggregate trip indicators. Reference repeats 0 arcs; largest weak road component contains 98.6% of indexed nodes.
- train 00000080: -0.1873; no clear cause from aggregate trip indicators. Reference repeats 0 arcs; largest weak road component contains 98.4% of indexed nodes.
- train 00000081: -0.1210; no clear cause from aggregate trip indicators. Reference repeats 0 arcs; largest weak road component contains 90.5% of indexed nodes.
- train 00000082: -0.3015; no clear cause from aggregate trip indicators. Reference repeats 0 arcs; largest weak road component contains 96.6% of indexed nodes.
- train 00000083: -0.2455; no clear cause from aggregate trip indicators. Reference repeats 0 arcs; largest weak road component contains 98.0% of indexed nodes.
- train 00000084: -0.0420; no clear cause from aggregate trip indicators. Reference repeats 2 arcs; largest weak road component contains 98.9% of indexed nodes.
- train 00000085: +0.0000; no clear cause from aggregate trip indicators. Reference repeats 0 arcs; largest weak road component contains 97.7% of indexed nodes.
- train 00000086: -0.0490; no clear cause from aggregate trip indicators. Reference repeats 2 arcs; largest weak road component contains 98.1% of indexed nodes.
- train 00000088: -0.0308; no clear cause from aggregate trip indicators. Reference repeats 0 arcs; largest weak road component contains 97.1% of indexed nodes.
- train 00000092: -0.0073; no clear cause from aggregate trip indicators. Reference repeats 14 arcs; largest weak road component contains 99.0% of indexed nodes.
- train 00000093: -0.0512; no clear cause from aggregate trip indicators. Reference repeats 0 arcs; largest weak road component contains 98.8% of indexed nodes.
- train 00000095: -0.0361; no clear cause from aggregate trip indicators. Reference repeats 7 arcs; largest weak road component contains 69.3% of indexed nodes.
- train 00000097: -0.0023; no clear cause from aggregate trip indicators. Reference repeats 0 arcs; largest weak road component contains 98.6% of indexed nodes.

## Interpretation notes

Route precision, recall and F1 use the longest common subsequence against the offline reference route. Length mismatch uses the lengths of arcs in the multiset symmetric difference divided by reference length; it ignores arc order and can exceed one. Consecutive GPS observations can skip intermediate arcs, so the collapsed per-point prediction does not reconstruct every traversed arc. Repeated reference arcs and weak-component sizes are proxies for loops and network fragmentation, not proof that either caused an error. Listed trip characteristics are associations rather than established causes.

Top-k pruning retained 15 candidates per point; counts of reference-label losses are in pruning.csv. Per-trip diagnostics and ablations are in per_trip_metrics.csv.
