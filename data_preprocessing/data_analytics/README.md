# PathSnap data analytics

`PathSnap_EDA.ipynb` is the single notebook used to analyse and validate the
PathSnap GPS and road-network data before model training. It does not train a
model and does not create cleaned data.

## How the analysis was done

The notebook automatically discovers the `train` and `student_test` partitions
inside `Dataset/`. It reads `.track`, `.nodes`, `.arcs`, and `.route` files with
explicit schemas and keeps their native row indexes unchanged.

The analysis checks:

- File inventory, row counts, missing values, coordinate plausibility, and ID bounds.
- Duplicate GPS points, stationary points, repeated road arcs, timestamps, GPS gaps,
  speed, and robust speed outliers.
- Per-trip UTM projection, directed road-network connectivity, and reference-route
  continuity.
- Candidate road segments at several metre-based radii.
- Nearest-route versus monotonic route-label agreement.
- Candidate feature distributions and train-vs-student distribution shift.

Important: GPS-row indexes, node IDs, arc IDs, and route positions are different
index spaces. GPS observations are related to arcs only through projected spatial
distance, never by matching rows across files.

## Data used

- Training: 80 trips. Used for data-quality analysis and candidate/feature testing.
- Student-test: 10 trips. Used only for descriptive comparison; it does not choose
  thresholds, features, or transformations.

## Results

- No missing values, invalid coordinates, `(0,0)` GPS placeholders, invalid arc
  endpoints, or invalid route arc IDs were found.
- Sampling is close to 1 Hz: median one-second interval share is 99.1% in training
  and 99.9% in student-test.
- No exact duplicate GPS rows, duplicate timestamps, or negative timestamp deltas
  were found. Stationary points and long GPS gaps exist and should be flagged.
- The road network is per-trip, so `arc_id` is a local identifier and should not be
  used as a global multiclass label.
- Pilot road networks had no isolated nodes and had fully connected consecutive
  reference-route transitions.
- Candidate-radius testing showed that 50 m gives about 99.3% route-set coverage
  with about 5.47 candidate arcs per sampled GPS point. A 100 m radius gives 100%
  coverage but about 10.46 candidates per point.
- Initial recommendation: use 50 m as the normal candidate radius and 100 m only as
  a fallback for difficult points.
- Nearest-route and monotonic route labels disagreed substantially on one pilot
  trip. Use a route-constrained dynamic-programming or Viterbi-style labelling rule
  rather than nearest-arc labels alone.
- Arc length is strongly right-skewed; use `log1p(arc_length_m)` when it becomes a
  model feature. Perpendicular distance and candidate rank should be retained.
- KS tests found no strong train-vs-student shift for path length, maximum speed, or
  gap count in the current summaries.

## Outputs

- `outputs/tables/`: CSV inventories and test summaries.
- `outputs/figures/`: timestamp and feature-distribution figures.
- `outputs/maps/`: Folium training-trip map.

No cleaned data, trained model, or frozen preprocessing configuration is written by
this notebook.

## Run

Install `requirements.txt`, open `PathSnap_EDA.ipynb`, and use **Restart and Run
All** from the repository root.
