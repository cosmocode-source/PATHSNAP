# PathSnap data analysis

`PathSnap_EDA.ipynb` is the single notebook used to understand and validate the
raw GPS and road-network data before model training. It does not train a model.

## Data and index spaces

Each trip contains `.track`, `.nodes`, `.arcs`, and `.route` files. Their rows
have different meanings and must never be aligned by row number:

- GPS row: an observation in time order.
- `node_id`: the row index in `.nodes`.
- `arc_id`: the row index in `.arcs`.
- Route position: a position in `.route`, whose value is an `arc_id`.

The notebook creates arc geometries from node IDs and relates GPS points to road
arcs only by distances in a per-trip UTM projection.

## Partition rules

Training trips are used to learn data-quality thresholds, transforms, candidate
settings, labels, and feature decisions. Student-test trips receive the same
fixed preprocessing and flags, and are used only for descriptive train-vs-student
shift checks. GPS rows are never dropped from student-test trips.

## Run

Create a virtual environment, install `requirements.txt`, then open
`PathSnap_EDA.ipynb` and use **Restart and Run All**. The notebook discovers the
available dataset partitions from `../Dataset`.

## Notebook process

The numbered notebook sections inventory files; validate schemas and IDs;
demonstrate the four index spaces; analyse duplicates, time, GPS plausibility,
road topology, and reference routes; generate candidates; compare possible
point-labelling rules; inspect candidate features; and check train/student shift.
It records all thresholds in `CONFIG` and saves learned decisions under
`outputs/` after the preprocessing plan is approved.

## Outputs

- `outputs/tables/`: inventories and summary tables.
- `outputs/figures/`: analysis charts.
- `outputs/maps/`: Folium maps created from training trips.
- `outputs/cleaned/`: cleaned training data and flagged non-training data.
- `outputs/decisions/`: data-quality and feature-decision logs.
- `outputs/preprocessing_config.json`: fixed, training-derived preprocessing
  settings for downstream inference.

## Limitations

Route files are treated as reference information, not unquestionable physical
truth. Candidate settings and labels are selected only from training evidence;
the notebook flags uncertainty around gaps, noisy GPS, and ambiguous road
segments for downstream map matching.
