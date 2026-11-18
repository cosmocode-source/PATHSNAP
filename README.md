# PathSnap

Machine-learning system for matching noisy GPS observations to a road network and reconstructing vehicle routes.

## Team

| USN | Name |
|---|---|
| PES1UG24CS078 | Archana Shivakumar |
| PES1UG24CS079 | Arin Singh |
| PES1UG24CS120 | Catherine D |
| PES1UG24CS121 | Ch Yashwitha |

## Dataset

The `Dataset` folder contains:

- `train/` - 80 trips for training, validation, and model optimisation.
- `student_test/` - 10 trips for development-stage testing.
- `.track` - GPS longitude, latitude, and timestamp.
- `.nodes` - road-network node coordinates.
- `.arcs` - road-network connections.
- `.route` - reference road-segment sequence.

Evaluator data must remain separate and must not be used for training or tuning.

## Required System

- Data preprocessing and feature engineering.
- At least five trained and compared ML models.
- Bayesian hyperparameter optimisation for the selected model.
- HMM/Viterbi route reconstruction.
- FastAPI backend for inference and evaluation.
- UI for selecting trips and displaying predicted routes/results.
- Folium map visualisations.

## Required Submission

- Complete source code, model files, configurations, experiment results, and run instructions in this repository.
- One technical-report PDF containing EDA, model comparisons, optimisation results, HMM/Viterbi results, maps, and API/UI screenshots.

## Repository

GitHub: https://github.com/cosmocode-source/PATHSNAP.git

## Status

Project setup and dataset are present. Processing, modelling, API, UI, evaluation, and report files will be added as development continues.
