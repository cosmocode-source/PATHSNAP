"""Reusable preprocessing pipeline for PathSnap."""

from pathlib import Path


DATASET_DIR = Path(__file__).resolve().parents[2] / "Dataset"


def preprocess_dataset() -> None:
    """Prepare raw PathSnap data for modelling."""
    print(f"Dataset directory: {DATASET_DIR}")
    print("Preprocessing steps will be added here.")


if __name__ == "__main__":
    preprocess_dataset()
