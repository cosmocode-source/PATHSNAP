"""Exploratory data analysis for the PathSnap dataset.

Add analysis functions here as the dataset investigation progresses.
"""

from pathlib import Path


DATASET_DIR = Path(__file__).resolve().parents[2] / "Dataset"


def main() -> None:
    """Run the exploratory analysis."""
    print(f"Dataset directory: {DATASET_DIR}")
    print("Analytics steps will be added here.")


if __name__ == "__main__":
    main()
