r"""Redraw the loss-curve figures of LightGBM / MLP from their saved REAL histories (loss_history.json).

Used after a plotting-style change, so the models do not have to be retrained. No numbers are changed.
Run from trained_models:  ..\.venv\Scripts\python -m common.redraw_loss_curves
"""
import json
from pathlib import Path

from common import plotting

ROOT = Path(__file__).resolve().parents[1]
for folder, name, ylabel in [("model4", "LightGBM", "binary log loss (sample-weighted)"),
                             ("model5", "Shallow MLP", "binary cross-entropy (sample-weighted, eval mode)")]:
    hist = json.loads((ROOT / folder / "loss_history.json").read_text())
    for seed, hs in hist.items():
        plotting.plot_loss_curves(hs, f"{name} (seed {seed}): real loss history per CV fold and final fit", ylabel,
                                  ROOT / folder / "figures" / f"loss_curves_seed{seed}.png")
        print("redrawn", folder, seed)
