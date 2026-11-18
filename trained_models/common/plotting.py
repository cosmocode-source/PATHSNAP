"""Plotting helpers shared by the model scripts and the comparison notebook.

Every plot is drawn from real results passed in by the caller. Loss curves are only drawn when a real per-iteration /
per-epoch history exists (LightGBM, MLP); no curve is ever drawn for models without a training history.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")            # file output only (scripts run without a display)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_fold_metrics(fold_df: pd.DataFrame, title: str, path: Path) -> None:
    """Bar chart of Top-1 (all), Top-1 (in candidates) and Macro F1 for each CV fold (train partition, out-of-fold)."""
    cols = ["top1_all", "top1_in_cands", "macro_f1"]
    sub = fold_df[fold_df["subset"] == "all"].set_index("fold")[cols]
    fig, ax = plt.subplots(figsize=(7, 4))
    sub.plot.bar(ax=ax)
    ax.set_ylim(max(0.0, float(sub.min().min()) - 0.05), 1.0)
    ax.set_ylabel("score")
    ax.set_title(f"{title}: 5-fold CV (train partition, all points)")
    ax.legend(loc="lower right")
    _save(fig, path)


def plot_confusion(cm, title: str, path: Path) -> None:
    """Candidate-level confusion matrix at the top-1 decision ([[TN, FP], [FN, TP]])."""
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", color="black")
    ax.set_xticks([0, 1], ["pred 0", "pred 1 (top-1)"])
    ax.set_yticks([0, 1], ["true 0", "true 1"])
    ax.set_title(title, fontsize=9)
    _save(fig, path)


def plot_reliability(tables: dict[str, pd.DataFrame], title: str, path: Path) -> None:
    """Calibration (reliability) curve(s): mean predicted probability vs observed positive rate per bin."""
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="perfect")
    for name, t in tables.items():
        ax.plot(t["mean_pred"], t["frac_pos"], marker="o", label=name)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed share of true arcs")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_loss_curves(histories: dict[str, dict], title: str, ylabel: str, path: Path) -> None:
    """Real training / validation loss histories. ``histories[name] = {"train": [...], "val": [...]}`` (either optional)."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, h in histories.items():
        if h.get("train") is not None:
            ax.plot(np.arange(1, len(h["train"]) + 1), h["train"], ls="--", lw=1, label=f"{name} train")
        if h.get("val") is not None:
            ax.plot(np.arange(1, len(h["val"]) + 1), h["val"], lw=1.3, label=f"{name} validation")
    ax.set_xlabel("iteration / epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    _save(fig, path)


def plot_per_trip(per_trip: pd.DataFrame, title: str, path: Path, flag_below: float) -> None:
    """Per-trip Top-1 (all points), weakest first; trips below ``flag_below`` are drawn in a warning colour."""
    d = per_trip.sort_values("top1_all")
    colors = np.where(d["top1_all"] < flag_below, "#d62728", "#1f77b4")
    fig, ax = plt.subplots(figsize=(max(6, 0.12 * len(d) + 2), 4))
    ax.bar(d["trip_id"], d["top1_all"], color=colors)
    ax.set_ylabel("Top-1 (all points)")
    ax.set_title(title, fontsize=9)
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    _save(fig, path)


def plot_model_bars(df: pd.DataFrame, cols: list[str], title: str, ylabel: str, path: Path, err: dict | None = None) -> None:
    """Grouped bars of one or more metrics per model (comparison plots). ``err`` maps column -> std column."""
    x = np.arange(len(df))
    wbar = 0.8 / len(cols)
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(df)), 4.2))
    for i, c in enumerate(cols):
        yerr = df[err[c]].to_numpy() if err and c in err else None
        ax.bar(x + i * wbar - 0.4 + wbar / 2, df[c].to_numpy(), wbar, yerr=yerr, capsize=3, label=c)
    ax.set_xticks(x, df["model"], rotation=15)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    vals = df[cols].to_numpy(float)
    if np.nanmin(vals) > 0.5:
        ax.set_ylim(np.nanmin(vals) - 0.05, min(1.0, np.nanmax(vals) + 0.03))
    _save(fig, path)
