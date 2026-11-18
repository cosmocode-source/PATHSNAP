"""Model 5 - Shallow MLP (neural network family, PyTorch) for PathSnap candidate scoring.

What: one hidden layer (22 -> 64 ReLU -> dropout -> 1 logit), trained with the sample-weighted binary cross-entropy.
Why: a neural model can learn smooth non-linear combinations of the features. It is kept shallow because 22 tabular
features do not justify a deep network. PyTorch is used because sklearn's MLPClassifier cannot take sample weights,
and the far / ambiguous down-weighting is part of our training design.

Data: SCALED table (neural networks need standardised inputs).

Early stopping (trip-grouped): inside the rows ``fit`` receives, EARLY_STOP_VAL_SHARE of the trips are held out
(GroupShuffleSplit by trip_id). After every epoch the real train and validation loss are computed in eval mode (no
dropout) on the full train-part and validation-part, and saved. Training stops after PATIENCE epochs without
validation improvement and the best epoch's weights are restored. Unlike LightGBM there is no refit on all rows: an MLP
refit would need the epoch count to transfer to a different data size, and the 3-seed design already costs 18 fits.

Seeds: the whole CV + final fit is run for 3 seeds (42, 43, 44) to show how stable the network is. Headline numbers in
results.json are the mean over seeds; ``seed_std`` holds the spread.

Run from the trained_models folder:   ..\.venv\Scripts\python model5\train_mlp.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))           # make trained_models/common importable

from common.data_loading import MODEL_DATA, SEED, feature_list  # noqa: E402
from common.experiment import ModelSpec, run_experiment  # noqa: E402

# ---- configuration -----------------------------------------------------------------------------------------------
CONFIG = {
    "architecture": "22 -> 64 (ReLU) -> dropout 0.1 -> 1 logit",
    "hidden_units": 64,
    "dropout": 0.1,
    "optimizer": "Adam",
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,           # small L2 on the weights
    "batch_size": 4096,
    "max_epochs": 40,
    "patience": 5,                  # early stopping on the trip-grouped validation loss
    "early_stop_val_share": 0.2,
    "refit_on_all_rows": False,     # see module docstring
    "seeds": [SEED, SEED + 1, SEED + 2],
    "torch_threads": 16,
    "sample_weight": "clean 1.0 / ambiguous 0.5 / far 0.3 (in the loss, train and validation)",
}
torch.set_num_threads(CONFIG["torch_threads"])


class ShallowMLP(nn.Module):
    """22 -> hidden (ReLU) -> dropout -> 1 logit."""

    def __init__(self, n_in: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


@torch.no_grad()
def _weighted_bce(model, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor, chunk: int = 262_144) -> float:
    """Sample-weighted mean BCE over a whole set, eval mode (the real loss used for curves and early stopping)."""
    model.eval()
    tot = 0.0
    for i in range(0, len(X), chunk):
        logit = model(X[i:i + chunk])
        tot += float((nn.functional.binary_cross_entropy_with_logits(logit, y[i:i + chunk], reduction="none") * w[i:i + chunk]).sum())
    return tot / float(w.sum())


def fit(X, y, w, trip_ids, seed):
    """Train with mini-batch Adam, early stopping on a trip-grouped inner validation split; return best-epoch model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    gss = GroupShuffleSplit(n_splits=1, test_size=CONFIG["early_stop_val_share"], random_state=seed)
    tr, va = next(gss.split(X, y, groups=trip_ids))
    Xt, yt, wt = (torch.from_numpy(X[tr]), torch.from_numpy(y[tr].astype(np.float32)), torch.from_numpy(w[tr]))
    Xv, yv, wv = (torch.from_numpy(X[va]), torch.from_numpy(y[va].astype(np.float32)), torch.from_numpy(w[va]))

    model = ShallowMLP(X.shape[1], CONFIG["hidden_units"], CONFIG["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    gen = torch.Generator().manual_seed(seed)          # seeded shuffling of mini-batches
    hist = {"train": [], "val": []}
    best, best_state, bad = np.inf, None, 0
    for epoch in range(CONFIG["max_epochs"]):
        model.train()
        perm = torch.randperm(len(Xt), generator=gen)
        for i in range(0, len(perm), CONFIG["batch_size"]):
            b = perm[i:i + CONFIG["batch_size"]]
            loss = (nn.functional.binary_cross_entropy_with_logits(model(Xt[b]), yt[b], reduction="none") * wt[b]).sum() / wt[b].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        hist["train"].append(_weighted_bce(model, Xt, yt, wt))
        hist["val"].append(_weighted_bce(model, Xv, yv, wv))
        if hist["val"][-1] < best - 1e-6:
            best, bad = hist["val"][-1], 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= CONFIG["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    hist["best_epoch"] = int(np.argmin(hist["val"]) + 1)
    print(f"    epochs run {len(hist['val'])}, best epoch {hist['best_epoch']}, val loss {best:.5f}")
    return model, hist


@torch.no_grad()
def predict(model, X):
    """Sigmoid of the logit = probability of class 1 (eval mode, chunked)."""
    model.eval()
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    return torch.cat([torch.sigmoid(model(Xt[i:i + 262_144])) for i in range(0, len(Xt), 262_144)]).numpy().astype(np.float64)


def save(model, folder: Path, seed: int):
    """Save the final network weights for this seed, the architecture, and a copy of the train-fitted scaler."""
    torch.save(model.state_dict(), folder / f"mlp_seed{seed}.pt")
    (folder / "mlp_architecture.json").write_text(json.dumps(
        {"n_in": len(feature_list()), "hidden": CONFIG["hidden_units"], "dropout": CONFIG["dropout"],
         "features": feature_list(), "class": "ShallowMLP in train_mlp.py"}, indent=2), encoding="utf-8")
    shutil.copy(MODEL_DATA / "scaler.joblib", folder / "scaler.joblib")
    return [f"mlp_seed{seed}.pt", "mlp_architecture.json", "scaler.joblib"]


if __name__ == "__main__":
    run_experiment(ModelSpec(
        name="Shallow MLP", family="Neural network", folder=HERE, scaled=True, config=CONFIG,
        fit=fit, predict=predict, save=save, seeds=CONFIG["seeds"],
        loss_name="sample-weighted binary cross-entropy (Adam, weight decay 1e-5)",
        history_ylabel="binary cross-entropy (sample-weighted, eval mode)",
    ))
