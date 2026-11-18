"""Finish the untuned-vs-tuned comparison from the SAVED harness outputs (no model is re-trained).

Why this file exists: the notebook kernel died (memory) after both harness runs (untuned_rerun/, tuned_model/) had
completed and written all their files, while it was building the calibration plot from in-memory out-of-fold
predictions. This script runs the same remaining notebook cells (table, paired comparison, bootstrap, over-fitting
checks, decision, final_model/, results.json, REPORT_ANSWERS.md) by executing their code from tune_lightgbm.ipynb.
The only change: calibration uses the saved files instead of in-memory OOF predictions:
  * CV out-of-fold ECE comes from each run's results.json (computed by the harness on every training row),
  * reliability curves are drawn from the saved student predictions, and each run's harness calibration.png
    (which includes the CV out-of-fold curve) is kept in its own figures/ folder.
Run from this folder:  python finish_comparison.py
"""
import json, sys, types
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
nb = json.loads((HERE / "tune_lightgbm.ipynb").read_text(encoding="utf-8"))
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
g = {"display": lambda x: print(x.to_string() if hasattr(x, "to_string") else x), "__name__": "__finish__"}

exec(cells[0], g)                     # setup
exec(cells[1], g)                     # load the finished study (no new trials)
exec(cells[2], g)                     # trials table, chosen trial, top-5 spread
# parameter importance only (figures already saved by the notebook run)
import optuna
g["imp"] = optuna.importance.get_param_importances(g["study"], evaluator=optuna.importance.FanovaImportanceEvaluator(seed=g["SEED"]))
g["frozen"] = json.loads((HERE / "best_params.json").read_text()); g["best_params"] = g["frozen"]["params"]
# model 4 module + tuned parameter dict (the spec cell minus the harness runs)
spec_cell = cells[5].split("caps = {")[0]
exec(spec_cell, g)
g["RES"] = {"untuned": json.loads((HERE / "untuned_rerun" / "results.json").read_text()),
            "tuned": json.loads((HERE / "tuned_model" / "results.json").read_text())}
for i in (6, 7, 8, 9):                # re-run check, table, paired + bootstrap, over-fitting
    exec(cells[i].replace("plt.show()", "plt.close('all')"), g)

# calibration from saved files
from common.metrics import reliability_table, expected_calibration_error
CAL = {}; fig, ax = plt.subplots(figsize=(6, 4.2))
for name, folder, c in [("untuned", HERE / "untuned_rerun", "#8d99ae"), ("tuned", HERE / "tuned_model", "#2a6f97")]:
    sp = pd.read_parquet(folder / "student_predictions.parquet"); y, p = sp.label.to_numpy(), sp.p_seed42.to_numpy()
    rt = reliability_table(y, p); ece_st = expected_calibration_error(y, np.clip(p, 0, 1))
    ax.plot(rt.mean_pred, rt.frac_pos, "o-", c=c, label=f"{name} (student ECE {ece_st:.4f})")
    r = g["RES"][name]
    CAL[name] = {"ece_cv_oof": r["cv"]["all"]["ece"]["mean"], "ece_student": ece_st,
                 "logloss_norm_cv": r["cv"]["all"]["logloss_norm"]["mean"], "logloss_norm_student": r["student"]["all"]["logloss_norm"]}
ax.plot([0, 1], [0, 1], "k--", lw=.8); ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed share of true arcs")
ax.set_title("Calibration on student-test (CV out-of-fold curves: each run's figures/calibration.png)"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(g["FIG"] / "calibration_untuned_vs_tuned.png", dpi=140); plt.close("all")
g["CAL"] = CAL; g["ECE_LIMIT"] = 0.02; g["need_cal"] = CAL["tuned"]["ece_cv_oof"] > 0.02
print("calibration:", {k: {kk: round(vv, 5) for kk, vv in v.items()} for k, v in CAL.items()}, "| calibration variant needed:", g["need_cal"])

for i in (11, 12, 13):                # decision + final_model/, results.json, REPORT_ANSWERS.md
    exec(cells[i], g)
print("finished")
