"""Run all five PathSnap models one after another in ONE terminal, then print every model's results together.

What it does
------------
1. Runs model1 ... model5 training scripts in order, each in its own Python process, streaming its log live to this
   terminal (CV folds, final fit, student-test result). One model at a time, so timings stay comparable.
2. If a model fails, it stops and says which one (nothing after it is run).
3. At the end, reads every model's results.json (written by the runs) and prints one summary table plus the selected
   model. Every number printed comes from those files; nothing is typed in.

Re-running overwrites each model's outputs (results.json, figures, model files) with the new run's outputs.

Usage (from the repository root, in a VS Code terminal):
    .venv\\Scripts\\python trained_models\\run_all_models.py                 # train all five, then the summary
    .venv\\Scripts\\python trained_models\\run_all_models.py --models 1 4    # only some models, then the summary
    .venv\\Scripts\\python trained_models\\run_all_models.py --summary-only  # no training: summary of saved results
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent          # trained_models/
MODELS = {                                      # number -> (folder, script, display name)
    1: ("model1", "train_logreg.py", "Logistic Regression"),
    2: ("model2", "train_knn.py", "KNN (distance-weighted)"),
    3: ("model3", "train_random_forest.py", "Random Forest"),
    4: ("model4", "train_lightgbm.py", "LightGBM"),
    5: ("model5", "train_mlp.py", "Shallow MLP (3 seeds)"),
}
LINE = "=" * 110


def banner(text: str) -> None:
    print(f"\n{LINE}\n  {text}\n{LINE}", flush=True)


def run_model(num: int) -> float:
    """Run one training script as a child process with live (unbuffered) output; return wall time in seconds."""
    folder, script, name = MODELS[num]
    banner(f"MODEL {num}/5: {name}   ({folder}/{script})")
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONWARNINGS": "ignore"}
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, "-u", str(HERE / folder / script)], cwd=HERE, env=env)
    secs = time.perf_counter() - t0
    if proc.returncode != 0:
        banner(f"MODEL {num} ({name}) FAILED with exit code {proc.returncode}. Stopping here.")
        sys.exit(proc.returncode)
    print(f"\n>>> model {num} ({name}) finished in {secs / 60:.1f} min", flush=True)
    return secs


def summary() -> None:
    """Print one table with every model's results, read from the saved results.json files."""
    rows = []
    for num, (folder, _, name) in MODELS.items():
        path = HERE / folder / "results.json"
        if not path.exists():
            rows.append((num, name, None))
            continue
        rows.append((num, name, json.loads(path.read_text(encoding="utf-8"))))

    banner("RESULTS OF ALL MODELS  (CV = 5-fold out-of-fold on the 80 TRAIN trips; Student = 10 student_test trips)")
    head = (f"{'#':<2} {'Model':<24} {'CV Top-1 all':>15} {'CV Top-1':>9} {'CV Macro':>9} {'CV LL':>7} "
            f"{'Stud.':>7} {'Stud.':>7} {'Stud.':>7} {'Route':>7} {'Infer ms':>9} {'Fit s':>7}")
    sub = (f"{'':<2} {'':<24} {'mean +/- std':>15} {'in-cands':>9} {'F1':>9} {'norm':>7} "
           f"{'Top-1':>7} {'MacroF1':>7} {'LL norm':>7} {'contin.':>7} {'/1k pts':>9} {'final':>7}")
    print(head)
    print(sub)
    print("-" * 110)
    for num, name, r in rows:
        if r is None:
            print(f"{num:<2} {name:<24} (no results.json yet: model not run)")
            continue
        cv, st = r["cv"]["all"], r["student"]["all"]
        print(f"{num:<2} {name:<24} "
              f"{cv['top1_all']['mean']:.4f} +/- {cv['top1_all']['std']:.3f} "
              f"{cv['top1_in_cands']['mean']:>9.4f} {cv['macro_f1']['mean']:>9.4f} {cv['logloss_norm']['mean']:>7.3f} "
              f"{st['top1_all']:>7.4f} {st['macro_f1']:>7.4f} {st['logloss_norm']:>7.3f} "
              f"{r['route_continuity']['student']['pooled']:>7.3f} "
              f"{r['timing']['inference_ms_per_1000_points']:>9.1f} {r['timing']['final_fit_seconds']:>7.1f}")
    print("-" * 110)

    done = [(num, name, r) for num, name, r in rows if r is not None]
    if len(done) < 2:
        return
    first = done[0][2]
    print(f"Candidate miss rate (ceiling on Top-1, same for every model): train {first['candidate_miss_rate']['train']:.4f}, "
          f"student {first['candidate_miss_rate']['student']:.4f}")

    # Selection rule: primary CV Top-1 (all points); a gap below the MLP's seed-to-seed std counts as a tie;
    # ties are broken by the CV normalised log loss (what the HMM uses).
    tol = 0.0
    mlp = HERE / "model5" / "results.json"
    if mlp.exists() and "seed_std" in json.loads(mlp.read_text(encoding="utf-8")):
        tol = json.loads(mlp.read_text(encoding="utf-8"))["seed_std"]["cv"]["all"]["top1_all"]["mean"]
    best_top1 = max(r["cv"]["all"]["top1_all"]["mean"] for _, _, r in done)
    tied = [(num, name, r) for num, name, r in done if r["cv"]["all"]["top1_all"]["mean"] >= best_top1 - tol]
    chosen = min(tied, key=lambda t: t[2]["cv"]["all"]["logloss_norm"]["mean"])
    print(f"Selection rule: highest CV Top-1 (all points); gaps below {tol:.5f} (MLP seed-to-seed std) count as a tie; "
          f"tie -> lowest CV normalised log loss.")
    print(f"Tied on CV Top-1: {', '.join(n for _, n, _ in tied)}")
    print(f"SELECTED MODEL: {chosen[1]}  (model{chosen[0]})")
    print(f"Full comparison: trained_models/comparison/model_comparison.csv and model_comparison.ipynb\n", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run all PathSnap models in one terminal and print a combined summary.")
    ap.add_argument("--models", type=int, nargs="+", choices=sorted(MODELS), default=sorted(MODELS),
                    help="model numbers to train (default: all five)")
    ap.add_argument("--summary-only", action="store_true", help="do not train; print the summary of saved results")
    args = ap.parse_args()

    if not args.summary_only:
        banner(f"PathSnap: training models {args.models} one after another (python {sys.version.split()[0]})")
        times = {n: run_model(n) for n in args.models}
        banner("ALL REQUESTED MODELS FINISHED: " + ", ".join(f"model{n} {t / 60:.1f} min" for n, t in times.items()))
    summary()


if __name__ == "__main__":
    main()
