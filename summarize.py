#!/usr/bin/env python3
"""Tabulate park.py result JSONs against the classical and published references."""
import glob
import json
import sys

CONTROLS = {
    "binary": [("logreg (AB+COH)", 0.731), ("random forest", 0.721),
               ("Park et al. 2021 published", 0.938)],
    "multiclass": [("logreg (AB+COH)", 0.428), ("random forest", 0.449),
                   ("majority class (Mood 266/664)", 0.401)],
}
LABEL = {"binary": "binary: Schizophrenia vs Healthy, n=212",
         "multiclass": "4-way: Healthy/SZ/Mood/Addictive, n=664"}


def main(pattern="res/*.json"):
    rows = []
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f))
        a, s = d["args"], d["summary"]
        rows.append((a["task"], a["model"], a.get("node_features", "power"),
                     s["accuracy"], s["bal_acc"], s["macro_f1"], s["macro_auc"]))

    for task in ("binary", "multiclass"):
        sub = sorted([r for r in rows if r[0] == task], key=lambda r: -r[3][0])
        if not sub:
            continue
        print(f"\n=== {LABEL[task]} ===")
        print(f"{'model':<8} {'nodefeat':<10} {'accuracy':>16} {'bal_acc':>10} "
              f"{'macro_f1':>10} {'macro_auc':>11}")
        for _, m, nf, acc, bal, f1, auc in sub:
            print(f"{m:<8} {nf:<10} {acc[0]:>9.3f} +-{acc[1]:.3f} {bal[0]:>10.3f} "
                  f"{f1[0]:>10.3f} {auc[0]:>11.3f}")
        print(f"{'':<10} {'-' * 48}")
        for name, val in CONTROLS[task]:
            print(f"{name:<34} {val:>9.3f}  (accuracy)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "res/*.json")
