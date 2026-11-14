"""How much a window-level split inflates results: the same model, two ways of splitting.

    leaky   StratifiedKFold over 10 s windows - a person's windows land in train AND test, as in many
            published EEG papers
    honest  StratifiedGroupKFold over people - each person is entirely in train or in test

Model: logistic regression (C=0.05) on relative band power + coherence per window; window
probabilities scored directly (no subject averaging) so both arms are judged identically.
If the leaky arm is near-perfect even where the honest arm is near chance (AD vs FTD,
Semmelweis), the leaky number measures recognising individuals, not disease.

    python -m eeg.experiments.leakage_demo --out ~/eeg/raw2
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from eeg.data import raw_eeg as R
from eeg.experiments.baselines import lr_probe

TASKS = [("alzheimers", "ds004504"), ("ftd", "ds004504"), ("alzheimers_vs_ftd", "ds004504"),
         ("mood", "mumtaz"), ("schizophrenia", "aseeg"), ("schizophrenia", "warsaw"), ("schizophrenia", "semmelweis")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/eeg/raw2")
    ap.add_argument("--save", default="results/leakage_demo.json")
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    report = {}
    for task, ds in TASKS:
        pos, neg, _ = R.TASKS[task]
        ab, coh, subj, groups, _, _ = R.load_features(out, [ds])
        keep, y = R.task_rows(groups, pos, neg)
        iu = np.triu_indices(coh.shape[1], 1)
        X = np.c_[ab[keep].reshape(len(keep), -1), coh[keep][:, iu[0], iu[1], :].reshape(len(keep), -1)]
        s = subj[keep]
        res = {}
        for name, split in (("leaky", StratifiedKFold(5, shuffle=True, random_state=0).split(X, y)),
                            ("honest", StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, s))):
            p = np.zeros(len(y))
            for a, b in split:
                p[b] = lr_probe().fit(X[a], y[a]).predict_proba(X[b])[:, 1]
            res[name] = {"auc": float(roc_auc_score(y, p)), "accuracy": float(accuracy_score(y, p > 0.5))}
        report[f"{task}:{ds}"] = {"windows": int(len(y)), "people": int(len(np.unique(s))), **res}
        print(f"{task:<18} {ds:<11} leaky AUC {res['leaky']['auc']:.3f} ({res['leaky']['accuracy']:.0%})   "
              f"honest AUC {res['honest']['auc']:.3f} ({res['honest']['accuracy']:.0%})", flush=True)
    Path(args.save).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
