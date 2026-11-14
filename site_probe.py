"""How much 'which hospital' is in the features, next to how much 'which disease'.

Per multi-hospital task, one row per subject (mean over its windows of the same
node and edge features the GNNs read). A standardised logistic regression,
5-fold subject-wise, predicts:
    site            hospital, all subjects
    site_controls   hospital, controls only - healthy people, so any skill is pure site signal
    disease         patient vs control, pooled across hospitals
Balanced accuracy; chance = 1 / n_classes. If site_controls is near 1.0 while
disease is near chance, a model has an easier shortcut than the disease.

    python site_probe.py --out ~/eeg/raw2
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import raw_eeg as R


def subject_table(out, datasets, positive, negative):
    ab, coh, subject, groups, ds, _ = R.load_features(out, datasets)
    keep, y = R.task_rows(groups, positive, negative)
    ab, coh, subject, ds = ab[keep], coh[keep], subject[keep], ds[keep]
    iu = np.triu_indices(coh.shape[1], 1)
    feats = np.concatenate([ab.reshape(len(ab), -1), coh[:, iu[0], iu[1], :].reshape(len(coh), -1)], 1)
    u = np.unique(subject)
    x = np.stack([feats[subject == s].mean(0) for s in u])
    first = np.array([np.where(subject == s)[0][0] for s in u])
    return x, y[first], ds[first]


def probe(x, target):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))
    folds = min(5, np.bincount(np.unique(target, return_inverse=True)[1]).min())
    pred = cross_val_predict(clf, x, target, cv=StratifiedKFold(folds, shuffle=True, random_state=0))
    return {"bal_acc": float(balanced_accuracy_score(target, pred)),
            "chance": 1 / len(np.unique(target)), "n": int(len(target))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/eeg/raw2")
    ap.add_argument("--save", default="results/site_probe.json")
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    report = {}
    for task, (pos, neg, wanted) in R.TASKS.items():
        datasets = [d for d in wanted if (out / "features" / f"{d}.npz").exists()]
        if len(datasets) < 2:
            continue
        x, y, site = subject_table(out, datasets, pos, neg)
        r = {"datasets": datasets, "site": probe(x, site),
             "site_controls": probe(x[y == 0], site[y == 0]), "disease": probe(x, y)}
        report[task] = r
        print(f"{task:<14} " + "  ".join(f"{k} {v['bal_acc']:.2f} (chance {v['chance']:.2f}, n={v['n']})"
                                         for k, v in r.items() if k != "datasets"), flush=True)
    Path(args.save).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
