"""Pooled multi-site training, tested within one site.

For a target dataset, test folds are its own subjects (5-fold stratified,
repeated 5x); the training set is the target's other folds PLUS every subject of
the other same-condition datasets of that disorder. Site offsets are removed
first without labels:
    lr_base / lr_rich   per-site z-scoring of subject features
    riemann             per-site re-centring: each site's covariances whitened by
                        that site's Riemannian mean (Zanini et al. 2018), per band
All three use every subject of a site, test subjects included, but never a label,
so this is transductive normalisation, not label leakage. Compare with the
within-site numbers in baselines_*.json / riemann.json.

    python pooled.py --out ~/eeg/raw2
"""
import argparse
import json
from pathlib import Path

import numpy as np
from pyriemann.utils.base import invsqrtm
from pyriemann.utils.mean import mean_riemann
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import raw_eeg as R
import riemann as RM
from baselines import lr_probe, subject_features

TASKS = ["mood", "schizophrenia", "parkinsons"]


def site_z(x):
    return (x - x.mean(0)) / (x.std(0) + 1e-8)


def recentre(covs):
    out = np.empty_like(covs)
    for b in range(covs.shape[1]):
        w = invsqrtm(mean_riemann(covs[:, b]))
        out[:, b] = w @ covs[:, b] @ w
    return out


def site_tables(out, task, ds):
    pos, neg, _ = R.TASKS[task]
    xb, y = subject_features(out, task, ds, "base")
    xr, _ = subject_features(out, task, ds, "rich")
    _, _, subject, groups, _, win = R.load_features(out, [ds], raw=True)
    keep, _ = R.task_rows(groups, pos, neg)
    _, covs = RM.subject_covs(win[keep], subject[keep])
    return {"lr_base": site_z(xb), "lr_rich": site_z(xr), "riemann": recentre(covs)}, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/eeg/raw2")
    ap.add_argument("--save", default="results/pooled.json")
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    report = {}
    for task in TASKS:
        sites = R.TASKS[task][2]
        data = {ds: site_tables(out, task, ds) for ds in sites}
        for target in sites:
            xs, yt = data[target]
            res = {}
            for m in ("lr_base", "lr_rich", "riemann"):
                others = [data[d][0][m] for d in sites if d != target]
                oy = np.concatenate([data[d][1] for d in sites if d != target])
                aucs = []
                for r in range(5):
                    p = np.zeros(len(yt))
                    for a, b in StratifiedKFold(5, shuffle=True, random_state=r).split(xs[m], yt):
                        xtr = np.concatenate([xs[m][a]] + others)
                        ytr = np.concatenate([yt[a], oy])
                        clf = RM.FBTangentLR() if m == "riemann" else lr_probe()
                        p[b] = clf.fit(xtr, ytr).predict_proba(xs[m][b])[:, 1]
                    aucs.append(roc_auc_score(yt, p))
                res[m] = [float(np.mean(aucs)), float(np.std(aucs))]
            report[f"{task}:{target}"] = res
            print(f"{task:<14} {target:<11} n={len(yt):<4} train+{len(oy):<4}"
                  + "".join(f"  {k} {a:.3f}" for k, (a, _) in res.items()), flush=True)
    Path(args.save).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
