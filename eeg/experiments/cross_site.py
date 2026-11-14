"""External validation between two independent dementia cohorts: Thessaloniki (ds004504) <-> BrainLat.

For each task, a model is trained on every subject of one cohort and tested on every
subject of the other - no subject or hospital is shared. Methods: logistic regression
(band power + coherence), logistic regression (rich markers), filter-bank Riemannian,
frozen LaBraM, and their pre-declared mean-rank ensemble.

Two input variants per method:
    raw         features as extracted
    site_norm   each cohort's subject features z-scored (Riemannian: re-centred) on that cohort's
                own subjects, without labels - removes cohort offsets, uses the test cohort's
                unlabeled data (transductive), and assumes similar patient/control mixes

    python -m eeg.experiments.cross_site --out ~/eeg/raw2
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from eeg.experiments import per_disorder as D
from eeg.data import raw_eeg as R
from eeg.models import riemann as RM
from eeg.experiments.baselines import lr_probe
from eeg.experiments.pooled import recentre, site_z

TASKS = ["alzheimers", "ftd", "alzheimers_vs_ftd"]
COHORTS = ["ds004504", "brainlat"]
METHODS = ["lr_base", "lr_rich", "riemann", "labram"]


def cohort(out, task, ds):
    pos, neg, _ = R.TASKS[task]
    ab, coh, subject, groups, _, win = R.load_features(out, [ds], raw=True)
    keep, y = R.task_rows(groups, pos, neg)
    ab, coh, subject, win = ab[keep], coh[keep], subject[keep], win[keep]
    rich = np.load(out / "features" / f"{ds}_rich.npy")[keep]
    lab = np.load(out / "features" / f"{ds}_labram.npz")["block6"][keep]
    iu = np.triu_indices(coh.shape[1], 1)
    base = np.concatenate([ab.reshape(len(ab), -1), coh[:, iu[0], iu[1], :].reshape(len(coh), -1)], 1)
    u = np.unique(subject)
    rows = {s: np.where(subject == s)[0] for s in u}
    mean = lambda f: np.stack([f[rows[s]].reshape(len(rows[s]), -1).mean(0) for s in u])  # noqa: E731
    _, covs = RM.subject_covs(win, subject)
    return {"lr_base": mean(base), "lr_rich": mean(rich), "riemann": covs, "labram": mean(lab)}, \
        np.array([y[rows[s][0]] for s in u])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/eeg/raw2")
    ap.add_argument("--save", default="results/cross_site.json")
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    report = {}
    for task in TASKS:
        data = {ds: cohort(out, task, ds) for ds in COHORTS}
        for variant in ("raw", "site_norm"):
            for src, dst in (COHORTS, COHORTS[::-1]):
                (xs, ys), (xt, yt) = data[src], data[dst]
                if variant == "site_norm":
                    xs = {m: recentre(v) if m == "riemann" else site_z(v) for m, v in xs.items()}
                    xt = {m: recentre(v) if m == "riemann" else site_z(v) for m, v in xt.items()}
                scores = {}
                for m in METHODS:
                    clf = RM.FBTangentLR() if m == "riemann" else lr_probe()
                    scores[m] = clf.fit(xs[m], ys).predict_proba(xt[m])[:, 1]
                scores["ensemble"] = np.mean([rankdata(scores[m]) / len(yt) for m in METHODS], axis=0)
                res = {}
                for m, sc in scores.items():
                    lo, hi = D.bootstrap_ci(yt, sc, roc_auc_score)
                    res[m] = {"auc": float(roc_auc_score(yt, sc)), "ci": [lo, hi]}
                key = f"{task}:{src}->{dst}:{variant}"
                report[key] = {"n_train": int(len(ys)), "n_test": int(len(yt)), "n_test_pos": int(yt.sum()), **res}
                print(f"{task:<18} {src:>8} -> {dst:<8} {variant:<9} n={len(yt):<3} "
                      + "  ".join(f"{m} {r['auc']:.3f}" for m, r in res.items()), flush=True)
    Path(args.save).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
