"""Riemannian tangent-space features (Barachant et al.), filter-bank version.

Per window: band-pass into BANDS, OAS-shrunk spatial covariance per band. Per
subject: Riemannian mean of its window covariances per band. Tangent-space
projection at the Riemannian mean of the TRAINING subjects only, then L2
logistic regression on the concatenated band vectors. Same repeated subject-wise
CV as eeg/experiments/baselines.py, so the numbers are directly comparable.

    python -m eeg.models.riemann --out ~/eeg/raw2
"""
import argparse
import json
from pathlib import Path

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from eeg.data import raw_eeg as R
from eeg.experiments.baselines import WITHIN

SF = 128
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]


def subject_covs(win, subject):
    import mne
    mne.set_log_level("ERROR")
    u = np.unique(subject)
    out = np.zeros((len(u), len(BANDS), win.shape[1], win.shape[1]))
    for bi, (lo, hi) in enumerate(BANDS):
        for si, s in enumerate(u):
            w = mne.filter.filter_data(win[subject == s].astype(np.float64), SF, lo, hi)
            c = Covariances("oas").fit_transform(w)
            out[si, bi] = mean_riemann(c)
    return u, out


class FBTangentLR(BaseEstimator, ClassifierMixin):
    """Tangent space per band (reference = training mean), concatenated, then logistic regression."""

    def __init__(self, C=0.1):
        self.C = C

    def fit(self, covs, y):
        self.ts_ = [TangentSpace(metric="riemann").fit(covs[:, b]) for b in range(covs.shape[1])]
        x = self._x(covs)
        self.sc_ = StandardScaler().fit(x)
        self.lr_ = LogisticRegression(C=self.C, max_iter=5000).fit(self.sc_.transform(x), y)
        self.classes_ = self.lr_.classes_
        return self

    def _x(self, covs):
        return np.concatenate([ts.transform(covs[:, b]) for b, ts in enumerate(self.ts_)], 1)

    def predict_proba(self, covs):
        return self.lr_.predict_proba(self.sc_.transform(self._x(covs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/eeg/raw2")
    ap.add_argument("--save", default="results/riemann.json")
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    report = {}
    cache = {}
    for task, ds in WITHIN:
        pos, neg, _ = R.TASKS[task]
        if ds not in cache:
            _, _, subject, groups, _, win = R.load_features(out, [ds], raw=True)
            u, covs = subject_covs(win, subject)
            cache[ds] = (u, covs, {s: groups[subject == s][0] for s in u})
        u, covs, grp = cache[ds]
        keep = np.array([grp[s] in (pos, neg) for s in u])
        x, y = covs[keep], np.array([int(grp[s] == pos) for s in u[keep]])
        aucs = []
        for r in range(5):
            p = np.zeros(len(y))
            for a, b in StratifiedKFold(5, shuffle=True, random_state=r).split(x, y):
                p[b] = FBTangentLR().fit(x[a], y[a]).predict_proba(x[b])[:, 1]
            aucs.append(roc_auc_score(y, p))
        report[f"{task}:{ds}"] = {"riemann_fb": [float(np.mean(aucs)), float(np.std(aucs))]}
        print(f"{task:<18} {ds:<9} n={len(y):<4} riemann_fb {np.mean(aucs):.3f}±{np.std(aucs):.3f}", flush=True)
    Path(args.save).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
