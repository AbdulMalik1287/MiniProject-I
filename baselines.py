"""Classical baselines on the same features and subject-wise folds as the GNNs.

One row per subject (mean over its windows): log relative band power (19x6) and
band coherence (171x6). Models: L2 logistic regression, RBF SVM, gradient
boosting. 5-fold stratified CV over subjects, repeated 5 times; AUC is the mean
over repeats. A subject is a row, so no window of a test subject can be trained on.

    python baselines.py --out ~/eeg/raw2
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import raw_eeg as R

def lr_probe():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000))


MODELS = {
    "logreg": lr_probe,
    "svm": lambda: make_pipeline(StandardScaler(), SVC(C=1.0, probability=True, random_state=0)),
    "gboost": lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3,
                                                     random_state=0),
}
WITHIN = [("alzheimers", "ds004504"), ("ftd", "ds004504"), ("alzheimers_vs_ftd", "ds004504"),
          ("mood", "mumtaz"), ("mood", "ds003478"), ("schizophrenia", "aseeg"),
          ("schizophrenia", "warsaw"), ("parkinsons", "ds004584"), ("parkinsons", "ds003490"),
          ("parkinsons", "ds002778"), ("parkinsons", "ds007526")]


def subject_features(out, task, dataset, kind="base"):
    pos, neg, _ = R.TASKS[task]
    ab, coh, subject, groups, _, _ = R.load_features(out, [dataset])
    keep, y = R.task_rows(groups, pos, neg)
    ab, coh, subject = ab[keep], coh[keep], subject[keep]
    iu = np.triu_indices(coh.shape[1], 1)
    parts = []
    if kind in ("base", "both"):
        parts += [ab.reshape(len(ab), -1), coh[:, iu[0], iu[1], :].reshape(len(coh), -1)]
    if kind in ("rich", "both"):  # rich_features.py markers, (windows, 19, 15)
        rich = np.load(out / "features" / f"{dataset}_rich.npy")[keep]
        parts.append(rich.reshape(len(rich), -1))
    f = np.concatenate(parts, 1)
    u = np.unique(subject)
    return np.stack([f[subject == s].mean(0) for s in u]), np.array([y[subject == s][0] for s in u])


def cv_auc(make, x, y, repeats=5):
    aucs = []
    for r in range(repeats):
        p = np.zeros(len(y))
        for a, b in RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=r).split(x, y):
            p[b] = make().fit(x[a], y[a]).predict_proba(x[b])[:, 1]
        aucs.append(roc_auc_score(y, p))
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/eeg/raw2")
    ap.add_argument("--features", choices=["base", "rich", "both"], default="base")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    report = {}
    for task, ds in WITHIN:
        x, y = subject_features(out, task, ds, args.features)
        r = {m: cv_auc(f, x, y) for m, f in MODELS.items()}
        report[f"{task}:{ds}"] = r
        print(f"{task:<18} {ds:<9} n={len(y):<4}" + "".join(f"  {m} {a:.3f}±{s:.3f}" for m, (a, s) in r.items()),
              flush=True)
    Path(args.save or f"results/baselines_{args.features}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
