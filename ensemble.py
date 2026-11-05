"""Pre-declared ensemble, within each dataset, on one shared subject-wise split.

Members (fixed before seeing any result, identical for every task - choosing the
best member per task by its test score would be selection on the test folds):
    gatv2       Park-tuned GATv2 on relative band power + coherence (per_disorder.py)
    msbp        MSBP-GNN on raw 2 s windows (msbp.py)
    lr_base     logistic regression on subject-mean band power + coherence
    lr_rich     logistic regression on subject-mean rich_features.py markers
    riemann     filter-bank tangent space + logistic regression (riemann.py)
    labram      logistic regression on frozen LaBraM block-6 embeddings (labram_probe.py)
    ensemble    mean of all members' within-fold rank-normalised subject scores

v1 (results/ensemble.json) had the first four members. v2 adds every method
built afterwards - all of them, not the ones that scored well - so membership is
still not chosen by test score.

5-fold StratifiedKFold over subjects (seed 42); every member sees the same folds,
so its out-of-fold subject scores are directly comparable and combinable.

    python ensemble.py --out ~/eeg/raw2
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import msbp as M
import per_disorder as D
import raw_eeg as R
from baselines import WITHIN

MEMBERS = ["gatv2", "msbp", "lr_base", "lr_rich", "riemann", "labram"]


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000))


def run(task, dataset, out, cfg, device):
    pos, neg, _ = R.TASKS[task]
    ab, coh, subject, groups, _, win = R.load_features(out, [dataset], raw=True)
    keep, y = R.task_rows(groups, pos, neg)
    ab, coh, subject, win = ab[keep], coh[keep], subject[keep], win[keep]
    rich = np.load(out / "features" / f"{dataset}_rich.npy")[keep]
    iu = np.triu_indices(coh.shape[1], 1)
    base = np.concatenate([ab.reshape(len(ab), -1), coh[:, iu[0], iu[1], :].reshape(len(coh), -1)], 1)
    u = np.unique(subject)
    rows_of = {s: np.where(subject == s)[0] for s in u}
    ys = np.array([y[rows_of[s][0]] for s in u])
    xb = np.stack([base[rows_of[s]].mean(0) for s in u])
    xr = np.stack([rich[rows_of[s]].reshape(len(rows_of[s]), -1).mean(0) for s in u])
    import riemann as RM
    cu, covs = RM.subject_covs(win, subject)
    assert (cu == u).all()
    emb = np.load(out / "features" / f"{dataset}_labram.npz")["block6"][keep]
    xl = np.stack([emb[rows_of[s]].mean(0) for s in u])
    score = {m: np.zeros(len(u)) for m in MEMBERS + ["ensemble"]}
    for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(xb, ys):
        tr = np.concatenate([rows_of[s] for s in u[a]])
        fold = {}
        g = D.fit(cfg, ab, coh, y, tr, device)
        m = M.fit(win, y, tr, device)
        for name, model_p in (("gatv2", lambda r: D.predict_proba(g, cfg, ab, coh, r, device)),
                              ("msbp", lambda r: M.predict_proba(m, win, r, device))):
            fold[name] = np.array([model_p(rows_of[s]).mean() for s in u[b]])
        fold["lr_base"] = lr().fit(xb[a], ys[a]).predict_proba(xb[b])[:, 1]
        fold["lr_rich"] = lr().fit(xr[a], ys[a]).predict_proba(xr[b])[:, 1]
        fold["riemann"] = RM.FBTangentLR().fit(covs[a], ys[a]).predict_proba(covs[b])[:, 1]
        fold["labram"] = lr().fit(xl[a], ys[a]).predict_proba(xl[b])[:, 1]
        for k, v in fold.items():
            score[k][b] = v
        # rank-normalise within the fold so members on different scales weigh equally
        score["ensemble"][b] = np.mean([rankdata(fold[k]) / len(b) for k in MEMBERS], axis=0)
    res = {}
    for k, v in score.items():
        lo, hi = D.bootstrap_ci(ys, v, roc_auc_score)
        res[k] = {"auc": float(roc_auc_score(ys, v)), "ci": [lo, hi]}
    return res, len(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/eeg/raw2")
    ap.add_argument("--tune", default="results/per_disorder/tune.json")
    ap.add_argument("--save", default="results/ensemble_v2.json")
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    cfg = {**json.loads(Path(args.tune).read_text())["best"]["config"], "epochs": R.RAW_EPOCHS}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save = Path(args.save)
    report = json.loads(save.read_text()) if save.exists() else {}
    for task, ds in WITHIN:
        key = f"{task}:{ds}"
        if key in report:  # resumable: a reaped run picks up where it stopped
            continue
        res, n = run(task, ds, out, cfg, device)
        report[key] = res
        save.write_text(json.dumps(report, indent=2))
        print(f"{task:<18} {ds:<9} n={n:<4}" + "".join(f"  {k} {v['auc']:.3f}" for k, v in res.items()), flush=True)


if __name__ == "__main__":
    main()
