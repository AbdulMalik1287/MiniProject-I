"""Final per-disease system: frozen members on preprocessing-v2 features, ensemble, calibration, abstention.

Fixed before running (13 Nov 2026), from earlier results only - nothing here is chosen per task:
    protocol   master model for every task: all datasets of the disorder, one StratifiedKFold(5,
               seed 42) over subjects stratified by site x label (depression = Mumtaz only, so the
               same split is plain within-dataset CV)
    features   prep_v2.py windows (60 s skipped when possible, up to 60 evenly spread windows),
               all scale-free so the BrainLat rs-HEP / rs amplitude gap cannot drive scores
    members    lr          logistic regression on subject-median relative band power + coherence +
                           1/f-corrected (periodic) band power, z-scored per site
               riemann_tn  filter-bank tangent space on trace-normalised covariances, re-centred per site
               gatv2       Park-tuned GATv2 on window graphs (pooled_deep.window_gnn)
               gt          Graph Transformer, same training
               labram_ft   LaBraM, last 2 blocks fine-tuned, windows z-scored per recording
    ensemble   mean of the members' ranks
    calibration  cross-fitted Platt scaling on the out-of-fold ensemble score: each subject's
               probability comes from a calibrator fitted on the OTHER folds' subjects
    abstention the model may answer only its most confident subjects (|p - 0.5| largest);
               accuracy reported at 100/90/80/70/60/50% coverage, threshold 0.5

    python final_system.py --out ~/eeg/raw2
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

import pooled_deep as Q
import prep_v2 as V
import riemann as RM
from baselines import lr_probe

TASKS = ["alzheimers", "ftd", "alzheimers_vs_ftd", "mood", "schizophrenia"]
MEMBERS = ["lr", "riemann_tn", "gatv2", "gt", "labram_ft"]
COVERAGE = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]


def calibrate(score, y, folds):
    """Cross-fitted Platt: probabilities for fold k come from a fit on the other folds."""
    p = np.zeros(len(y))
    for a, b in folds:
        m = LogisticRegression().fit(score[a, None], y[a])
        p[b] = m.predict_proba(score[b, None])[:, 1]
    return p


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum(abs(p[idx == k].mean() - y[idx == k].mean()) * (idx == k).mean()
                     for k in range(bins) if (idx == k).any()))


def coverage_table(y, p):
    order = np.argsort(-np.abs(p - 0.5))
    out = {}
    for c in COVERAGE:
        k = max(1, int(round(c * len(y))))
        sel = order[:k]
        out[f"{int(c * 100)}%"] = {"n": int(k), "accuracy": float(((p[sel] >= 0.5) == y[sel]).mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/eeg/raw2")
    ap.add_argument("--tune", default="results/per_disorder/tune.json")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--save", default="results/final_system.json")
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.loads(Path(args.tune).read_text())["best"]["config"]
    save = Path(args.save)
    report = json.loads(save.read_text()) if save.exists() else {}
    import truststore
    truststore.inject_into_ssl()
    from braindecode.models import Labram
    state = {k: v.clone() for k, v in Labram.from_pretrained("braindecode/labram-pretrained", n_times=2000).state_dict().items()}

    for task in args.tasks.split(","):
        rec = report.setdefault(task, {"oof": {}})
        todo = [m for m in MEMBERS if m not in rec["oof"]]  # resumable per member
        if todo:
            # subject tables first (needs windows for covariances), then free them
            u, ys, site, tab, covs = V.subject_table(out / "features_v2", task, "median")
            z = lambda a: (a - a.mean(0)) / (a.std(0) + 1e-8)  # noqa: E731
            x_lr = V.per_site(np.concatenate([tab["ab"], tab["coh"], tab["per"]], 1), site, z)
            covs = covs / np.trace(covs, axis1=-2, axis2=-1)[..., None, None]
            x_rm = V.recentre_sites(covs, site)
            del tab, covs
            d = Q.load_task(out, task, raw="labram_ft" in todo, feat="features_v2")
            assert (d["u"] == u).all() and (d["ys"] == ys).all()
            if d["win"] is not None:
                d["winz"] = Q.recording_z(d)
                d["win"] = None
            g = Q.graphs(d, cfg) if {"gatv2", "gt"} & set(todo) else None
            folds = Q.folds(d)
            print(f"[{task}] {len(u)} subjects", flush=True)
            for m in todo:
                score = np.zeros(len(u))
                for a, b in folds:
                    if m == "lr":
                        score[b] = lr_probe().fit(x_lr[a], ys[a]).predict_proba(x_lr[b])[:, 1]
                    elif m == "riemann_tn":
                        score[b] = RM.FBTangentLR().fit(x_rm[a], ys[a]).predict_proba(x_rm[b])[:, 1]
                    elif m in ("gatv2", "gt"):
                        score[b] = Q.window_gnn("gatv2" if m == "gatv2" else "transformer", cfg, g, d,
                                                u[a], u[b], device)
                    else:
                        score[b] = Q.labram_ft(d, u[a], u[b], device, state)
                rec["oof"][m] = score.tolist()
                rec["per_site_" + m] = Q.summarise(d, score)
                rec["subjects"], rec["y"], rec["site"] = u.tolist(), ys.tolist(), site.tolist()
                tmp = save.with_suffix(".tmp")
                tmp.write_text(json.dumps(report))
                os.replace(tmp, save)
                print(f"  {m:<11} " + "  ".join(f"{s} {r['auc']:.3f}" for s, r in rec["per_site_" + m].items()), flush=True)
            del d, g

        # ensemble + calibration + abstention (cheap; recomputed every time)
        ys, site = np.array(rec["y"]), np.array(rec["site"])
        ens = np.mean([rankdata(rec["oof"][m]) / len(ys) for m in MEMBERS], 0)
        dd = {"u": np.array(rec["subjects"]), "ys": ys, "site": site}
        folds = Q.folds(dd)
        p = calibrate(ens, ys, folds)
        rec["ensemble"] = {"per_site": Q.summarise(dd, ens), "brier": float(brier_score_loss(ys, p)),
                           "ece": ece(ys, p), "coverage": coverage_table(ys, p),
                           "prob": p.tolist()}
        tmp = save.with_suffix(".tmp")
        tmp.write_text(json.dumps(report))
        os.replace(tmp, save)
        e = rec["ensemble"]
        print(f"  ensemble    " + "  ".join(f"{s} {r['auc']:.3f}" for s, r in e["per_site"].items())
              + f"  | Brier {e['brier']:.3f} ECE {e['ece']:.3f} | acc@100% {e['coverage']['100%']['accuracy']:.2f}"
              + f" @70% {e['coverage']['70%']['accuracy']:.2f} @50% {e['coverage']['50%']['accuracy']:.2f}", flush=True)


if __name__ == "__main__":
    main()
