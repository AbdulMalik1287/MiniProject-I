"""Three-class AD / FTD / healthy control on ds004504 (team plan E02, comparison target DT-GCN).

All 88 Thessaloniki subjects, one label each. 5 repeats of stratified 5-fold CV
over subjects (seeds 0-4); every method sees the same folds; everything fitted
(scalers, tangent-space reference, models) is fitted on training subjects only.

Methods (fixed before running):
    gatv2      Park-tuned GATv2 with a 3-class head on the window graphs; subject = mean window softmax
    lr_base    multinomial logistic regression, subject-mean band power + coherence
    lr_rich    same on rich_features.py markers
    riemann    filter-bank tangent space + logistic regression (riemann.py)
    labram     logistic regression on frozen LaBraM block-6 embeddings
    rf         random forest, base + rich features
    xgb        XGBoost, base + rich features
    ensemble   mean of all seven members' class probabilities

Metrics, subject level (team plan section 9): macro-F1, balanced accuracy, macro
one-vs-rest AUC, per-class recall, confusion matrix (summed over the 5 repeats).
DT-GCN (Liu & Meng, Information 2026) reports 82% +/- 7% accuracy and 0.79 +/- 0.07
macro-F1 for this task with subject-level 5-fold CV.

    python three_class.py --out ~/eeg/raw2
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader

import park as P
import per_disorder as D
import raw_eeg as R
import riemann as RM
from train import SEED, build_model

CLASSES = ["control", "ftd", "alzheimers"]
METHODS = ["gatv2", "lr_base", "lr_rich", "riemann", "labram", "rf", "xgb"]


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000))


def xgb():
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.5, eval_metric="mlogloss", random_state=0, n_jobs=4)


def gat_fit(cfg, ab, coh, y, idx, device):
    """D.fit with a 3-class head and class weights from the training rows."""
    torch.manual_seed(SEED)
    edge_dim = P.N_BANDS if cfg["edge_mode"] == "bands" else 1
    model = build_model(cfg["model"], edge_dim, n_nodes=P.N_NODES, n_classes=3, normalize=True,
                        in_dim=P.node_feature_dim(cfg["node_features"]), pool=cfg["pool"]).to(device)
    loader = DataLoader(D._dataset(cfg, ab, coh, y, idx), batch_size=cfg["batch_size"], shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    counts = np.bincount(y[idx], minlength=3)
    w = torch.tensor(counts.sum() / (3.0 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    for _ in range(cfg["epochs"]):
        model.train()
        for b in loader:
            b = b.to(device)
            opt.zero_grad()
            nn.functional.cross_entropy(model(b.x, b.edge_index, b.edge_attr, b.batch), b.y, weight=w).backward()
            opt.step()
    return model


@torch.no_grad()
def gat_proba(model, cfg, ab, coh, idx, device):
    model.eval()
    loader = DataLoader(D._dataset(cfg, ab, coh, np.zeros(len(ab), dtype=np.int64), idx),
                        batch_size=256, shuffle=False)
    out = [torch.softmax(model(b.x, b.edge_index, b.edge_attr, b.batch).double(), 1).cpu().numpy()
           for b in (bb.to(device) for bb in loader)]
    return np.concatenate(out)


def metrics(y, prob):
    pred = prob.argmax(1)
    return {"macro_f1": float(f1_score(y, pred, average="macro")),
            "bal_acc": float(balanced_accuracy_score(y, pred)),
            "accuracy": float(accuracy_score(y, pred)),
            "macro_auc_ovr": float(roc_auc_score(y, prob, multi_class="ovr", average="macro")),
            "recall": dict(zip(CLASSES, recall_score(y, pred, average=None, labels=[0, 1, 2]).round(4).tolist())),
            "confusion": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/eeg/raw2")
    ap.add_argument("--tune", default="results/per_disorder/tune.json")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--dataset", default="ds004504", help="ds004504 or brainlat")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = {**json.loads(Path(args.tune).read_text())["best"]["config"], "epochs": R.RAW_EPOCHS}

    ab, coh, subject, groups, _, win = R.load_features(out, [args.dataset], raw=True)
    yw = np.array([CLASSES.index(g) for g in groups])
    rich = np.load(out / "features" / f"{args.dataset}_rich.npy")
    lab = np.load(out / "features" / f"{args.dataset}_labram.npz")["block6"]
    iu = np.triu_indices(coh.shape[1], 1)
    base = np.concatenate([ab.reshape(len(ab), -1), coh[:, iu[0], iu[1], :].reshape(len(coh), -1)], 1)
    u = np.unique(subject)
    rows_of = {s: np.where(subject == s)[0] for s in u}
    ys = np.array([yw[rows_of[s][0]] for s in u])
    mean = lambda f: np.stack([f[rows_of[s]].reshape(len(rows_of[s]), -1).mean(0) for s in u])  # noqa: E731
    xb, xr, xl = mean(base), mean(rich), mean(lab)
    xbr = np.concatenate([xb, xr], 1)
    cu, covs = RM.subject_covs(win, subject)
    assert (cu == u).all()
    print(f"[3-class] {len(u)} subjects: " + ", ".join(f"{c} {int((ys == i).sum())}" for i, c in enumerate(CLASSES)),
          flush=True)

    runs = {m: [] for m in METHODS + ["ensemble"]}
    pooled = {m: [] for m in runs}  # per-repeat (y, prob) for the summed confusion matrix
    for rep in range(args.repeats):
        prob = {m: np.zeros((len(u), 3)) for m in METHODS}
        for a, b in StratifiedKFold(5, shuffle=True, random_state=rep).split(xb, ys):
            tr = np.concatenate([rows_of[s] for s in u[a]])
            g = gat_fit(cfg, ab, coh, yw, tr, device)
            wp = gat_proba(g, cfg, ab, coh, np.concatenate([rows_of[s] for s in u[b]]), device)
            k = 0
            for i, s in enumerate(u[b]):
                n = len(rows_of[s])
                prob["gatv2"][b[i]] = wp[k:k + n].mean(0)
                k += n
            prob["lr_base"][b] = lr().fit(xb[a], ys[a]).predict_proba(xb[b])
            prob["lr_rich"][b] = lr().fit(xr[a], ys[a]).predict_proba(xr[b])
            prob["riemann"][b] = RM.FBTangentLR().fit(covs[a], ys[a]).predict_proba(covs[b])
            prob["labram"][b] = lr().fit(xl[a], ys[a]).predict_proba(xl[b])
            prob["rf"][b] = RandomForestClassifier(500, class_weight="balanced", random_state=0,
                                                   n_jobs=4).fit(xbr[a], ys[a]).predict_proba(xbr[b])
            prob["xgb"][b] = xgb().fit(xbr[a], ys[a]).predict_proba(xbr[b])
        prob["ensemble"] = np.mean([prob[m] for m in METHODS], axis=0)
        for m, p in prob.items():
            runs[m].append(metrics(ys, p))
            pooled[m].append((ys, p))
        print(f"[rep {rep}] " + "  ".join(f"{m} F1 {runs[m][-1]['macro_f1']:.3f}" for m in runs), flush=True)

    report = {"n": {c: int((ys == i).sum()) for i, c in enumerate(CLASSES)}, "repeats": args.repeats,
              "reference": {"DT-GCN (Liu & Meng 2026)": {"accuracy": "0.82 +/- 0.07", "macro_f1": "0.79 +/- 0.07"}},
              "methods": {}}
    for m, rs in runs.items():
        agg = {k: [float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs]))]
               for k in ("macro_f1", "bal_acc", "accuracy", "macro_auc_ovr")}
        agg["recall"] = {c: float(np.mean([r["recall"][c] for r in rs])) for c in CLASSES}
        agg["confusion_summed"] = np.sum([r["confusion"] for r in rs], axis=0).tolist()
        report["methods"][m] = agg
    report["dataset"] = args.dataset
    Path(args.save or ("results/three_class.json" if args.dataset == "ds004504" else f"results/three_class_{args.dataset}.json")).write_text(json.dumps(report, indent=2))
    print("\nmethod      macro-F1        bal-acc         accuracy        macro-AUC   recall ctl/ftd/ad")
    for m, a in report["methods"].items():
        print(f"{m:<10} " + "  ".join(f"{a[k][0]:.3f}±{a[k][1]:.3f}" for k in ("macro_f1", "bal_acc", "accuracy", "macro_auc_ovr"))
              + "   " + "/".join(f"{a['recall'][c]:.2f}" for c in CLASSES))


if __name__ == "__main__":
    main()
