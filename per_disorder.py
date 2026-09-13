#!/usr/bin/env python3
"""One GNN per psychiatric disorder, and an ensemble that reports all of them.

Each disorder gets its own binary graph neural network (that disorder vs
healthy controls), trained on 19-channel EEG graphs from the Park et al. 2021
corpus. The ensemble runs every model on a subject and reports a probability
per disorder.

Protocol, in the order the commands must run:

    python per_disorder.py tune       # pick ONE GNN config by CV on the train split
    python per_disorder.py train      # fit one model per disorder on the train split
    python per_disorder.py evaluate   # score everything on the untouched test split
    python per_disorder.py predict --ids 12 57   # probabilities for given subjects

The test split is fixed once (split.json) and never seen by tune or train, so
the evaluate numbers are not inflated by config selection. One config is chosen
for all six models by mean AUC across disorders; choosing per disorder on folds
of 90-230 subjects would mostly fit noise.
"""
import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.loader import DataLoader

import park as P
from train import SEED, build_model

HEALTHY = "Healthy control"
DISORDERS = ["Schizophrenia", "Mood disorder", "Addictive disorder",
             "Trauma and stress related disorder", "Anxiety disorder"]
# OCD has 46 subjects in the whole corpus - 9 in the test split against 19
# controls, a 95% CI of roughly +/-0.22 on AUC. Excluded from modelling and
# from evaluation rather than reported as an uninterpretable number. Its rows
# stay in split.json so the split is unchanged.
EXCLUDED = ["Obsessive compulsive disorder"]
ALL_CLASSES = [HEALTHY] + DISORDERS  # index 0 = healthy, 1.. = DISORDERS
SHORT = {"Schizophrenia": "schizophrenia", "Mood disorder": "mood",
         "Addictive disorder": "addictive", "Trauma and stress related disorder": "trauma",
         "Anxiety disorder": "anxiety", "Obsessive compulsive disorder": "ocd",
         HEALTHY: "healthy"}

EDGE_INDEX = torch.tensor(list(product(range(P.N_NODES), range(P.N_NODES))),
                          dtype=torch.long).t().contiguous()

# Everything held fixed across the search. profile node features were the
# clear winner in the node-feature grid, so they are not re-searched here.
FIXED = {"node_features": "profile", "edge_mode": "bands", "epochs": 150,
         "batch_size": 64, "lr": 1e-3, "weight_decay": 5e-4}
GRID = [{**FIXED, "model": m, "pool": p, "topk": k}
        for m in ("gcn", "gatv2") for p in ("add", "meanmax") for k in (None, 4)]


# --------------------------------------------------------------------------- data


def load_all(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    unknown = set(df["main.disorder"]) - set(ALL_CLASSES) - set(EXCLUDED)
    if unknown:
        raise ValueError(f"unexpected labels: {unknown}")
    ids = df["no."].astype(str).to_numpy()
    if len(set(ids)) != len(ids):
        raise ValueError("subject ids in 'no.' are not unique; the split would leak")
    ab, coh = P.features_from_df(df)
    return ab, coh, df["main.disorder"].to_numpy(), ids


def make_split(labels, ids, path, test_size=0.2):
    """Stratified train/test split over subjects, written once and then reused."""
    pos = {s: i for i, s in enumerate(ids)}
    if path.exists():
        s = json.loads(path.read_text())
        tr = np.array([pos[i] for i in s["train_ids"]])
        te = np.array([pos[i] for i in s["test_ids"]])
    else:
        tr, te = train_test_split(np.arange(len(ids)), test_size=test_size,
                                  stratify=labels, random_state=SEED)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"seed": SEED, "test_size": test_size,
                                    "train_ids": ids[tr].tolist(),
                                    "test_ids": ids[te].tolist()}, indent=1))
    if set(tr) & set(te):
        raise ValueError("a subject is in both train and test")
    keep = lambda idx: idx[~np.isin(labels[idx], EXCLUDED)]  # noqa: E731
    return np.sort(keep(tr)), np.sort(keep(te))


def binary_rows(labels, pool, disorder):
    """Rows of `pool` that are `disorder` or healthy, and 0/1 labels over ALL rows.

    The healthy controls in the train split are shared by every disorder's
    model - with 95 controls in the whole corpus there is no alternative. The
    test-split controls stay unseen by all of them.
    """
    rows = pool[np.isin(labels[pool], [disorder, HEALTHY])]
    return rows, (labels == disorder).astype(np.int64)


# ------------------------------------------------------------------------ modelling


def make_model(cfg):
    edge_dim = P.N_BANDS if cfg["edge_mode"] == "bands" else 1
    return build_model(cfg["model"], edge_dim, n_nodes=P.N_NODES, n_classes=2,
                       normalize=True, in_dim=P.node_feature_dim(cfg["node_features"]),
                       pool=cfg["pool"])


def _dataset(cfg, ab, coh, y, idx):
    return P.ParkGraphDataset(ab, coh, y, idx, EDGE_INDEX, cfg["edge_mode"],
                              cfg["node_features"], cfg["topk"])


def fit(cfg, ab, coh, y, idx, device):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    loader = DataLoader(_dataset(cfg, ab, coh, y, idx), batch_size=cfg["batch_size"],
                        shuffle=True)
    model = make_model(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    counts = np.bincount(y[idx], minlength=2)
    weight = torch.tensor(counts.sum() / (2.0 * np.maximum(counts, 1)),
                          dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=weight)
    for _ in range(cfg["epochs"]):
        model.train()
        for b in loader:
            b = b.to(device)
            opt.zero_grad()
            crit(model(b.x, b.edge_index, b.edge_attr, b.batch), b.y).backward()
            opt.step()
    return model


@torch.no_grad()
def predict_margin(model, cfg, ab, coh, idx, device):
    """Logit margin z_disorder - z_healthy for each row in idx, in float64.

    softmax in float32 rounds to exactly 1.0 once the margin passes ~17, and
    these small overfit models hit that for a large share of subjects. Ties at
    1.0 erase ranking (distorting AUC) and make calibration impossible, so all
    scoring goes through the unsaturated margin.
    """
    dummy = np.zeros(len(ab), dtype=np.int64)
    loader = DataLoader(_dataset(cfg, ab, coh, dummy, idx), batch_size=256, shuffle=False)
    model.eval()
    out = []
    for b in loader:
        b = b.to(device)
        z = model(b.x, b.edge_index, b.edge_attr, b.batch).double()
        out.append((z[:, 1] - z[:, 0]).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def sigmoid(m):
    return 1 / (1 + np.exp(-m))


def predict_proba(model, cfg, ab, coh, idx, device):
    """P(disorder) for each row in idx - equal to softmax[:, 1], computed without float32 ties."""
    return sigmoid(predict_margin(model, cfg, ab, coh, idx, device))


def binary_metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return {"auc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan"),
            "accuracy": accuracy_score(y, pred),
            "bal_acc": balanced_accuracy_score(y, pred),
            "f1": f1_score(y, pred, zero_division=0),
            "n": int(len(y)), "n_pos": int(y.sum())}


def bootstrap_ci(y, s, stat, b=2000, seed=SEED):
    """95% percentile bootstrap CI of stat(y, s), resampling subjects.

    The test split is 189 subjects with only 19 healthy controls, so a single
    point estimate hides very wide uncertainty; this makes it visible.
    """
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(b):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) >= 2:
            vals.append(stat(y[i], s[i]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def decide(probs, thr=0.5):
    """Ensemble rule over (n, 6) disorder probabilities -> index into ALL_CLASSES.

    The most confident disorder model wins if it clears thr, otherwise healthy.
    The six models were never trained against each other, so this rule is only
    as good as their calibration - evaluate reports that directly.
    """
    return np.where(probs.max(1) >= thr, probs.argmax(1) + 1, 0)


def load_models(model_dir, device):
    models = {}
    for d in DISORDERS:
        f = Path(model_dir) / f"{SHORT[d]}.pt"
        if not f.exists():
            raise FileNotFoundError(f"{f} missing - run `per_disorder.py train` first")
        ck = torch.load(f, map_location=device, weights_only=False)
        m = make_model(ck["config"]).to(device)
        m.load_state_dict(ck["state_dict"])
        models[d] = (m, ck["config"])
    return models


def fit_platt(y, margin):
    """Platt scaling on the logit margin, fitted with BALANCED class weights.

    The five models saw very different case:control ratios (mood 213:76,
    anxiety 85:76). Calibrating each to its own ratio would keep the
    high-prevalence models inflated; a balanced prior makes their
    probabilities comparable before the ensemble compares them.
    """
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, class_weight="balanced", max_iter=1000)
    lr.fit(np.asarray(margin, dtype=np.float64).reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def apply_platt(margin, a, b):
    return sigmoid(a * margin + b)


def load_calibration(model_dir):
    f = Path(model_dir) / "calibration.json"
    return json.loads(f.read_text()) if f.exists() else None


def calibrated_probs(models, calib, ab, coh, idx, device):
    """(raw, reported) (n, 5) probabilities; reported == raw without calibration.json."""
    margins = np.stack([predict_margin(m, c, ab, coh, idx, device) for m, c in models.values()], 1)
    raw = sigmoid(margins)
    if calib is None:
        return raw, raw
    cal = np.stack([apply_platt(margins[:, j], calib[d]["a"], calib[d]["b"])
                    for j, d in enumerate(DISORDERS)], 1)
    return raw, cal


# ------------------------------------------------------------------------ commands


def cmd_tune(args, ab, coh, labels, ids, device):
    """CV each grid config on the train split. Each config writes its own part
    file, so tuning is resumable and `--grid` lets several processes split it."""
    tr, _ = make_split(labels, ids, args.out / "split.json")
    parts = args.out / "tune_parts"
    parts.mkdir(exist_ok=True)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    for g in (args.grid if args.grid is not None else range(len(GRID))):
        part, cfg = parts / f"cfg_{g}.json", GRID[g]
        tag = f"{cfg['model']}/{cfg['pool']}/topk={cfg['topk']}"
        if part.exists():
            print(f"[tune] {tag:<28} already done, skipping", flush=True)
            continue
        t0, per = time.time(), {}
        for d in DISORDERS:
            rows, y = binary_rows(labels, tr, d)
            oof = np.zeros(len(rows))
            for a, b in skf.split(rows, y[rows]):
                model = fit(cfg, ab, coh, y, rows[a], device)
                oof[b] = predict_proba(model, cfg, ab, coh, rows[b], device)
            per[d] = roc_auc_score(y[rows], oof)  # pooled out-of-fold AUC
        mean = float(np.mean(list(per.values())))
        part.write_text(json.dumps({"grid_index": g, "config": cfg, "mean_auc": mean,
                                    "per_disorder_auc": per}, indent=2))
        print(f"[tune] {tag:<28} mean AUC {mean:.3f}   " +
              "  ".join(f"{SHORT[d]}={v:.3f}" for d, v in per.items()) +
              f"   ({time.time() - t0:.0f}s)", flush=True)

    results = [json.loads(p.read_text()) for p in sorted(parts.glob("cfg_*.json"))]
    if len(results) < len(GRID):
        print(f"[tune] {len(results)}/{len(GRID)} configs done; tune.json written once all finish")
        return
    # Select on the modelled disorders only. Part files written before OCD was
    # excluded still carry its AUC; recomputing here keeps it out of selection.
    for r in results:
        r["mean_auc"] = float(np.mean([r["per_disorder_auc"][d] for d in DISORDERS]))
        print(f"[tune] {r['config']['model']}/{r['config']['pool']}/topk={r['config']['topk']}"
              f"  mean AUC (excl. OCD) {r['mean_auc']:.3f}")
    best = max(results, key=lambda r: r["mean_auc"])
    (args.out / "tune.json").write_text(json.dumps({"results": results, "best": best}, indent=2))
    print(f"\n[tune] best: {best['config']['model']}/{best['config']['pool']}/"
          f"topk={best['config']['topk']}  mean CV AUC {best['mean_auc']:.3f}")


def cmd_train(args, ab, coh, labels, ids, device):
    tr, _ = make_split(labels, ids, args.out / "split.json")
    tune = args.out / "tune.json"
    if not tune.exists():
        raise FileNotFoundError("tune.json missing - run `per_disorder.py tune` first")
    cfg = json.loads(tune.read_text())["best"]["config"]
    model_dir = args.out / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for d in DISORDERS:
        rows, y = binary_rows(labels, tr, d)
        model = fit(cfg, ab, coh, y, rows, device)
        torch.save({"state_dict": model.state_dict(), "config": cfg, "disorder": d,
                    "channels": P.CHANNELS, "bands": P.BANDS,
                    "n_train": int(len(rows)), "n_train_pos": int(y[rows].sum())},
                   model_dir / f"{SHORT[d]}.pt")
        print(f"[train] {d:<36} {int(y[rows].sum()):>3} cases + "
              f"{int(len(rows) - y[rows].sum()):>3} controls -> models/{SHORT[d]}.pt", flush=True)


def cmd_calibrate(args, ab, coh, labels, ids, device):
    """Out-of-fold predictions on the TRAIN split -> one Platt scaler per model.

    Uses the same folds and config as tuning. The test split is untouched, so
    calibration cannot leak test information into evaluate.
    """
    tr, _ = make_split(labels, ids, args.out / "split.json")
    cfg = json.loads((args.out / "tune.json").read_text())["best"]["config"]
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    calib = {}
    for d in DISORDERS:
        rows, y = binary_rows(labels, tr, d)
        oof = np.zeros(len(rows))
        for a, b in skf.split(rows, y[rows]):
            model = fit(cfg, ab, coh, y, rows[a], device)
            oof[b] = predict_margin(model, cfg, ab, coh, rows[b], device)
        a_, b_ = fit_platt(y[rows], oof)
        if a_ <= 0:
            raise ValueError(f"{d}: calibration slope {a_:.3f} <= 0, scores are inverted")
        p = sigmoid(oof)
        sat = float(np.mean((p > 0.99) | (p < 0.01)))
        calib[d] = {"a": a_, "b": b_, "oof_auc": float(roc_auc_score(y[rows], oof)),
                    "oof_saturated": sat}
        print(f"[calibrate] {SHORT[d]:<14} slope {a_:.3f} intercept {b_:+.3f}  "
              f"OOF AUC {calib[d]['oof_auc']:.3f}  saturated {sat:.0%}", flush=True)
    (args.out / "models" / "calibration.json").write_text(json.dumps(calib, indent=2))


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def cmd_evaluate(args, ab, coh, labels, ids, device):
    _, te = make_split(labels, ids, args.out / "split.json")
    models = load_models(args.out / "models", device)
    calib = load_calibration(args.out / "models")
    raw, probs = calibrated_probs(models, calib, ab, coh, te, device)
    true = np.array([ALL_CLASSES.index(l) for l in labels[te]])

    report = {"per_disorder": {}, "n_test": int(len(te)), "calibrated": calib is not None}
    print(f"=== held-out test split: {len(te)} subjects, never used for tuning or training ===")
    print(f"probabilities: {'Platt-calibrated on train-split out-of-fold predictions' if calib else 'RAW (no calibration.json)'}\n")
    print(f"{'model':<14} {'vs healthy':>38}   {'vs other disorders':>22}")
    print(f"{'':<14} {'AUC [95% CI]':>21} {'acc':>6} {'bal_acc':>8} {'cases':>6}"
          f"   {'AUC [95% CI]':>21}")
    for j, d in enumerate(DISORDERS):
        k = ALL_CLASSES.index(d)
        mh = np.isin(true, [k, 0])  # this disorder vs healthy: the task it was trained on
        yh = (true[mh] == k).astype(int)
        m = binary_metrics(yh, probs[mh, j])
        m["auc_ci"] = bootstrap_ci(yh, probs[mh, j], roc_auc_score)
        # this disorder vs every OTHER disorder: does the model detect *this*
        # condition, or just "patient"? ~0.5 means it cannot tell them apart.
        mo = true != 0
        yo = (true[mo] == k).astype(int)
        m["auc_vs_other_disorders"] = roc_auc_score(yo, probs[mo, j])
        m["auc_vs_other_ci"] = bootstrap_ci(yo, probs[mo, j], roc_auc_score)
        m["brier_raw"], m["brier"] = brier(yh, raw[mh, j]), brier(yh, probs[mh, j])
        report["per_disorder"][d] = m
        ci, co = m["auc_ci"], m["auc_vs_other_ci"]
        print(f"{SHORT[d]:<14} {m['auc']:>6.3f} [{ci[0]:.2f}, {ci[1]:.2f}] {m['accuracy']:>6.3f} "
              f"{m['bal_acc']:>8.3f} {m['n_pos']:>6}   "
              f"{m['auc_vs_other_disorders']:>6.3f} [{co[0]:.2f}, {co[1]:.2f}]")
    print("\nA CI spanning 0.5 means that result is not distinguishable from chance on this test set.")
    if calib:
        print("Brier score vs healthy (lower is better), raw -> calibrated: " + "  ".join(
            f"{SHORT[d]} {report['per_disorder'][d]['brier_raw']:.3f}->"
            f"{report['per_disorder'][d]['brier']:.3f}" for d in DISORDERS))
    report["saturated_raw"] = float(np.mean((raw > 0.99) | (raw < 0.01)))
    report["saturated"] = float(np.mean((probs > 0.99) | (probs < 0.01)))
    print(f"probabilities >0.99 or <0.01: raw {report['saturated_raw']:.0%}, "
          f"reported {report['saturated']:.0%}")

    act = np.array([probs[true == i].mean(0) if (true == i).any() else np.full(6, np.nan)
                    for i in range(len(ALL_CLASSES))])
    report["activation"] = act.tolist()
    print("\n=== mean P(disorder) by true class — a specific model lights up only its own row ===")
    print(f"{'true class':<14}" + "".join(f"{SHORT[d]:>14}" for d in DISORDERS))
    for i, c in enumerate(ALL_CLASSES):
        print(f"{SHORT[c]:<14}" + "".join(f"{v:>14.2f}" for v in act[i]))

    pred = decide(probs, args.threshold)
    cm = confusion_matrix(true, pred, labels=range(len(ALL_CLASSES)))
    majority = np.bincount(true).max() / len(true)
    report["ensemble"] = {"threshold": args.threshold,
                          "accuracy": accuracy_score(true, pred),
                          "bal_acc": balanced_accuracy_score(true, pred),
                          "bal_acc_ci": bootstrap_ci(true, pred, balanced_accuracy_score),
                          "majority_rate": float(majority),
                          "confusion": cm.tolist(), "classes": ALL_CLASSES}
    print(f"\n=== ensemble decision (max P >= {args.threshold} wins, else healthy) ===")
    corner = "true \\ pred"  # backslash outside the f-string: Python <3.12 rejects it inside
    print(f"{corner:<14}" + "".join(f"{SHORT[c]:>14}" for c in ALL_CLASSES))
    for i, c in enumerate(ALL_CLASSES):
        print(f"{SHORT[c]:<14}" + "".join(f"{v:>14d}" for v in cm[i]))
    e = report["ensemble"]
    print(f"\naccuracy {e['accuracy']:.3f}   balanced accuracy {e['bal_acc']:.3f} "
          f"[{e['bal_acc_ci'][0]:.2f}, {e['bal_acc_ci'][1]:.2f}]   "
          f"(majority-class rate {majority:.3f}, {len(ALL_CLASSES)}-class chance bal_acc "
          f"{1 / len(ALL_CLASSES):.3f})")
    (args.out / "evaluate.json").write_text(json.dumps(report, indent=2))


def cmd_predict(args, device):
    df = pd.read_csv(args.csv, low_memory=False)
    if args.ids:
        df = df[df["no."].astype(str).isin(args.ids)].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"no rows with no. in {args.ids}")
    ab, coh = P.features_from_df(df)
    models = load_models(args.out / "models", device)
    calib = load_calibration(args.out / "models")
    idx = np.arange(len(df))
    _, probs = calibrated_probs(models, calib, ab, coh, idx, device)
    pred = decide(probs, args.threshold)
    if calib is None:
        print("NOTE: no calibration.json - probabilities are raw and overconfident")

    split = args.out / "split.json"
    train_ids = set(json.loads(split.read_text())["train_ids"]) if split.exists() else set()
    has_label = "main.disorder" in df.columns
    for i in idx:
        sid = str(df.loc[i, "no."]) if "no." in df.columns else str(i)
        note = "  [WARNING: training subject, not a fair test]" if sid in train_ids else ""
        truth = f"  (recorded diagnosis: {df.loc[i, 'main.disorder']})" if has_label else ""
        print(f"\nsubject {sid}{truth}{note}")
        for j in np.argsort(-probs[i]):
            bar = "#" * int(round(probs[i, j] * 30))
            print(f"  {DISORDERS[j]:<36} {probs[i, j]:.3f} {bar}")
        print(f"  -> {ALL_CLASSES[pred[i]]}")


# ---------------------------------------------------------------------- self-check


def selfcheck():
    rng = np.random.default_rng(0)
    n = 140
    ab = rng.random((n, P.N_NODES, P.N_BANDS)).astype(np.float32)
    coh = rng.random((n, P.N_NODES, P.N_NODES, P.N_BANDS)).astype(np.float32)
    coh = (coh + coh.transpose(0, 2, 1, 3)) / 2
    pool_labels = ALL_CLASSES + EXCLUDED
    labels = np.array([pool_labels[i % len(pool_labels)] for i in range(n)])
    ids = np.array([str(1000 + i) for i in range(n)])

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        sp = Path(tmp) / "split.json"
        tr, te = make_split(labels, ids, sp)
        n_excl = int(np.isin(labels, EXCLUDED).sum())
        assert not set(tr) & set(te) and len(tr) + len(te) == n - n_excl
        assert not np.isin(labels[np.r_[tr, te]], EXCLUDED).any(), "excluded class leaked in"
        tr2, te2 = make_split(labels, ids, sp)  # re-read must give the same split
        assert np.array_equal(tr, tr2) and np.array_equal(te, te2), "split not stable"
        for c in ALL_CLASSES:  # stratified: every class on both sides
            assert (labels[te] == c).any() and (labels[tr] == c).any(), c

    rows, y = binary_rows(labels, tr, "Schizophrenia")
    assert set(labels[rows]) == {"Schizophrenia", HEALTHY}, set(labels[rows])
    assert set(rows) <= set(tr), "binary rows must come from the given pool only"
    assert y[rows].sum() == (labels[rows] == "Schizophrenia").sum()

    device = torch.device("cpu")
    for cfg in (GRID[0], GRID[-1]):  # gcn/add/complete and gatv2/meanmax/topk
        small = {**cfg, "epochs": 2}
        model = fit(small, ab, coh, y, rows, device)
        p = predict_proba(model, small, ab, coh, te, device)
        assert p.shape == (len(te),) and ((p >= 0) & (p <= 1)).all(), (cfg, p)

    assert len(DISORDERS) == 5 and "Obsessive compulsive disorder" not in ALL_CLASSES
    probs = np.array([[0.1, 0.2, 0.9, 0.1, 0.1],   # clear addictive
                      [0.3, 0.2, 0.1, 0.4, 0.2],   # nothing clears 0.5
                      [0.6, 0.7, 0.1, 0.1, 0.1]])  # mood beats schizophrenia
    assert decide(probs).tolist() == [3, 0, 2], decide(probs).tolist()
    assert ALL_CLASSES[3] == "Addictive disorder" and ALL_CLASSES[0] == HEALTHY

    m = binary_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9]))
    assert m["auc"] == 1.0 and m["accuracy"] == 1.0, m

    # Bootstrap CI must bracket the point estimate, and shrink as n grows.
    def ci_width(nn):
        yy = np.r_[np.zeros(nn), np.ones(nn)].astype(int)
        ss = yy * 0.3 + np.random.default_rng(1).random(2 * nn)
        lo, hi = bootstrap_ci(yy, ss, roc_auc_score, b=400)
        assert lo <= roc_auc_score(yy, ss) <= hi, (lo, hi)
        return hi - lo
    assert ci_width(20) > ci_width(400), "CI must narrow with more subjects"

    # Calibration: an overconfident but well-ranked scorer must get a slope
    # below 1 (softening), keep its ranking, and pull saturated outputs inward.
    rng2 = np.random.default_rng(3)
    yy = np.r_[np.zeros(300), np.ones(300)].astype(int)
    true_logit = (yy * 2 - 1) * 0.8 + rng2.normal(0, 1, 600)
    over_m = 6 * true_logit                             # same ranking, 6x too confident
    over = sigmoid(over_m)
    a_, b_ = fit_platt(yy, over_m)
    assert 0 < a_ < 0.5, a_
    cal = apply_platt(over_m, a_, b_)
    assert np.isclose(roc_auc_score(yy, cal), roc_auc_score(yy, over_m)), "calibration changed ranking"
    assert np.mean((cal > 0.99) | (cal < 0.01)) < np.mean((over > 0.99) | (over < 0.01))
    assert brier(yy, cal) < brier(yy, over), "calibration must improve Brier on overconfident scores"
    # float32 softmax ties at 1.0 are exactly what the margin route avoids.
    z = torch.tensor(np.stack([np.zeros(4), [18.0, 19.0, 20.0, 21.0]], 1), dtype=torch.float32)
    assert (F.softmax(z, 1)[:, 1] == 1.0).all(), "expected float32 saturation in this example"
    assert len(np.unique(sigmoid(np.array([18.0, 19.0, 20.0, 21.0])))) == 4, "margin route must keep ranks"
    # Balanced prior: with 3:1 cases, a zero-information score must land near 0.5, not 0.75.
    yb = np.r_[np.zeros(100), np.ones(300)].astype(int)
    flat = np.full(400, 2.2) + rng2.normal(0, 1e-3, 400)
    a_b, b_b = fit_platt(yb, flat)
    assert abs(apply_platt(np.array([2.2]), a_b, b_b)[0] - 0.5) < 0.1, "prior not balanced"
    print("selfcheck ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["tune", "train", "calibrate", "evaluate", "predict"])
    ap.add_argument("--data", default="~/eeg/park/park.csv")
    ap.add_argument("--out", default="~/eeg/per_disorder")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--grid", nargs="*", type=int, default=None,
                    help="tune: only these GRID indices (run several processes in parallel)")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--csv", default=None, help="predict: Park-format CSV (default --data)")
    ap.add_argument("--ids", nargs="*", default=None, help="predict: subject 'no.' values")
    args = ap.parse_args()
    args.out = Path(args.out).expanduser()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.threads:
        torch.set_num_threads(args.threads)

    if args.command == "predict":
        args.csv = Path(args.csv or args.data).expanduser()
        return cmd_predict(args, device)
    ab, coh, labels, ids = load_all(Path(args.data).expanduser())
    {"tune": cmd_tune, "train": cmd_train, "calibrate": cmd_calibrate,
     "evaluate": cmd_evaluate}[args.command](
        args, ab, coh, labels, ids, device)


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else main()
