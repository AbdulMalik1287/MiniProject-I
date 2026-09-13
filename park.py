#!/usr/bin/env python3
"""Multi-disorder EEG classification on the Park et al. 2021 corpus.

Park SM et al., "Identification of Major Psychiatric Disorders From Resting-State
Electroencephalography Using a Machine Learning Approach", Front. Psychiatry 2021.
945 subjects from a single hospital (SMG-SNU Boramae, Seoul), 19 channels.

Single-site matters: the EEG-GCNN baseline in train.py has every diseased subject
from TUH and every healthy one from LEMON, so label and recording site are
perfectly collinear there. Here every subject comes from one hospital and one
protocol, so a multi-class result cannot be explained by site.

The published work on this corpus is all binary (each disorder vs healthy).
Multi-class across disorders is the open question, so `--task multiclass` is the
deliverable and `--task binary` exists to produce comparable reference numbers.

    python park.py --task multiclass --model gcn
    python park.py --task binary --target "Schizophrenia" --model gcn
"""
import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from train import SEED, build_model

BANDS = ["delta", "theta", "alpha", "beta", "highbeta", "gamma"]
# 10-20 montage, order fixes the node index and must stay stable.
CHANNELS = ["FP1", "FP2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
            "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
N_NODES = len(CHANNELS)
N_BANDS = len(BANDS)

DEFAULT_CLASSES = ["Healthy control", "Schizophrenia", "Mood disorder", "Addictive disorder"]


# --------------------------------------------------------------------------- data


def load_park(csv_path, classes):
    """Returns node features (S, 19, 6), coherence (S, 19, 19, 6), labels, class names."""
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df["main.disorder"].isin(classes)].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"no rows matched {classes}")
    ab, coh = features_from_df(df)
    names = sorted(classes)
    y = df["main.disorder"].map({c: i for i, c in enumerate(names)}).to_numpy(np.int64)
    return ab, coh, y, names


def features_from_df(df):
    """Band power (S, 19, 6) and coherence (S, 19, 19, 6) from Park-format columns.

    Needs no label column, so it also serves inference on new recordings.
    Node features are log1p'd then L2-normalised per subject. Both are row-wise,
    so no statistic crosses the train/test boundary.
    """
    # --- node features: AB.<letter>.<band>.<letter>.<CHAN>
    ab = np.full((len(df), N_NODES, N_BANDS), np.nan, dtype=np.float32)
    for col in (c for c in df.columns if c.startswith("AB.")):
        parts = col.split(".")
        band, chan = parts[2], parts[4]
        if band in BANDS and chan in CHANNELS:
            ab[:, CHANNELS.index(chan), BANDS.index(band)] = df[col].to_numpy(np.float32)
    if np.isnan(ab).any():
        raise ValueError(f"{int(np.isnan(ab).sum())} missing band-power cells")

    # Band power spans orders of magnitude; log first so one loud channel does
    # not dominate the normalisation.
    ab = np.log1p(np.clip(ab, 0, None))
    flat = ab.reshape(len(df), -1)
    ab = (flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8)).reshape(ab.shape)

    # --- edges: COH.<letter>.<band>.<letter>.<CH1>.<letter>.<CH2>
    # Park stores coherence as a PERCENTAGE (0-100), not the mathematical [0, 1].
    # Feeding the raw values to an unnormalised GCN over a 19-node complete graph
    # blows activations up by ~4e8 and drives accuracy below chance, so rescale
    # here and assert the range rather than trusting it.
    coh = np.zeros((len(df), N_NODES, N_NODES, N_BANDS), dtype=np.float32)
    seen = np.zeros((N_NODES, N_NODES, N_BANDS), dtype=bool)
    for col in (c for c in df.columns if c.startswith("COH.")):
        parts = col.split(".")
        band, c1, c2 = parts[2], parts[4], parts[6]
        if band in BANDS and c1 in CHANNELS and c2 in CHANNELS:
            i, j, b = CHANNELS.index(c1), CHANNELS.index(c2), BANDS.index(band)
            v = df[col].to_numpy(np.float32) / 100.0
            coh[:, i, j, b] = coh[:, j, i, b] = v  # coherence is symmetric
            seen[i, j, b] = seen[j, i, b] = True
    off_diag = coh[:, ~np.eye(N_NODES, dtype=bool), :]
    if not (0.0 <= off_diag.min() and off_diag.max() <= 1.0):
        raise ValueError(f"coherence outside [0,1] after rescaling: "
                         f"[{off_diag.min():.3f}, {off_diag.max():.3f}]")
    coh[:, np.arange(N_NODES), np.arange(N_NODES), :] = 1.0  # perfect self-coherence

    off = ~np.eye(N_NODES, dtype=bool)
    missing = int((~seen[off]).sum())
    if missing:
        raise ValueError(f"{missing} channel-pair/band coherence cells never filled")
    return ab, coh


def node_feature_dim(mode):
    return {"power": N_BANDS,
            "strength": N_BANDS * 2,
            "profile": N_BANDS + N_NODES * N_BANDS}[mode]


def build_node_features(ab_i, coh_i, mode):
    """Node features for one subject. ab_i (19, 6), coh_i (19, 19, 6).

    "power" is band power only - what the graph originally carried, and an
    information bottleneck: the classical baselines read all 1140 columns while
    the GNN saw 114 here plus coherence squeezed into a scalar edge weight.
    "profile" appends each channel's full coherence row, so a node carries the
    same connectivity information the classical models get.
    """
    if mode == "power":
        return ab_i
    if mode == "strength":
        # Mean coherence from this channel to every other, per band (19, 6).
        off = coh_i.sum(1) - coh_i[np.arange(N_NODES), np.arange(N_NODES), :]
        return np.concatenate([ab_i, off / (N_NODES - 1)], axis=1)
    if mode == "profile":
        return np.concatenate([ab_i, coh_i.reshape(N_NODES, -1)], axis=1)
    raise ValueError(mode)


def topk_edge_mask(coh_i, k):
    """(19, 19) bool adjacency keeping each channel's k most coherent neighbours.

    Ranked on coherence averaged over bands. Symmetrised - an edge survives if
    either endpoint ranks the other in its top k - and self-loops always stay.
    On a complete graph message passing mostly averages all channels together;
    sparsifying is what gives each node a distinct neighbourhood.
    """
    w = coh_i.mean(-1).copy()
    np.fill_diagonal(w, -np.inf)
    best = np.argsort(-w, axis=1)[:, :k]
    m = np.zeros((N_NODES, N_NODES), dtype=bool)
    np.put_along_axis(m, best, True, axis=1)
    m |= m.T
    np.fill_diagonal(m, True)
    return m


class ParkGraphDataset(torch.utils.data.Dataset):
    """One subject = one graph. 19 nodes; complete + self-loops (361 edges) unless topk."""

    def __init__(self, ab, coh, y, indices, edge_index, edge_mode="bands",
                 node_mode="power", topk=None):
        self.ab, self.coh, self.y, self.indices = ab, coh, y, indices
        self.edge_index, self.edge_mode, self.node_mode = edge_index, edge_mode, node_mode
        self.topk = topk

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        edge_index = self.edge_index
        if self.topk is not None:
            m = topk_edge_mask(self.coh[idx], self.topk)
            edge_index = edge_index[:, torch.from_numpy(m[edge_index[0], edge_index[1]])]
        src, dst = edge_index
        e = self.coh[idx][src.numpy(), dst.numpy(), :]  # (E, 6)
        if self.edge_mode == "mean":
            e = e.mean(1, keepdims=True)
        # Node features always see the full coherence row, even when edges are
        # sparsified - topk limits message passing, not what a node knows.
        x = build_node_features(self.ab[idx], self.coh[idx], self.node_mode)
        return Data(x=torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
                    edge_index=edge_index,
                    edge_attr=torch.from_numpy(np.ascontiguousarray(e)),
                    y=torch.tensor([self.y[idx]]),
                    dataset_idx=torch.tensor([idx]))


# ------------------------------------------------------------------------ metrics


def multiclass_metrics(probs, y_true, names):
    pred = probs.argmax(1)
    out = {
        "accuracy": accuracy_score(y_true, pred),
        "bal_acc": balanced_accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
    }
    # AUC needs every class present in this fold's test split. sklearn also wants
    # a 1-D score vector in the two-class case, not an (n, 2) matrix.
    if len(np.unique(y_true)) < len(names):
        out["macro_auc"] = float("nan")
    elif len(names) == 2:
        out["macro_auc"] = roc_auc_score(y_true, probs[:, 1])
    else:
        out["macro_auc"] = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    per = f1_score(y_true, pred, average=None, labels=range(len(names)), zero_division=0)
    for n, v in zip(names, per):
        out[f"f1[{n}]"] = v
    return out


def print_confusion(cm, names):
    w = max(len(n) for n in names)
    short = [n[:14] for n in names]
    corner = "true \\ pred"  # backslash outside the f-string: Python <3.12 rejects it inside
    print(f"\n{corner:>{w}} | " + " ".join(f"{s:>14}" for s in short))
    for i, n in enumerate(names):
        row = " ".join(f"{v:>14d}" for v in cm[i])
        print(f"{n:>{w}} | {row}")
    diag = np.trace(cm) / max(cm.sum(), 1)
    print(f"\ndiagonal mass: {diag:.3f}  (chance for {len(names)} balanced classes "
          f"= {1/len(names):.3f})")


# ------------------------------------------------------------------------ training


def run_fold(args, ab, coh, y, names, tr, te, edge_index, device):
    edge_dim = N_BANDS if args.edge_mode == "bands" else 1
    mk = lambda idx: ParkGraphDataset(ab, coh, y, idx, edge_index, args.edge_mode,
                                      args.node_features)
    tl = DataLoader(mk(tr), batch_size=args.batch_size, shuffle=True)
    vl = DataLoader(mk(te), batch_size=args.batch_size, shuffle=False)

    # normalize=True: 19-node complete graph, unlike the 8-node baseline graph.
    model = build_model(args.model, edge_dim, n_nodes=N_NODES, n_classes=len(names),
                        normalize=True,
                        in_dim=node_feature_dim(args.node_features)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    counts = np.bincount(y[tr], minlength=len(names))
    w = torch.tensor(counts.sum() / (len(names) * np.maximum(counts, 1)),
                     dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w)

    for _ in range(args.epochs):
        model.train()
        for b in tl:
            b = b.to(device)
            opt.zero_grad()
            crit(model(b.x, b.edge_index, b.edge_attr, b.batch), b.y).backward()
            opt.step()

    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for b in vl:
            b = b.to(device)
            probs.append(F.softmax(model(b.x, b.edge_index, b.edge_attr, b.batch), 1).cpu().numpy())
            ys.append(b.y.cpu().numpy())
    probs, ys = np.concatenate(probs), np.concatenate(ys)
    return multiclass_metrics(probs, ys, names), confusion_matrix(
        ys, probs.argmax(1), labels=range(len(names)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="~/eeg/park/park.csv")
    p.add_argument("--model", default="gcn", choices=["fcnn", "gcn", "cheb", "gatv2"])
    p.add_argument("--task", default="multiclass", choices=["multiclass", "binary"])
    p.add_argument("--target", default="Schizophrenia",
                   help="binary task: this disorder vs Healthy control")
    p.add_argument("--classes", default=",".join(DEFAULT_CLASSES),
                   help="multiclass task: comma-separated main.disorder values")
    p.add_argument("--edge-mode", default="bands", choices=["bands", "mean"])
    p.add_argument("--node-features", default="power",
                   choices=["power", "strength", "profile"],
                   help="power=band power only; strength=+mean coherence per band; "
                        "profile=+full coherence row (same info the classical baselines get)")
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = ([args.target, "Healthy control"] if args.task == "binary"
               else [c.strip() for c in args.classes.split(",")])
    ab, coh, y, names = load_park(Path(args.data).expanduser(), classes)
    print(f"[data] {len(y)} subjects, {len(names)} classes")
    for i, n in enumerate(names):
        print(f"        {n:<36} {int((y == i).sum())}")

    edge_index = torch.tensor(list(product(range(N_NODES), range(N_NODES))),
                              dtype=torch.long).t().contiguous()
    print(f"[graph] {N_NODES} nodes, {edge_index.shape[1]} edges (complete + self-loops), "
          f"edge_dim={N_BANDS if args.edge_mode == 'bands' else 1}, "
          f"node_features={args.node_features} (dim {node_feature_dim(args.node_features)})")

    # One row per subject, so a stratified split cannot leak a subject across
    # folds the way window-level data can.
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    results, cms = [], np.zeros((len(names), len(names)), dtype=int)
    for k, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        m, cm = run_fold(args, ab, coh, y, names, tr, te, edge_index, device)
        cms += cm
        keys = ("accuracy", "bal_acc", "macro_f1", "macro_auc")
        print(f"[fold {k}] " + "  ".join(f"{n}={m[n]:.3f}" for n in keys))
        results.append(m)

    print(f"\n=== {args.model} | task={args.task} | edge-mode={args.edge_mode} ===")
    summary = {}
    for key in results[0]:
        vals = [r[key] for r in results]
        summary[key] = [float(np.nanmean(vals)), float(np.nanstd(vals))]
        print(f"{key:>22}: {np.nanmean(vals):.3f} ({np.nanstd(vals):.3f})")
    print_confusion(cms, names)

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"args": vars(args), "summary": summary, "confusion": cms.tolist(),
             "classes": names}, indent=2))


# ---------------------------------------------------------------------- self-check


def selfcheck():
    rng = np.random.default_rng(0)
    n = 40
    ab = rng.random((n, N_NODES, N_BANDS)).astype(np.float32)
    coh = rng.random((n, N_NODES, N_NODES, N_BANDS)).astype(np.float32)
    coh = (coh + coh.transpose(0, 2, 1, 3)) / 2  # enforce symmetry
    y = rng.integers(0, 4, n)
    names = ["a", "b", "c", "d"]
    ei = torch.tensor(list(product(range(N_NODES), range(N_NODES))), dtype=torch.long).t().contiguous()
    assert ei.shape[1] == N_NODES * N_NODES == 361

    ds = ParkGraphDataset(ab, coh, y, np.arange(n), ei, "bands")
    g = ds[0]
    assert g.x.shape == (19, 6), g.x.shape
    assert g.edge_attr.shape == (361, 6), g.edge_attr.shape
    assert ParkGraphDataset(ab, coh, y, np.arange(n), ei, "mean")[0].edge_attr.shape == (361, 1)

    # Edge attributes must line up with the (src, dst) pair they are attached to.
    src, dst = ei
    for k in (0, 137, 200, 360):  # incl. a self-loop (0) and the last edge
        assert np.allclose(g.edge_attr[k].numpy(), coh[0, src[k], dst[k], :]), \
            f"edge_attr misaligned at edge {k}"

    batch = next(iter(DataLoader(ds, batch_size=4)))
    for attr in ("y", "dataset_idx"):
        assert getattr(batch, attr).shape == (4,), f"{attr} collated wrong"
    for name in ("fcnn", "gcn", "cheb", "gatv2"):
        m = build_model(name, 6, n_nodes=N_NODES, n_classes=4)
        assert m(batch.x, batch.edge_index, batch.edge_attr, batch.batch).shape == (4, 4), name

    # Node-feature modes must have the advertised width, and "profile" must
    # actually carry the coherence row - that is the whole point of the mode.
    for mode, want in [("power", 6), ("strength", 12), ("profile", 6 + 19 * 6)]:
        feats = build_node_features(ab[0], coh[0], mode)
        assert feats.shape == (N_NODES, want), f"{mode}: {feats.shape}"
        assert node_feature_dim(mode) == want, mode
        assert np.allclose(feats[:, :6], ab[0]), f"{mode} must keep band power first"
    prof = build_node_features(ab[0], coh[0], "profile")
    assert np.allclose(prof[3, 6:].reshape(19, 6), coh[0, 3]), "profile row misaligned"
    strg = build_node_features(ab[0], coh[0], "strength")
    expect = (coh[0, 5].sum(0) - coh[0, 5, 5]) / (N_NODES - 1)
    assert np.allclose(strg[5, 6:], expect), "strength must exclude the self-loop"

    for mode in ("power", "strength", "profile"):
        d = ParkGraphDataset(ab, coh, y, np.arange(n), ei, "bands", mode)
        bb = next(iter(DataLoader(d, batch_size=4)))
        for name in ("fcnn", "gcn", "gatv2"):
            m = build_model(name, 6, n_nodes=N_NODES, n_classes=4,
                            normalize=True, in_dim=node_feature_dim(mode))
            assert m(bb.x, bb.edge_index, bb.edge_attr, bb.batch).shape == (4, 4), f"{name}/{mode}"

    # topk sparsification: every node keeps >= k real neighbours plus its
    # self-loop, the mask is symmetric, and it actually removes edges.
    for kk in (2, 4):
        msk = topk_edge_mask(coh[0], kk)
        assert (msk == msk.T).all(), "topk mask must be symmetric"
        assert msk.diagonal().all(), "self-loops must survive"
        assert ((msk.sum(1) - 1) >= kk).all(), f"a node kept fewer than {kk} neighbours"
        assert msk.sum() < N_NODES * N_NODES, "topk must drop edges"
        w = coh[0].mean(-1).copy()
        np.fill_diagonal(w, -np.inf)
        assert msk[3, int(np.argmax(w[3]))], "a node's strongest neighbour must be kept"
    sp = ParkGraphDataset(ab, coh, y, np.arange(n), ei, "bands", "profile", topk=3)
    g3 = sp[0]
    assert g3.edge_index.shape[1] == topk_edge_mask(coh[0], 3).sum()
    s3, d3 = g3.edge_index
    assert np.allclose(g3.edge_attr.numpy(), coh[0][s3.numpy(), d3.numpy(), :]), \
        "edge_attr misaligned after sparsification"
    assert g3.x.shape == (N_NODES, node_feature_dim("profile")), "topk must not shrink node features"
    b3 = next(iter(DataLoader(sp, batch_size=4)))
    for pool in ("add", "mean", "meanmax"):
        for name in ("gcn", "gatv2"):
            mm = build_model(name, 6, n_nodes=N_NODES, n_classes=2, normalize=True,
                             in_dim=node_feature_dim("profile"), pool=pool)
            assert mm(b3.x, b3.edge_index, b3.edge_attr, b3.batch).shape == (4, 2), (name, pool)

    # Scale regression guard. A 19-node complete graph with unnormalised GCN
    # aggregation explodes; this is what drove accuracy below chance until the
    # coherence rescale and normalize=True went in.
    torch.manual_seed(0)
    normed = build_model("gcn", 6, n_nodes=N_NODES, n_classes=4, normalize=True).eval()
    raw = build_model("gcn", 6, n_nodes=N_NODES, n_classes=4, normalize=False).eval()
    # Identical weights, so normalisation is the only difference being tested
    # (comparing two randomly initialised models made this assert flaky).
    raw.load_state_dict(normed.state_dict())
    with torch.no_grad():
        h_n = normed.conv1(batch.x, batch.edge_index, edge_weight=batch.edge_attr.sum(1))
        h_r = raw.conv1(batch.x, batch.edge_index, edge_weight=batch.edge_attr.sum(1))
        out_n = normed(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    assert out_n.abs().max().item() < 1e3, f"normalised GCN still exploding: {out_n.abs().max()}"
    assert h_r.abs().mean() > 5 * h_n.abs().mean(), \
        "normalize=False should aggregate far larger than normalize=True on a complete graph"

    # A perfect predictor must score 1.0; a constant one must not.
    perfect = np.eye(4)[y]
    assert multiclass_metrics(perfect, y, names)["macro_f1"] == 1.0
    const = np.tile([1.0, 0, 0, 0], (n, 1))
    assert multiclass_metrics(const, y, names)["bal_acc"] < 0.5

    # Balanced accuracy must ignore class imbalance; accuracy must not.
    yi = np.array([0] * 36 + [1] * 4)
    maj = np.tile([1.0, 0.0], (40, 1))
    mi = multiclass_metrics(maj, yi, ["a", "b"])
    assert mi["accuracy"] == 0.9 and mi["bal_acc"] == 0.5, mi
    print("selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else main()
