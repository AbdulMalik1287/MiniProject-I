#!/usr/bin/env python3
"""EEG-GCNN reproduction + model-agnostic ablation harness.

Baseline is Wagh & Varatharajah, ML4H @ NeurIPS 2020 (arXiv:2011.12107).
Graph: 8-channel bipolar montage, complete graph w/ self-loops (64 edges),
edge weight = normalised geodesic electrode distance + spectral coherence,
node features = 6-band PSD.

Everything except the conv layer is shared across models, so `--model` is the
only thing that changes between the September baseline and the October
attention experiments.

    python train.py --model gcn                   # published baseline
    python train.py --model gatv2 --edge-mode split
    python train.py --model gcn --topk 3          # sparsity sweep
"""
import argparse
import json
import math
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import load
from sklearn import preprocessing
from sklearn.metrics import (balanced_accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (BatchNorm, ChebConv, GATv2Conv, GCNConv,
                                global_add_pool)

N_NODES = 8
N_BANDS = 6
SEED = 42

# Order defines node index - must stay consistent with spec_coh_values.npy.
CH_NAMES = ["F7-F3", "F8-F4", "T7-C3", "T8-C4", "P7-P3", "P8-P4", "O1-P3", "O2-P4"]
# 10-10 electrode sitting between each bipolar pair, used for geodesic distance.
# PO3/PO4 are absent from standard_1010, upstream substitutes O1/O2.
REF_NAMES = ["F5", "F6", "C5", "C6", "P5", "P6", "O1", "O2"]


# --------------------------------------------------------------------------- data


def geodesic_distances(root: Path, edge_index: torch.Tensor) -> np.ndarray:
    """Great-circle distance between reference electrodes, min-max normalised."""
    coords = pd.read_csv(root / "standard_1010.tsv.txt", sep="\t").set_index("label")
    xyz = {n: coords.loc[n, ["x", "y", "z"]].astype(float).values for n in set(REF_NAMES)}
    d = []
    for a, b in edge_index.t().tolist():
        dot = float(np.dot(xyz[REF_NAMES[a]], xyz[REF_NAMES[b]]))
        d.append(math.acos(round(dot, 2)))  # rounding keeps acos domain valid
    d = np.asarray(d)
    return (d - d.min()) / (d.max() - d.min())


def load_data(root: Path):
    X = load(root / "psd_features_data_X")
    y = load(root / "labels_y")
    meta = pd.read_csv(root / "master_metadata_index.csv")
    coh = np.load(root / "spec_coh_values.npy", allow_pickle=True)

    # Per-window L2 normalisation, exactly as upstream. Row-wise, so no
    # cross-sample statistics leak across the split.
    X = preprocessing.normalize(np.asarray(X).reshape(len(y), N_NODES * N_BANDS))
    label_mapping, y = np.unique(y, return_inverse=True)
    groups = meta["patient_ID"].astype(str).values
    # Fail here rather than after a fold of training - subject-level metrics
    # depend on this holding.
    mixed = pd.DataFrame({"g": groups, "y": y}).groupby("g")["y"].nunique()
    if mixed.max() > 1:
        raise ValueError(f"{(mixed > 1).sum()} subjects have windows with conflicting labels")
    return X.astype(np.float32), y.astype(np.int64), groups, coh.astype(np.float32), label_mapping


class EEGGraphDataset(torch.utils.data.Dataset):
    """Windows -> PyG graphs. Sparsification is per-window because coherence is."""

    def __init__(self, X, y, coh, indices, edge_index, distances,
                 edge_mode="sum", topk=None, thresh=None):
        self.X, self.y, self.coh, self.indices = X, y, coh, indices
        self.edge_index, self.distances = edge_index, distances
        self.edge_mode, self.topk, self.thresh = edge_mode, topk, thresh

    def __len__(self):
        return len(self.indices)

    def _sparsify(self, w_scalar):
        """Return a boolean keep-mask over edges. Self-loops always survive."""
        src, dst = self.edge_index
        keep = np.ones(len(w_scalar), dtype=bool)
        if self.thresh is not None:
            keep &= w_scalar >= self.thresh
        if self.topk is not None:
            keep &= np.zeros(len(w_scalar), dtype=bool)
            for n in range(N_NODES):
                rows = np.where((src.numpy() == n) & (src.numpy() != dst.numpy()))[0]
                best = rows[np.argsort(-w_scalar[rows])[: self.topk]]
                keep[best] = True
        keep |= (src.numpy() == dst.numpy())
        return keep

    def __getitem__(self, i):
        idx = self.indices[i]
        x = torch.from_numpy(self.X[idx].reshape(N_NODES, N_BANDS))
        dist, coh = self.distances, self.coh[idx]
        combined = dist + coh  # upstream sums the two into one scalar weight

        edge_index = self.edge_index
        if self.topk is not None or self.thresh is not None:
            keep = self._sparsify(combined)
            edge_index = edge_index[:, keep]
            dist, coh, combined = dist[keep], coh[keep], combined[keep]

        if self.edge_mode == "split":
            edge_attr = torch.from_numpy(np.stack([dist, coh], 1).astype(np.float32))
        else:
            edge_attr = torch.from_numpy(combined.astype(np.float32)).unsqueeze(1)

        # Both must be shape-(1,) tensors, not Python scalars: PyG collates
        # non-tensor attributes into lists, which breaks .cpu() downstream.
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                    y=torch.tensor([self.y[idx]]),
                    dataset_idx=torch.tensor([idx]))


# ------------------------------------------------------------------------- models


class GraphNet(nn.Module):
    """Two conv layers -> BN -> add-pool -> MLP head. Upstream topology, swappable conv."""

    def __init__(self, model, edge_dim, n_classes=2, normalize=False):
        super().__init__()
        self.model = model
        if model == "gcn":
            # normalize=False reproduces the published model exactly, because its
            # 8-node graph carries small edge weights. On a denser graph with
            # larger weights the unnormalised sum explodes (a 19-node complete
            # graph blew activations up ~4e8x), so callers there pass True.
            self.conv1 = GCNConv(N_BANDS, 32, improved=True, cached=False, normalize=normalize)
            self.conv2 = GCNConv(32, 20, improved=True, cached=False, normalize=normalize)
        elif model == "cheb":
            self.conv1 = ChebConv(N_BANDS, 32, K=3)
            self.conv2 = ChebConv(32, 20, K=3)
        elif model == "gatv2":
            # edge_dim is what lets attention actually read coherence/distance;
            # without it GATv2 is blind to the edge weights the paper computes.
            self.conv1 = GATv2Conv(N_BANDS, 8, heads=4, edge_dim=edge_dim)
            self.conv2 = GATv2Conv(32, 20, heads=1, edge_dim=edge_dim)
        else:
            raise ValueError(model)
        self.bn = BatchNorm(20)
        self.fc1, self.fc2 = nn.Linear(20, 10), nn.Linear(10, n_classes)
        for m in (self.fc1, self.fc2):
            nn.init.xavier_normal_(m.weight, gain=1)

    def _conv(self, conv, x, edge_index, edge_attr):
        if self.model == "gcn":
            return conv(x, edge_index, edge_weight=edge_attr.sum(1))
        if self.model == "cheb":
            return conv(x, edge_index, edge_weight=edge_attr.sum(1))
        return conv(x, edge_index, edge_attr=edge_attr)

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.leaky_relu(self._conv(self.conv1, x, edge_index, edge_attr))
        x = F.leaky_relu(self.bn(self._conv(self.conv2, x, edge_index, edge_attr)))
        out = global_add_pool(x, batch)
        out = F.dropout(out, p=0.2, training=self.training)
        return self.fc2(F.leaky_relu(self.fc1(out)))


class FCNN(nn.Module):
    """Graph-blind control. If this matches the GNNs, structure is not being used."""

    def __init__(self, n_nodes=N_NODES, n_classes=2):
        super().__init__()
        self.n_nodes = n_nodes
        self.net = nn.Sequential(
            nn.Linear(n_nodes * N_BANDS, 32), nn.LeakyReLU(),
            nn.Linear(32, 20), nn.LeakyReLU(), nn.Dropout(0.2),
            nn.Linear(20, 10), nn.LeakyReLU(), nn.Linear(10, n_classes))

    def forward(self, x, edge_index, edge_attr, batch):
        n_graphs = int(batch.max()) + 1
        return self.net(x.view(n_graphs, self.n_nodes * N_BANDS))


def build_model(name, edge_dim, n_nodes=N_NODES, n_classes=2, normalize=False):
    if name == "fcnn":
        return FCNN(n_nodes, n_classes)
    return GraphNet(name, edge_dim, n_classes, normalize)


# ------------------------------------------------------------------------ metrics


DISEASED = 0  # np.unique(['diseased','healthy']) -> diseased=0, healthy=1


def patient_metrics(probs, y_true, groups):
    """Aggregate window predictions to one prediction per subject (upstream does this).

    Reporting window-level numbers on multi-window subjects inflates results;
    subject-level is the number that means anything clinically.

    Diseased is the positive class. That is not cosmetic: the corpus is 87%
    diseased / 13% healthy, so precision and recall computed against `healthy`
    describe detection of the rare class and look nothing like the published
    figures. AUC and balanced accuracy are symmetric under that flip, which is
    exactly why they can match while precision/recall/f1 do not.
    """
    df = pd.DataFrame({"g": groups, "p": probs[:, DISEASED], "y": y_true})
    # "first" is only valid because a subject is diseased or healthy as a whole.
    # If that ever breaks, every subject-level number below is silently wrong.
    if df.groupby("g")["y"].nunique().max() > 1:
        raise ValueError("labels vary within a subject; subject-level aggregation is invalid")
    agg = df.groupby("g").agg({"p": "mean", "y": "first"})
    pos = (agg["y"] == DISEASED).astype(int)  # 1 = diseased
    auc = roc_auc_score(pos, agg["p"])
    # Youden's J for the operating point, as upstream.
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(pos, agg["p"])
    cut = thr[np.argmax(tpr - fpr)]
    pred = (agg["p"] >= cut).astype(int)
    return {
        "auc": auc,
        "precision": precision_score(pos, pred, zero_division=0),
        "recall": recall_score(pos, pred, zero_division=0),
        "f1": f1_score(pos, pred, zero_division=0),
        "bal_acc": balanced_accuracy_score(pos, pred),
        "n_subjects": len(agg),
    }


# ------------------------------------------------------------------------ training


def run_fold(args, data, tr_idx, te_idx, edge_index, distances, device):
    X, y, groups, coh = data
    edge_dim = 2 if args.edge_mode == "split" else 1
    mk = lambda idx: EEGGraphDataset(X, y, coh, idx, edge_index, distances,
                                     args.edge_mode, args.topk, args.thresh)
    tl = DataLoader(mk(tr_idx), batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    vl = DataLoader(mk(te_idx), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = build_model(args.model, edge_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Diseased/healthy is imbalanced in the pooled corpus; weight the loss.
    counts = np.bincount(y[tr_idx], minlength=2)
    w = torch.tensor(counts.sum() / (2.0 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w)

    for _ in range(args.epochs):
        model.train()
        for b in tl:
            b = b.to(device)
            opt.zero_grad()
            loss = crit(model(b.x, b.edge_index, b.edge_attr, b.batch), b.y)
            loss.backward()
            opt.step()

    model.eval()
    probs, ys, idxs = [], [], []
    with torch.no_grad():
        for b in vl:
            b = b.to(device)
            out = model(b.x, b.edge_index, b.edge_attr, b.batch)
            probs.append(F.softmax(out, 1).cpu().numpy())
            ys.append(b.y.cpu().numpy())
            idxs.append(b.dataset_idx.cpu().numpy())
    probs, ys, idxs = np.concatenate(probs), np.concatenate(ys), np.concatenate(idxs)
    return patient_metrics(probs, ys, groups[idxs])


CKPT_RENAME = {"conv2_bn": "bn", "fc_block1": "fc1", "fc_block2": "fc2"}


def port_state_dict(state):
    """Port a 2020-era PyG checkpoint onto the current GCNConv layout.

    Two changes, and the second one is the dangerous one:
      - GCNConv moved its weight into an `nn.Linear` submodule, so
        `conv1.weight` is now `conv1.lin.weight`.
      - That also flipped the layout from (in, out) to nn.Linear's (out, in).
        The old layer computed `x @ W`; the new one computes `x @ W.T`. A
        rename without the transpose loads a wrongly-oriented matrix.

    Here the two conv layers are 6->32 and 32->20, so a missing transpose
    would be caught by the shape check in load_state_dict. That is luck, not
    safety - the reproduced AUC below is the real verification.
    """
    out = {}
    for key, v in state.items():
        parts = [CKPT_RENAME.get(p, p) for p in key.split(".")]
        if parts[0].startswith("conv") and parts[-1] == "weight" and v.dim() == 2:
            parts, v = [parts[0], "lin", "weight"], v.t().contiguous()
        out[".".join(parts)] = v
    return out


def eval_checkpoints(args, data, heldout_idx, edge_index, distances, device):
    """Reproduce the published Table 2 numbers from the released checkpoints.

    Upstream's own model class differs from ours only in attribute names, so we
    remap keys instead of patching their 2020-era PyG pipeline. Running their
    weights through our loader also proves our loader agrees with theirs.
    """
    X, y, groups, coh = data
    ds = EEGGraphDataset(X, y, coh, heldout_idx, edge_index, distances, "sum")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    folds = []
    for k in range(args.folds):
        ckpt = Path(args.ckpt_dir).expanduser() / f"{args.ckpt_prefix}_fold_{k}.ckpt"
        state = torch.load(ckpt, map_location=device, weights_only=False)["state_dict"]
        state = port_state_dict(state)

        model = build_model("gcn", 1).to(device).double()
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not missing and not unexpected, f"key mismatch: missing={missing} unexpected={unexpected}"

        model.eval()
        probs, ys, idxs = [], [], []
        with torch.no_grad():
            for b in loader:
                b = b.to(device)
                out = model(b.x.double(), b.edge_index, b.edge_attr.double(), b.batch)
                probs.append(F.softmax(out, 1).cpu().numpy())
                ys.append(b.y.cpu().numpy())
                idxs.append(b.dataset_idx.cpu().numpy())
        idxs = np.concatenate(idxs)
        m = patient_metrics(np.concatenate(probs), np.concatenate(ys), groups[idxs])
        print(f"[ckpt fold {k}] " + "  ".join(f"{n}={v:.3f}" for n, v in m.items() if n != "n_subjects"))
        folds.append(m)

    print(f"\n=== reproduction from released checkpoints ({args.ckpt_prefix}) ===")
    print("published (repo README, post-fix): auc=0.871(0.001) precision=0.989 "
          "recall=0.677 f1=0.804 bal_acc=0.810")
    for key in ("auc", "precision", "recall", "f1", "bal_acc"):
        vals = [r[key] for r in folds]
        print(f"{key:>10}: {np.mean(vals):.3f} ({np.std(vals):.3f})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="~/eeg/data", type=str)
    p.add_argument("--model", default="gcn", choices=["fcnn", "gcn", "cheb", "gatv2"])
    p.add_argument("--edge-mode", default="sum", choices=["sum", "split"])
    p.add_argument("--topk", type=int, default=None, help="keep top-k neighbours per node")
    p.add_argument("--thresh", type=float, default=None, help="drop edges below this weight")
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default=None)
    p.add_argument("--eval-ckpt", action="store_true",
                   help="reproduce published results from released checkpoints")
    p.add_argument("--ckpt-dir", default="~/eeg/data")
    p.add_argument("--ckpt-prefix", default="psd_gnn_shallow")
    args = p.parse_args()

    root = Path(args.data).expanduser()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y, groups, coh, mapping = load_data(root)
    print(f"[data] {len(y)} windows, {len(np.unique(groups))} subjects, labels {list(mapping)}")

    node_ids = range(N_NODES)
    edge_index = torch.tensor(list(product(node_ids, node_ids)), dtype=torch.long).t().contiguous()
    distances = geodesic_distances(root, edge_index)

    # Held-out test subjects, seed 42 - same split the published checkpoints used.
    subjects = pd.unique(groups)
    train_subj, test_subj = train_test_split(subjects, test_size=0.30, random_state=SEED)
    tv_idx = np.where(np.isin(groups, train_subj))[0]
    heldout_idx = np.where(np.isin(groups, test_subj))[0]
    assert not (set(train_subj) & set(test_subj)), "subject in both train and held-out"
    print(f"[split] {len(train_subj)} train/val subjects, {len(test_subj)} held out")

    if args.eval_ckpt:
        eval_checkpoints(args, (X, y, groups, coh), heldout_idx, edge_index, distances, device)
        return

    # Subject-wise CV inside the train pool - no subject spans a fold boundary.
    gkf = GroupKFold(n_splits=args.folds)
    results = []
    for k, (a, b) in enumerate(gkf.split(tv_idx, y[tv_idx], groups[tv_idx])):
        m = run_fold(args, (X, y, groups, coh), tv_idx[a], tv_idx[b], edge_index, distances, device)
        print(f"[fold {k}] " + "  ".join(f"{n}={v:.3f}" for n, v in m.items() if n != "n_subjects"))
        results.append(m)

    print(f"\n=== {args.model} (edge-mode={args.edge_mode}, topk={args.topk}, thresh={args.thresh}) ===")
    summary = {}
    for key in ("auc", "precision", "recall", "f1", "bal_acc"):
        vals = [r[key] for r in results]
        summary[key] = [float(np.mean(vals)), float(np.std(vals))]
        print(f"{key:>10}: {np.mean(vals):.3f} ({np.std(vals):.3f})")
    if args.out:
        Path(args.out).write_text(json.dumps({"args": vars(args), "summary": summary}, indent=2))


# ---------------------------------------------------------------------- self-check


def selfcheck():
    """Runs without the dataset: shapes, sparsification, and split integrity."""
    rng = np.random.default_rng(0)
    n, n_edges = 200, N_NODES * N_NODES
    X = rng.random((n, N_NODES * N_BANDS)).astype(np.float32)
    coh = rng.random((n, n_edges)).astype(np.float32)
    # 20 subjects x 10 windows. Label is a property of the subject, not the
    # window - matching the real corpus, where all of a subject's windows
    # carry that subject's diagnosis.
    groups = np.array([f"s{i // 10}" for i in range(n)])
    subj_label = {g: int(rng.integers(0, 2)) for g in np.unique(groups)}
    y = np.array([subj_label[g] for g in groups])
    ei = torch.tensor(list(product(range(N_NODES), range(N_NODES))), dtype=torch.long).t().contiguous()
    dist = rng.random(n_edges).astype(np.float32)

    ds = EEGGraphDataset(X, y, coh, np.arange(n), ei, dist, "sum")
    g = ds[0]
    assert g.x.shape == (N_NODES, N_BANDS), g.x.shape
    assert g.edge_index.shape[1] == 64, "complete graph + self-loops = 64 edges"
    assert g.edge_attr.shape == (64, 1)

    split = EEGGraphDataset(X, y, coh, np.arange(n), ei, dist, "split")
    assert split[0].edge_attr.shape == (64, 2), "split mode must keep distance and coherence apart"

    sp = EEGGraphDataset(X, y, coh, np.arange(n), ei, dist, "sum", topk=2)
    e = sp[0].edge_index
    assert e.shape[1] < 64, "top-k must drop edges"
    assert (e[0] == e[1]).sum() == N_NODES, "every self-loop must survive sparsification"
    for node in range(N_NODES):  # k neighbours + 1 self-loop
        assert int((e[0] == node).sum()) == 3, f"node {node} kept {(e[0] == node).sum()} edges"

    # A subject must never appear on both sides of a fold.
    gkf = GroupKFold(n_splits=5)
    for a, b in gkf.split(np.arange(n), y, groups):
        assert not (set(groups[a]) & set(groups[b])), "subject leaked across the split"

    # Batch attributes must survive collation as tensors of shape (batch,),
    # so the eval loop can call .cpu().numpy() on them.
    batch = next(iter(DataLoader(ds, batch_size=4)))
    for attr in ("y", "dataset_idx"):
        v = getattr(batch, attr)
        assert torch.is_tensor(v), f"{attr} collated to {type(v).__name__}, not a tensor"
        assert v.shape == (4,), f"{attr} collated to {tuple(v.shape)}, expected (4,)"
    assert batch.dataset_idx.tolist() == [0, 1, 2, 3], batch.dataset_idx.tolist()

    for name in ("fcnn", "gcn", "cheb", "gatv2"):
        m = build_model(name, 1)
        out = m(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        assert out.shape == (4, 2), f"{name} produced {out.shape}"

    gat = build_model("gatv2", 2)
    b2 = next(iter(DataLoader(split, batch_size=4)))
    assert gat(b2.x, b2.edge_index, b2.edge_attr, b2.batch).shape == (4, 2)

    # Checkpoint porting: keys must land exactly where the model wants them,
    # conv weights must be transposed, and fc weights must NOT be.
    old = {
        "conv1.weight": torch.randn(6, 32), "conv1.bias": torch.randn(32),
        "conv2.weight": torch.randn(32, 20), "conv2.bias": torch.randn(20),
        "conv2_bn.module.weight": torch.randn(20), "conv2_bn.module.bias": torch.randn(20),
        "conv2_bn.module.running_mean": torch.randn(20),
        "conv2_bn.module.running_var": torch.randn(20).abs(),
        "conv2_bn.module.num_batches_tracked": torch.tensor(0),
        "fc_block1.weight": torch.randn(10, 20), "fc_block1.bias": torch.randn(10),
        "fc_block2.weight": torch.randn(2, 10), "fc_block2.bias": torch.randn(2),
    }
    ported = port_state_dict(old)
    assert torch.equal(ported["conv1.lin.weight"], old["conv1.weight"].t()), "conv1 must transpose"
    assert torch.equal(ported["conv2.lin.weight"], old["conv2.weight"].t()), "conv2 must transpose"
    assert torch.equal(ported["fc1.weight"], old["fc_block1.weight"]), "fc must not transpose"
    assert torch.equal(ported["conv1.bias"], old["conv1.bias"]), "bias must pass through"
    want = set(build_model("gcn", 1).state_dict())
    assert set(ported) == want, f"ported keys != model keys: {set(ported) ^ want}"
    build_model("gcn", 1).load_state_dict(ported)  # strict: shapes must line up

    # Subject-level aggregation must collapse windows, not pass them through.
    probs = np.stack([1 - y, y], 1).astype(float)
    m = patient_metrics(probs, y, groups)
    assert m["n_subjects"] == 20, m["n_subjects"]
    assert m["auc"] == 1.0, "perfect probabilities must give AUC 1.0"

    # The aggregation guard must actually fire on within-subject label conflict.
    bad = y.copy()
    bad[0] = 1 - bad[0]
    try:
        patient_metrics(probs, bad, groups)
    except ValueError:
        pass
    else:
        raise AssertionError("mixed within-subject labels must be rejected")
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        selfcheck()
    else:
        main()
