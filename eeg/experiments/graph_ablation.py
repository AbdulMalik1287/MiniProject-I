"""Team plan E03 + E04: graph construction x GNN family, 3-class AD/FTD/HC on ds004504.

One change at a time, everything else fixed:
    node features  log relative band power per electrode (19 x 6) only - the main
                   GATv2 also feeds each node its coherence row, which would let any
                   graph look equally good, so here connectivity reaches the model
                   only through the edges
    graphs         per window, scalar edge weight, top-4 neighbours per node (as tuned):
                   spatial    exp(-d/d_mean) on standard 10-20 positions (same for every window)
                   pearson    |correlation| of the window's channel signals
                   plv        phase-locking value of the window (1-45 Hz Hilbert phase)
                   coherence  band-averaged magnitude-squared coherence (raw_eeg features)
                   hybrid     mean of min-max-normalised spatial and coherence weights
                   none       self-loops only (graph-blind control)
    models         GCN (edge weights), GraphSAGE (topology only, mean aggregation),
                   GATv2 (edge weight as edge feature); 2 layers 32/20, BatchNorm,
                   LeakyReLU, mean+max pool, MLP; Adam 1e-3, wd 5e-4, batch 64, 20 epochs
    folds          StratifiedKFold(5) over subjects, seeds 0..R-1 (same as eeg/experiments/three_class.py)
    metric         subject-level macro-F1 / balanced accuracy / macro OvR AUC

    python -m eeg.experiments.graph_ablation --out ~/eeg/raw2 --repeats 3
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import hilbert
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import BatchNorm, GATv2Conv, GCNConv, SAGEConv, global_max_pool, global_mean_pool

from eeg.data import park as P
from eeg.data import raw_eeg as R

CLASSES = ["control", "ftd", "alzheimers"]
GRAPHS = ["none", "spatial", "pearson", "plv", "coherence", "hybrid"]
MODELS = ["gcn", "sage", "gatv2"]
TOPK = 4


def spatial_w():
    import mne
    pos = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    xyz = np.array([pos[c.capitalize() if c.startswith("FP") else c] for c in P.CHANNELS])
    d = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    return np.exp(-d / d[np.triu_indices(len(d), 1)].mean())


def window_weights(kind, win, coh, sp):
    n = len(win)
    if kind == "none":
        return np.zeros((n, 19, 19))
    if kind == "spatial":
        return np.repeat(sp[None], n, 0)
    if kind == "coherence":
        return coh.mean(-1).astype(np.float64)
    if kind == "hybrid":
        mm = lambda a: (a - a.min((1, 2), keepdims=True)) / (np.ptp(a, axis=(1, 2), keepdims=True) + 1e-12)  # noqa: E731
        return (mm(np.repeat(sp[None], n, 0)) + mm(coh.mean(-1).astype(np.float64))) / 2
    out = np.zeros((n, 19, 19))
    for k in range(0, n, 256):
        w = win[k:k + 256].astype(np.float64)
        if kind == "pearson":
            z = (w - w.mean(-1, keepdims=True)) / (w.std(-1, keepdims=True) + 1e-12)
            out[k:k + 256] = np.abs(np.einsum("wct,wdt->wcd", z, z) / w.shape[-1])
        else:  # plv
            e = np.exp(1j * np.angle(hilbert(w, axis=-1)))
            out[k:k + 256] = np.abs(np.einsum("wct,wdt->wcd", e, e.conj()) / w.shape[-1])
    return out


def build(ab, weights, y, idx):
    data = []
    for i in idx:
        w = weights[i].copy()
        np.fill_diagonal(w, -np.inf)
        if np.isfinite(w).any() and w[np.isfinite(w)].max() > 0:
            nb = np.argsort(-w, axis=1)[:, :TOPK]
            src = np.repeat(np.arange(19), TOPK)
            dst = nb.ravel()
            ew = weights[i][src, dst]
        else:
            src = dst = np.zeros(0, dtype=int)
            ew = np.zeros(0)
        src, dst = np.concatenate([src, np.arange(19)]), np.concatenate([dst, np.arange(19)])  # + self-loops
        ew = np.concatenate([ew, np.ones(19)])
        data.append(Data(x=torch.tensor(ab[i], dtype=torch.float32),
                         edge_index=torch.tensor(np.stack([src, dst]), dtype=torch.long),
                         edge_attr=torch.tensor(ew, dtype=torch.float32)[:, None], y=torch.tensor([int(y[i])])))
    return data


class Net(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        mk = {"gcn": lambda a, b: GCNConv(a, b), "sage": lambda a, b: SAGEConv(a, b),
              "gatv2": lambda a, b: GATv2Conv(a, b, heads=1, edge_dim=1)}[kind]
        self.c1, self.c2, self.bn = mk(6, 32), mk(32, 20), BatchNorm(20)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(40, 10), nn.LeakyReLU(), nn.Linear(10, 3))

    def conv(self, c, x, b):
        if self.kind == "gcn":
            return c(x, b.edge_index, edge_weight=b.edge_attr[:, 0])
        if self.kind == "sage":
            return c(x, b.edge_index)
        return c(x, b.edge_index, edge_attr=b.edge_attr)

    def forward(self, b):
        x = F.leaky_relu(self.conv(self.c1, b.x, b))
        x = F.leaky_relu(self.bn(self.conv(self.c2, x, b)))
        return self.head(torch.cat([global_mean_pool(x, b.batch), global_max_pool(x, b.batch)], 1))


def fit_predict(kind, tr_data, te_data, ytr, device, seed):
    torch.manual_seed(seed)
    model = Net(kind).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    cnt = np.bincount(ytr, minlength=3)
    w = torch.tensor(cnt.sum() / (3.0 * np.maximum(cnt, 1)), dtype=torch.float32, device=device)
    for _ in range(R.RAW_EPOCHS):
        model.train()
        for b in DataLoader(tr_data, batch_size=64, shuffle=True):
            b = b.to(device)
            opt.zero_grad()
            F.cross_entropy(model(b), b.y, weight=w).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return np.concatenate([torch.softmax(model(b.to(device)).double(), 1).cpu().numpy()
                               for b in DataLoader(te_data, batch_size=512)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/eeg/raw2")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--save", default="results/graph_ablation.json")
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ab, coh, subject, groups, _, win = R.load_features(out, ["ds004504"], raw=True)
    yw = np.array([CLASSES.index(g) for g in groups])
    u = np.unique(subject)
    rows_of = {s: np.where(subject == s)[0] for s in u}
    ys = np.array([yw[rows_of[s][0]] for s in u])
    sp = spatial_w()
    save = Path(args.save)
    report = json.loads(save.read_text()) if save.exists() else {}
    for g in GRAPHS:
        weights = window_weights(g, win, coh, sp)
        allg = build(ab, weights, yw, np.arange(len(yw)))
        for m in MODELS:
            key = f"{g}:{m}"
            if key in report:  # resumable
                continue
            rs = []
            for rep in range(args.repeats):
                prob = np.zeros((len(u), 3))
                for a, b in StratifiedKFold(5, shuffle=True, random_state=rep).split(u, ys):
                    tr = np.concatenate([rows_of[s] for s in u[a]])
                    te_rows = [rows_of[s] for s in u[b]]
                    p = fit_predict(m, [allg[i] for i in tr], [allg[i] for r in te_rows for i in r],
                                    yw[tr], device, rep)
                    k = 0
                    for j, r in zip(b, te_rows):
                        prob[j] = p[k:k + len(r)].mean(0)
                        k += len(r)
                pred = prob.argmax(1)
                rs.append({"macro_f1": f1_score(ys, pred, average="macro"),
                           "bal_acc": balanced_accuracy_score(ys, pred),
                           "macro_auc": roc_auc_score(ys, prob, multi_class="ovr")})
            report[key] = {k: [float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs]))] for k in rs[0]}
            save.write_text(json.dumps(report, indent=2))
            print(f"{g:<10} {m:<6} F1 {report[key]['macro_f1'][0]:.3f}  bal-acc {report[key]['bal_acc'][0]:.3f}  "
                  f"AUC {report[key]['macro_auc'][0]:.3f}", flush=True)


if __name__ == "__main__":
    main()
