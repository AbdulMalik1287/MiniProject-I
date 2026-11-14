"""DT-GCN (Liu & Meng, Information 2026, 17(7):703) ported from its released code, 3-class AD/FTD/HC.

Source: github.com/vjieajf/AD-FTD-EEG-DT-GCN @ 7efabd2, script
"图网络注意力改良（动态阈值3分类）.py". The released scripts cannot run as published
(the copula matrix file the model loads is never written, an IndexError in the
extractors, Fs hard-coded to 500 after resampling to 250), so this is a port:

    connectivity  PLV per window (Hilbert phase of our cleaned 1-45 Hz windows), averaged
                  per subject -> one 19x19 matrix per subject. Their main input (Gumbel
                  copula) is not reproducible from the released code; their partial
                  correction against channel Fp1 is not applied.
    graph         per-matrix min-max; binary edges where value > percentile(thr*100)
                  (their lines 167-186); node features = [identity(19), degree] (20-d)
    model         GCN 20-64-32-16 (BatchNorm, ReLU, dropout 0.5) -> GATConv(16,16,heads=8,
                  concat=False, dropout=0.6) -> global add pool -> Linear(16,3)
    training      Adam lr 0.01 wd 1e-4, batch 8, 500 epochs, cross-entropy
    dynamic thr   their rule (lines 610-627): delta = (prev_loss - loss)*0.1*0.4*0.4*10,
                  thr = 0.5*thr + 0.5*(thr + delta), clipped to [0.3, 0.7], start 0.5;
                  held-out graphs at 0.5. Their reconstruction loss has no gradient
                  (identity vs a fixed matrix) and is omitted; so is the contrastive term.

Protocols:
    paper    KFold(5, shuffle, random_state=18); per fold the BEST held-out accuracy over
             the 500 epochs (their lines 594-603: the held-out fold is their validation set)
    strict   StratifiedKFold(5) x 5 seeds; the model after the last epoch only; nothing is
             chosen by looking at held-out subjects
    inner    StratifiedKFold(5); epoch chosen on a 20% inner validation split of the training
             subjects; the test fold is predicted with that epoch's model (--inner-only)
Each with the dynamic and the fixed (0.5) threshold.

    python dtgcn.py --out ~/eeg/raw2
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import hilbert
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import BatchNorm, GATConv, GCNConv, global_add_pool

import raw_eeg as R

CLASSES = ["control", "ftd", "alzheimers"]
EPOCHS = 500


def subject_plv(win, subject):
    """(windows, 19, T) -> per-subject mean PLV (S, 19, 19), subjects in np.unique order."""
    u = np.unique(subject)
    out = np.zeros((len(u), win.shape[1], win.shape[1]))
    for i, s in enumerate(u):
        ph = np.angle(hilbert(win[subject == s].astype(np.float64), axis=-1))  # (w, 19, T)
        z = np.exp(1j * ph)
        plv = np.abs(np.einsum("wct,wdt->wcd", z, z.conj()) / ph.shape[-1])
        out[i] = plv.mean(0)
    return u, out


_CACHE = {}


def graph(adj, thr, label):
    key = (adj.tobytes(), round(thr, 4), label)  # thresholds move slowly; rebuilding 70 graphs per epoch dominated
    if key not in _CACHE:
        _CACHE[key] = _graph(adj, thr, label)
    return _CACHE[key]


def _graph(adj, thr, label):
    a = (adj - adj.min()) / (adj.max() - adj.min() + 1e-12)
    m = a > np.percentile(a, thr * 100)
    np.fill_diagonal(m, False)
    src, dst = np.nonzero(m)
    deg = m.sum(1, keepdims=True).astype(np.float32)
    x = np.concatenate([np.eye(len(a), dtype=np.float32), deg], 1)
    return Data(x=torch.from_numpy(x), edge_index=torch.tensor(np.stack([src, dst]), dtype=torch.long),
                y=torch.tensor([label]))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        dims = [20, 64, 32, 16]
        self.convs = nn.ModuleList(GCNConv(a, b) for a, b in zip(dims, dims[1:]))
        self.bns = nn.ModuleList(BatchNorm(b) for b in dims[1:])
        self.gat = GATConv(16, 16, heads=8, concat=False, dropout=0.6)
        self.out = nn.Linear(16, 3)

    def forward(self, b):
        x = b.x
        for conv, bn in zip(self.convs, self.bns):
            x = F.dropout(F.relu(bn(conv(x, b.edge_index))), 0.5, self.training)
        x = self.gat(x, b.edge_index)
        return self.out(global_add_pool(x, b.batch))


@torch.no_grad()
def predict(model, adj, idx, device):
    model.eval()
    b = next(iter(DataLoader([graph(adj[i], 0.5, 0) for i in idx], batch_size=len(idx))))
    return model(b.to(device)).argmax(1).cpu().numpy()


def train(adj, y, tr, te, dynamic, device, seed, track_best=False):
    """Returns (last-epoch predictions on te, best held-out accuracy over epochs)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Net().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    thr, prev, best = 0.5, None, 0.0
    for _ in range(EPOCHS):
        model.train()
        loader = DataLoader([graph(adj[i], thr, int(y[i])) for i in tr], batch_size=8, shuffle=True)
        tot = 0.0
        for b in loader:
            b = b.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(b), b.y)
            loss.backward()
            opt.step()
            tot += loss.item() * b.num_graphs
        loss = tot / len(tr)
        if dynamic and prev is not None:
            delta = (prev - loss) * 0.1 * 0.4 * 0.4 * 10
            thr = float(np.clip(0.5 * thr + 0.5 * (thr + delta), 0.3, 0.7))
        prev = loss
        if track_best:  # their selection: best held-out accuracy over epochs (paper protocol only)
            best = max(best, accuracy_score(y[te], predict(model, adj, te, device)))
    return predict(model, adj, te, device), best


def train_inner(adj, y, tr, te, dynamic, device, seed):
    """Epoch chosen on an inner validation split of the TRAINING subjects (20%, stratified);
    the test fold is predicted with the model of that epoch and never looked at."""
    from sklearn.model_selection import StratifiedShuffleSplit
    itr, ival = next(StratifiedShuffleSplit(1, test_size=0.2, random_state=seed).split(tr, y[tr]))
    itr, ival = tr[itr], tr[ival]
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Net().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    thr, prev, best, best_pred = 0.5, None, -1.0, None
    for _ in range(EPOCHS):
        model.train()
        loader = DataLoader([graph(adj[i], thr, int(y[i])) for i in itr], batch_size=8, shuffle=True)
        tot = 0.0
        for b in loader:
            b = b.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(b), b.y)
            loss.backward()
            opt.step()
            tot += loss.item() * b.num_graphs
        loss = tot / len(itr)
        if dynamic and prev is not None:
            thr = float(np.clip(0.5 * thr + 0.5 * (thr + (prev - loss) * 0.1 * 0.4 * 0.4 * 10), 0.3, 0.7))
        prev = loss
        acc = accuracy_score(y[ival], predict(model, adj, ival, device))
        if acc > best:
            best, best_pred = acc, predict(model, adj, te, device)
    return best_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/eeg/raw2")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--save", default="results/dtgcn.json")
    ap.add_argument("--inner-only", action="store_true", help="only the inner-validation protocol, merged into --save")
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, subject, groups, _, win = R.load_features(out, ["ds004504"], raw=True)
    u, adj = subject_plv(win, subject)
    y = np.array([CLASSES.index(groups[subject == s][0]) for s in u])
    print(f"[dtgcn] {len(u)} subjects, PLV matrices ready", flush=True)
    save = Path(args.save)
    report = json.loads(save.read_text()) if save.exists() else {}
    if args.inner_only:
        for dynamic in (True, False):
            rs = []
            for rep in range(args.repeats):
                pred = np.zeros(len(y), dtype=int)
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(adj, y):
                    pred[te] = train_inner(adj, y, tr, te, dynamic, device, rep)
                rs.append({"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
                           "bal_acc": balanced_accuracy_score(y, pred)})
            name = f"inner_val_{'dynamic' if dynamic else 'fixed'}"
            report[name] = {k: [float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs]))] for k in rs[0]}
            print(name, json.dumps(report[name]), flush=True)
            save.write_text(json.dumps(report, indent=2))
        return
    for dynamic in (True, False):
        name = "dynamic" if dynamic else "fixed"
        # paper protocol: their KFold, best held-out accuracy over epochs
        bests = [train(adj, y, tr, te, dynamic, device, 18, track_best=True)[1]
                 for tr, te in KFold(5, shuffle=True, random_state=18).split(adj)]
        report[f"paper_protocol_{name}"] = {"accuracy_best_epoch": [float(np.mean(bests)), float(np.std(bests))]}
        # strict: stratified, last epoch, no held-out selection
        rs = []
        for rep in range(args.repeats):
            pred = np.zeros(len(y), dtype=int)
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(adj, y):
                pred[te] = train(adj, y, tr, te, dynamic, device, rep)[0]
            rs.append({"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
                       "bal_acc": balanced_accuracy_score(y, pred)})
        report[f"strict_{name}"] = {k: [float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs]))]
                                    for k in rs[0]}
        print(name, json.dumps({k: v for k, v in report.items() if name in k}), flush=True)
    report["paper_reported"] = {"accuracy": "0.82 +/- 0.07", "macro_f1": "0.79 +/- 0.07", "input": "Gumbel copula"}
    save.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
