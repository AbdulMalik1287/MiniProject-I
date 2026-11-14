"""Phase 2b: spatio-temporal GATv2 -> LSTM over a subject's windows.

Idea from DTS-GAN (Yan et al., Sci Rep 2025): a graph network reads each time
window and an LSTM reads the sequence. Here the graph part is the Park-tuned
GATv2 (same node/edge features and readout as the baseline), so the only change
is that windows are no longer scored independently - the model sees how a
subject's connectivity evolves across a minute of recording.

    window graph (19 nodes, coherence edges) -> GATv2 x2 -> mean+max pool (40-d)
    -> LSTM(40 -> 32) over L consecutive windows -> logit for the chunk

Training samples are chunks of L=6 consecutive clean windows (1 min) per
subject, re-cut from a random offset every epoch. At prediction every window
gets the probability of the chunk it falls in, and subjects are scored by the
mean over their windows exactly as for the other models. A subject with fewer
than L windows forms one shorter chunk.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch_geometric.data import Batch
from torch_geometric.nn import global_max_pool, global_mean_pool

from eeg.experiments import per_disorder as D
from eeg.models.train import SEED

L = 6


class STGNN(nn.Module):
    def __init__(self, cfg, hidden=32):
        super().__init__()
        self.g = D.make_model(cfg)
        self.lstm = nn.LSTM(self.g.fc1.in_features, hidden, batch_first=True)
        self.out = nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden, 1))

    def embed(self, b):
        g = self.g
        x = F.leaky_relu(g._conv(g.conv1, b.x, b.edge_index, b.edge_attr))
        x = F.leaky_relu(g.bn(g._conv(g.conv2, x, b.edge_index, b.edge_attr)))
        return torch.cat([global_mean_pool(x, b.batch), global_max_pool(x, b.batch)], 1)

    def forward(self, b, lengths):
        e = self.embed(b)  # (sum(lengths), 40), chunks laid out back to back
        seqs = pad_sequence(list(e.split(lengths)), batch_first=True)
        _, (h, _) = self.lstm(pack_padded_sequence(seqs, torch.tensor(lengths), batch_first=True,
                                                   enforce_sorted=False))
        return self.out(h[-1]).squeeze(-1)


def chunks(idx, subject, offset_rng=None):
    """Consecutive runs of up to L rows per subject (rows of one subject are stored in time order)."""
    out = []
    for s in np.unique(subject[idx]):
        r = np.sort(idx[subject[idx] == s])
        if len(r) <= L:
            out.append(r)
            continue
        start = int(offset_rng.integers(0, L)) if offset_rng is not None else 0
        if start:
            out.append(r[:start])
        out += [r[i:i + L] for i in range(start, len(r), L)]
    return out


def _graphs(cfg, ab, coh, idx):
    ds = D._dataset(cfg, ab, coh, np.zeros(len(ab), dtype=np.int64), idx)
    return {int(r): ds[i] for i, r in enumerate(idx)}


def _batch(graphs, cs):
    return Batch.from_data_list([graphs[int(r)] for c in cs for r in c]), [len(c) for c in cs]


def fit(cfg, ab, coh, y, subject, idx, device, epochs=20, batch=32):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    graphs = _graphs(cfg, ab, coh, idx)
    model = STGNN(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    pos = (y[idx] == 1).mean()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / max(pos, 1e-6), device=device))
    for _ in range(epochs):
        model.train()
        cs = chunks(idx, subject, rng)
        order = rng.permutation(len(cs))
        for k in range(0, len(cs), batch):
            part = [cs[j] for j in order[k:k + batch]]
            b, lengths = _batch(graphs, part)
            t = torch.tensor([float(y[c[0]]) for c in part], device=device)
            opt.zero_grad()
            loss_fn(model(b.to(device), lengths), t).backward()
            opt.step()
    return model


@torch.no_grad()
def predict_proba(model, cfg, ab, coh, subject, idx, device, batch=64):
    model.eval()
    graphs = _graphs(cfg, ab, coh, idx)
    cs = chunks(idx, subject)
    prob = {}
    for k in range(0, len(cs), batch):
        part = cs[k:k + batch]
        b, lengths = _batch(graphs, part)
        p = torch.sigmoid(model(b.to(device), lengths).double()).cpu().numpy()
        for c, pc in zip(part, p):
            for r in c:
                prob[int(r)] = pc
    return np.array([prob[int(r)] for r in idx])


if __name__ == "__main__":  # self-check: learns a planted band-power difference from sequences
    rng = np.random.default_rng(0)
    n_sub, w = 40, 12
    subject = np.repeat([f"s{i}" for i in range(n_sub)], w)
    y = np.repeat(np.arange(n_sub) % 2, w).astype(np.int64)
    ab = rng.normal(size=(len(y), 19, 6)).astype(np.float32)
    ab[y == 1, :, 2] += 1.5
    coh = rng.uniform(size=(len(y), 19, 19, 6)).astype(np.float32)
    coh = (coh + coh.transpose(0, 2, 1, 3)) / 2
    cfg = {**D.GRID[-1]}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, te = np.arange(0, 28 * w), np.arange(28 * w, n_sub * w)
    m = fit(cfg, ab, coh, y, subject, tr, dev, epochs=5)
    p = predict_proba(m, cfg, ab, coh, subject, te, dev)
    acc = ((p > 0.5) == y[te]).mean()
    print(f"st_gnn selfcheck acc {acc:.2f}")
    assert acc > 0.9
