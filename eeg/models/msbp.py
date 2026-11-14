"""MSBP-GNN (Geng et al., Sensors 26(9):2868, 2026), re-implemented from the paper's text.

No code was released, so this follows the described architecture:
    raw 2 s windows (19 ch) -> per-electrode multi-scale depthwise temporal convs
    -> conv embedding + attention pooling over time = 64-d node features
    -> two hemisphere graphs of 11 nodes (Fz, Cz, Pz shared), edges = cosine
       similarity of the node signals, ReLU, top-K=10 per node, rebuilt per sample
    -> 2 ChebConv (K=3) layers with residuals -> mean pool per hemisphere
    -> attention fusion of the two hemispheres -> logit

Deviations (ours, stated in the report): the paper samples at 256 Hz and uses
kernels 103/51/25; our pipeline resamples to 128 Hz, so kernels are halved to
51/25/13 to keep the same time spans. Epochs are fixed (no early stopping on
test folds). Each 10 s window of eeg/data/raw_eeg.py is cut into five 2 s windows; a 10 s
window's probability is the mean of its five, and subjects are scored by the
mean over their windows exactly as for the other models.
"""
import numpy as np
import torch
from torch import nn
from torch_geometric.nn import ChebConv

SF, SUB_S = 128, 2
CHANNELS = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T3", "C3", "CZ", "C4", "T4",
            "T5", "P3", "PZ", "P4", "T6", "O1", "O2"]
LEFT = ["FP1", "F7", "F3", "T3", "C3", "T5", "P3", "O1", "FZ", "CZ", "PZ"]
RIGHT = ["FP2", "F8", "F4", "T4", "C4", "T6", "P4", "O2", "FZ", "CZ", "PZ"]
CFG = {"epochs": 12, "lr": 1e-3, "batch": 256, "topk": 10, "dropout": 0.3, "hidden": 64}


class TemporalEncoder(nn.Module):
    """(N, T) single-electrode signals -> (N, hidden)."""

    def __init__(self, hidden, kernels=(51, 25, 13), f=8):
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Sequential(nn.Conv1d(1, f, k, padding=k // 2), nn.BatchNorm1d(f), nn.ELU())
            for k in kernels)
        self.embed = nn.Sequential(nn.Conv1d(f * len(kernels), hidden, 7, stride=2, padding=3),
                                   nn.BatchNorm1d(hidden), nn.ELU())
        self.att = nn.Conv1d(hidden, 1, 1)

    def forward(self, x):
        h = self.embed(torch.cat([b(x.unsqueeze(1)) for b in self.branches], 1))  # (N, H, T')
        return (h * torch.softmax(self.att(h), -1)).sum(-1)


def topk_edges(sig, k):
    """sig (B, n, T) -> edge_index (2, E) and weights (E,) over B disjoint n-node graphs."""
    b, n, _ = sig.shape
    z = nn.functional.normalize(sig, dim=-1)
    sim = torch.relu(z @ z.transpose(1, 2))
    sim = sim - 2 * torch.eye(n, device=sig.device)  # no self-loops
    w, j = sim.topk(min(k, n - 1), dim=-1)  # (B, n, k)
    i = torch.arange(n, device=sig.device).view(1, n, 1).expand_as(j)
    off = (torch.arange(b, device=sig.device) * n).view(b, 1, 1)
    keep = w > 0
    src, dst = (j + off)[keep], (i + off)[keep]
    return torch.stack([src, dst]), w[keep]


class MSBP(nn.Module):
    def __init__(self, hidden=64, topk=10, dropout=0.3):
        super().__init__()
        self.idx = {s: torch.tensor([CHANNELS.index(c) for c in side]) for s, side in
                    (("l", LEFT), ("r", RIGHT))}
        self.enc = TemporalEncoder(hidden)
        self.convs = nn.ModuleList(ChebConv(hidden, hidden, K=3) for _ in range(2))
        self.fuse = nn.Linear(hidden, 1)
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.topk, self.drop = topk, nn.Dropout(dropout)

    def hemisphere(self, x, feats, side):
        idx = self.idx[side].to(x.device)
        sig, h = x[:, idx], feats[:, idx]  # (B, 11, T), (B, 11, H)
        b, n, hd = h.shape
        ei, ew = topk_edges(sig, self.topk)
        h = h.reshape(b * n, hd)
        for conv in self.convs:
            h = h + self.drop(nn.functional.elu(conv(h, ei, ew)))
        return h.view(b, n, hd).mean(1)

    def forward(self, x):  # x (B, 19, T)
        b, c, t = x.shape
        feats = self.enc(x.reshape(b * c, t)).view(b, c, -1)
        hs = torch.stack([self.hemisphere(x, feats, s) for s in ("l", "r")], 1)  # (B, 2, H)
        fused = (hs * torch.softmax(self.fuse(hs), 1)).sum(1)
        return self.out(fused).squeeze(-1)


def split_2s(win):
    """(w, 19, 10 s) float16 -> (w*5, 19, 2 s) float32, per-window z-scored per channel."""
    n = SF * SUB_S
    w = win.astype(np.float32)
    k = w.shape[-1] // n
    w = w[..., :k * n].reshape(len(w), w.shape[1], k, n).transpose(0, 2, 1, 3).reshape(-1, w.shape[1], n)
    return (w - w.mean(-1, keepdims=True)) / (w.std(-1, keepdims=True) + 1e-6), k


def fit(win, y, idx, device, cfg=CFG, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    x, k = split_2s(win[idx])
    t = np.repeat(y[idx], k).astype(np.float32)
    model = MSBP(cfg["hidden"], cfg["topk"], cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    pos = torch.tensor((len(t) - t.sum()) / max(t.sum(), 1), device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos)
    xt, tt = torch.from_numpy(x), torch.from_numpy(t)
    model.train()
    for _ in range(cfg["epochs"]):
        for b in torch.randperm(len(xt)).split(cfg["batch"]):
            opt.zero_grad()
            loss_fn(model(xt[b].to(device)), tt[b].to(device)).backward()
            opt.step()
    return model


@torch.no_grad()
def predict_proba(model, win, idx, device, batch=512):
    """Probability per 10 s window = mean of its 2 s sub-window probabilities."""
    model.eval()
    x, k = split_2s(win[idx])
    xt = torch.from_numpy(x)
    p = torch.cat([torch.sigmoid(model(xt[i:i + batch].to(device)).double()).cpu()
                   for i in range(0, len(xt), batch)]).numpy()
    return p.reshape(len(idx), k).mean(1)


if __name__ == "__main__":  # self-check: learns a planted alpha-power difference
    rng = np.random.default_rng(0)
    n = 120
    tt = np.arange(10 * SF) / SF
    y = (np.arange(n) % 2).astype(np.int64)
    win = rng.normal(size=(n, 19, 10 * SF))
    win[y == 1] += 2 * np.sin(2 * np.pi * 10 * tt)
    win = win.astype(np.float16)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, te = np.arange(0, 80), np.arange(80, n)
    m = fit(win, y, tr, dev, {**CFG, "epochs": 5})
    p = predict_proba(m, win, te, dev)
    acc = ((p > 0.5) == y[te]).mean()
    print(f"selfcheck acc {acc:.2f}, params {sum(q.numel() for q in m.parameters())}")
    assert acc > 0.9, "MSBP failed to learn a planted signal"
