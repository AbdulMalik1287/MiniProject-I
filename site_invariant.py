"""Phase 4: three ways to stop a GNN learning the hospital instead of the disease.

site_probe.py shows a linear model names the hospital from healthy controls
alone 78-100% of the time, so cross-hospital failure is a shortcut problem.

align   Euclidean alignment (He & Wu 2019), per SUBJECT: whiten each subject's raw
        windows by the inverse square root of their own mean spatial covariance,
        then recompute the same spectral features. Label-free and uses no other
        subject, so it is applied identically to training and test subjects.
combat  ComBat (Johnson 2007, neuroCombat) on every window feature with batch =
        hospital. Fitted on all hospitals of the task including the held-out one,
        WITHOUT labels (transductive). No disease covariate can be given - test
        labels are unknown - so a hospital whose patient share differs is partly
        "corrected" towards the others. Stated as a limitation.
dann    Domain-adversarial training (Ganin et al. 2016): the Park-tuned GATv2, plus
        a hospital classifier on its pooled graph embedding behind a gradient
        reversal layer. The held-out hospital's windows enter the hospital loss
        unlabeled (standard unsupervised domain adaptation); the disease loss
        sees training rows only.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

import per_disorder as D
from train import SEED

# ------------------------------------------------------------------ alignment


def euclidean_align(win, subject, spectral_features, ridge=1e-3):
    """Per-subject whitening of raw windows, then features. Returns (ab, coh)."""
    ab = np.zeros((len(win), win.shape[1], 6), np.float32)
    coh = np.zeros((len(win), win.shape[1], win.shape[1], 6), np.float32)
    for s in np.unique(subject):
        rows = np.where(subject == s)[0]
        w = win[rows].astype(np.float64)
        w -= w.mean(-1, keepdims=True)
        c = np.einsum("wct,wdt->cd", w, w) / (w.shape[0] * w.shape[2])
        c += ridge * np.trace(c) / len(c) * np.eye(len(c))  # average reference makes C rank 18
        vals, vecs = np.linalg.eigh(c)
        r = vecs @ np.diag(vals ** -0.5) @ vecs.T
        a, h = spectral_features(np.einsum("cd,wdt->wct", r, w))
        ab[rows], coh[rows] = a, h
    return ab, coh


def _combat_block(flat, site):
    from neuroCombat import neuroCombat
    import pandas as pd
    flat = flat.astype(np.float64)
    var_ok = np.all([flat[site == s].std(0) > 1e-8 for s in np.unique(site)], axis=0)
    out = flat.copy()
    out[:, var_ok] = neuroCombat(dat=flat[:, var_ok].T, covars=pd.DataFrame({"site": site}),
                                 batch_col="site")["data"].T
    return out


def combat(ab, coh, site):
    """ComBat across hospitals, band power and coherence as separate blocks:
    the empirical-Bayes prior pools site effects across features, and the two
    families shift differently, so one joint fit under-corrects the smaller one."""
    n = len(ab)
    a = _combat_block(ab.reshape(n, -1), site).reshape(ab.shape).astype(np.float32)
    c = _combat_block(coh.reshape(n, -1), site).reshape(coh.shape)
    return a, np.clip(c, 0, 1).astype(np.float32)  # keep coherence a valid weight


# ----------------------------------------------------------------------- DANN


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class DANN(nn.Module):
    """The tuned GraphNet, exposing its pooled embedding to a hospital head."""

    def __init__(self, cfg, n_sites):
        super().__init__()
        self.g = D.make_model(cfg)
        dim = self.g.fc1.in_features
        self.site_head = nn.Sequential(nn.Linear(dim, 16), nn.ReLU(), nn.Linear(16, n_sites))

    def embed(self, x, edge_index, edge_attr, batch):
        g = self.g
        x = F.leaky_relu(g._conv(g.conv1, x, edge_index, edge_attr))
        x = F.leaky_relu(g.bn(g._conv(g.conv2, x, edge_index, edge_attr)))
        if g.pool == "add":
            return global_add_pool(x, batch)
        if g.pool == "mean":
            return global_mean_pool(x, batch)
        return torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], 1)

    def disease(self, e):
        e = F.dropout(e, p=0.2, training=self.training)
        return self.g.fc2(F.leaky_relu(self.g.fc1(e)))

    def forward(self, x, edge_index, edge_attr, batch):  # used by D.predict_proba
        return self.disease(self.embed(x, edge_index, edge_attr, batch))


def fit_dann(cfg, ab, coh, y, site, tr, te, device, lam_max=1.0):
    """Disease loss on tr; hospital loss on tr and the unlabeled te rows."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    sites = np.unique(site[np.concatenate([tr, te])])
    if len(sites) < 2:  # nothing to be invariant to: plain training
        return D.fit(cfg, ab, coh, y, tr, device)
    sid = np.searchsorted(sites, site)
    is_tr = np.zeros(len(y), bool)
    is_tr[tr] = True
    rows = np.concatenate([tr, te])
    # the dataset's y carries the row number; labels are looked up per batch
    loader = DataLoader(D._dataset(cfg, ab, coh, np.arange(len(y)), rows),
                        batch_size=cfg["batch_size"], shuffle=True)
    model = DANN(cfg, len(sites)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    counts = np.bincount(y[tr], minlength=2)
    w_dis = torch.tensor(counts.sum() / (2.0 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    yt, st, trt = (torch.tensor(a, device=device) for a in (y, sid, is_tr))
    steps, step = cfg["epochs"] * len(loader), 0
    for _ in range(cfg["epochs"]):
        model.train()
        for b in loader:
            b = b.to(device)
            r = b.y
            lam = lam_max * (2 / (1 + np.exp(-10 * step / steps)) - 1)  # Ganin schedule
            step += 1
            e = model.embed(b.x, b.edge_index, b.edge_attr, b.batch)
            loss = F.cross_entropy(model.site_head(GradReverse.apply(e, lam)), st[r])
            m = trt[r]
            if m.any():
                loss = loss + F.cross_entropy(model.disease(e[m]), yt[r][m], weight=w_dis)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


if __name__ == "__main__":  # self-check: DANN trains, alignment whitens, ComBat removes a site offset
    rng = np.random.default_rng(0)
    n = 80
    ab = rng.normal(size=(n, 19, 6)).astype(np.float32)
    coh = rng.uniform(size=(n, 19, 19, 6)).astype(np.float32)
    coh = (coh + coh.transpose(0, 2, 1, 3)) / 2
    y = (np.arange(n) % 2).astype(np.int64)
    site = np.where(np.arange(n) < 40, "a", "b")
    ab[site == "b"] += 3.0
    a2, _ = combat(ab, coh, site)
    gap = abs(a2[site == "a"].mean() - a2[site == "b"].mean())
    assert gap < 0.1, f"ComBat left a site offset of {gap}"
    win = rng.normal(size=(20, 19, 1280)).astype(np.float16)
    win[:, 0] *= 10
    import raw_eeg as R
    a3, _ = euclidean_align(win, np.repeat(["s1", "s2"], 10), R.spectral_features)
    assert np.isfinite(a3).all()
    cfg = {**D.GRID[-1], "epochs": 2}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = fit_dann(cfg, ab, coh, y, site, np.arange(0, 40), np.arange(40, 80), dev)
    p = D.predict_proba(m, cfg, ab, coh, np.arange(40, 80), dev)
    assert p.shape == (40,) and np.isfinite(p).all()
    print("site_invariant selfcheck ok")
