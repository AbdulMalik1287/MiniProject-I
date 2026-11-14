"""Transformer / attention models trained on every dataset of a disorder at once (team plan, 2nd column).

Data: per task, all same-condition datasets pooled (alzheimers / ftd / alzheimers_vs_ftd:
ds004504 + BrainLat; mood: Mumtaz + UNM; schizophrenia: ASEEG + Warsaw + Semmelweis).

Folds: one StratifiedKFold(5, seed 42) over all subjects of the task, stratified by
site x label, identical for every model. Each test subject is scored once by a model
trained on the other sites plus the rest of its own site. AUC is reported per site
(primary - comparable to the within-site tables) and over all subjects.

Inputs are scale-free: graph models see log RELATIVE band power and coherence; LaBraM
sees windows z-scored per recording and channel. That removes the BrainLat
patient/control amplitude gap (rs-HEP vs rs files), but other differences between
those file types can remain, so BrainLat AD/FTD-vs-control numbers carry that caveat.
Band power is also z-scored per site, without labels (transductive, as pooled.py).

Models, fixed before running:
    lr_base     logistic regression on subject-mean band power + coherence (reference)
    gatv2       Park-tuned GATv2 on window graphs, subject = mean window probability (reference)
    gt          Graph Transformer (TransformerConv), same widths/training as gatv2
    gt_mil      Graph Transformer window encoder + gated-attention MIL pooling over a
                subject's windows (Ilse et al. 2018), trained on subjects (bags)
    gt_mil_dro  gt_mil with GroupDRO over site x label groups (Sagawa et al. 2020)
    labram_ft   LaBraM (pretrained), last 2 of 12 blocks + fc_norm fine-tuned with a linear
                head on windows, subject = mean window logit

    --loho      leave-one-hospital-out: one model per held-out site, trained on all other
                sites; the per-site AUC is then pure transfer to an unseen hospital. The
                held-out site's band power is still z-scored on its own (unlabeled) subjects.

    python pooled_deep.py --out ~/eeg/raw2 [--models gt,gt_mil] [--tasks mood] [--loho]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter, softmax

import park as P
import per_disorder as D
import raw_eeg as R
from baselines import lr_probe
from train import build_model

TASKS = ["alzheimers", "ftd", "alzheimers_vs_ftd", "mood", "schizophrenia"]
MODELS = ["lr_base", "gatv2", "gt", "gt_mil", "gt_mil_dro", "labram_ft"]
SEED = 42
BALANCE = False  # --balance: every site x label group gets equal total weight in training


def group_w(site, y):
    """Weight 1/|group| per item, groups = site x label, normalised to mean 1."""
    key = np.char.add(np.asarray(site).astype(str), np.asarray(y).astype(str))
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]
    return w / w.mean()


# ------------------------------------------------------------------ data


def load_task(out, task, raw, feat="features"):
    pos, neg, sites = R.TASKS[task]
    ab, coh, subject, groups, dsname, win = R.load_features(out, sites, raw=raw, feat=feat)
    keep, y = R.task_rows(groups, pos, neg)
    ab, coh, subject, dsname = ab[keep], coh[keep], subject[keep], dsname[keep]
    win = win[keep] if raw else None
    for s in np.unique(dsname):  # per-site z-score of band power, no labels
        m = dsname == s
        ab[m] = (ab[m] - ab[m].mean(0)) / (ab[m].std(0) + 1e-6)
    u = np.unique(subject)
    rows = {s: np.where(subject == s)[0] for s in u}
    ys = np.array([y[rows[s][0]] for s in u])
    site = np.array([dsname[rows[s][0]] for s in u])
    return dict(ab=ab, coh=coh, win=win, y=y, subject=subject, u=u, rows=rows, ys=ys, site=site, wsite=dsname)


def folds(d, loho=False):
    if loho:  # leave one hospital out: train on every other site, test on all of this one
        return [(np.where(d["site"] != st)[0], np.where(d["site"] == st)[0]) for st in np.unique(d["site"])]
    strat = np.char.add(d["site"].astype(str), d["ys"].astype(str))
    return list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(d["u"], strat))


def graphs(d, cfg):
    ds = P.ParkGraphDataset(d["ab"], d["coh"], d["y"], np.arange(len(d["y"])), D.EDGE_INDEX,
                            cfg["edge_mode"], cfg["node_features"], cfg["topk"])
    return [ds[i] for i in range(len(ds))]


def pos_weight(y, device):
    return torch.tensor((y == 0).sum() / max((y == 1).sum(), 1), dtype=torch.float32, device=device)


# ------------------------------------------------------------------ models


def net(cfg, kind):
    return build_model(kind, P.N_BANDS, n_nodes=P.N_NODES, n_classes=2, normalize=True,
                       in_dim=P.node_feature_dim(cfg["node_features"]), pool=cfg["pool"])


def window_gnn(kind, cfg, g, d, tr_s, te_s, device):
    """Window-level training (as per_disorder.fit); subject score = mean window P(positive)."""
    torch.manual_seed(SEED)
    tr = np.concatenate([d["rows"][s] for s in tr_s])
    model = net(cfg, kind).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    cnt = np.bincount(d["y"][tr], minlength=2)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cnt.sum() / (2.0 * cnt), dtype=torch.float32, device=device))
    kw = {"shuffle": True}
    if BALANCE:  # sampler draws every site x label group equally often, so no class weight on top
        gen = torch.Generator().manual_seed(SEED)
        kw = {"sampler": torch.utils.data.WeightedRandomSampler(
            group_w(d["wsite"][tr], d["y"][tr]), len(tr), replacement=True, generator=gen)}
        crit = nn.CrossEntropyLoss()
    for _ in range(R.RAW_EPOCHS):
        model.train()
        for b in DataLoader([g[i] for i in tr], batch_size=cfg["batch_size"], **kw):
            b = b.to(device)
            opt.zero_grad()
            crit(model(b.x, b.edge_index, b.edge_attr, b.batch), b.y).backward()
            opt.step()
    model.eval()
    out = []
    with torch.no_grad():
        for s in te_s:
            b = Batch.from_data_list([g[i] for i in d["rows"][s]]).to(device)
            z = model(b.x, b.edge_index, b.edge_attr, b.batch).double()
            out.append(float((z[:, 1] - z[:, 0]).mean()))  # mean logit margin: no float32 saturation
    return np.array(out)


class MIL(nn.Module):
    """Graph Transformer encoder per window -> gated attention over the subject's windows -> logit."""

    def __init__(self, cfg):
        super().__init__()
        self.enc = net(cfg, "transformer")
        h = 40 if cfg["pool"] == "meanmax" else 20
        self.V, self.U, self.w = nn.Linear(h, 16), nn.Linear(h, 16), nn.Linear(16, 1)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(h, 10), nn.LeakyReLU(), nn.Linear(10, 1))

    def forward(self, b, bag):
        h = self.enc.embed(b.x, b.edge_index, b.edge_attr, b.batch)  # (windows, h)
        a = softmax(self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(1), bag)
        return self.head(scatter(a[:, None] * h, bag, dim=0, reduce="sum")).squeeze(1)


def mil(cfg, g, d, tr_s, te_s, device, dro, epochs=60, bags=16, sub=16):
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    model = MIL(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sidx = {s: i for i, s in enumerate(d["u"])}
    ytr = np.array([d["ys"][sidx[s]] for s in tr_s])
    grp_names = sorted({f"{d['site'][sidx[s]]}:{d['ys'][sidx[s]]}" for s in tr_s})
    gid = np.array([grp_names.index(f"{d['site'][sidx[s]]}:{d['ys'][sidx[s]]}") for s in tr_s])
    q = torch.ones(len(grp_names), device=device) / len(grp_names)
    pw = pos_weight(ytr, device)

    def batch(subjects, sample):
        data, bag = [], []
        for k, s in enumerate(subjects):
            r = d["rows"][s]
            if sample and len(r) > sub:
                r = rng.choice(r, sub, replace=False)  # random window subset per bag: augmentation
            data += [g[i] for i in r]
            bag += [k] * len(r)
        return Batch.from_data_list(data).to(device), torch.tensor(bag, device=device)

    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(tr_s))
        for k in range(0, len(order), bags):
            j = order[k:k + bags]
            b, bag = batch(tr_s[j], True)
            yb = torch.tensor(ytr[j], dtype=torch.float32, device=device)
            loss = F.binary_cross_entropy_with_logits(model(b, bag), yb, pos_weight=pw, reduction="none")
            if dro:  # GroupDRO: exponentiated-gradient weights on the worst site x label group
                gj = torch.tensor(gid[j], device=device)
                gl = torch.zeros(len(grp_names), device=device).index_add(0, gj, loss)
                gn = torch.zeros(len(grp_names), device=device).index_add(0, gj, torch.ones_like(loss))
                present = gn > 0
                gl = torch.where(present, gl / gn.clamp(min=1), torch.zeros_like(gl))
                q = q * torch.exp(0.01 * gl.detach())
                q = q / q.sum()
                loss = (q * gl).sum()
            else:
                loss = loss.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return np.array([float(model(*batch([s], False)).item()) for s in te_s])


def labram_ft(d, tr_s, te_s, device, state, epochs=3, bs=32):
    from braindecode.models import Labram
    from scipy.signal import resample_poly
    names = [c.upper() for c in P.CHANNELS]
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    m = Labram.from_pretrained("braindecode/labram-pretrained", n_times=2000)
    m.load_state_dict(state)
    m = m.to(device)
    for p in m.parameters():
        p.requires_grad = False
    tune = [p for blk in m.blocks[-2:] for p in blk.parameters()] + list(m.fc_norm.parameters())
    for p in tune:
        p.requires_grad = True
    head = nn.Linear(200, 1).to(device)
    opt = torch.optim.AdamW([{"params": tune, "lr": 1e-4}, {"params": head.parameters(), "lr": 1e-3}],
                            weight_decay=0.05)
    scaler = torch.amp.GradScaler()
    w = d["winz"]

    def logits(idx):
        x = resample_poly(w[idx].astype(np.float32), 25, 16, axis=-1)[..., :2000]  # 128 -> 200 Hz
        x = torch.tensor(x, dtype=torch.float32, device=device)
        with torch.autocast("cuda", dtype=torch.float16):
            f = m(x, return_features=True, ch_names=names)["features"].mean(1)
        return head(f.float()).squeeze(1)

    tr = np.concatenate([d["rows"][s] for s in tr_s])
    pw = pos_weight(d["y"][tr], device)
    if BALANCE:
        bw = group_w(d["wsite"][tr], d["y"][tr])
        pw = torch.tensor(1.0, device=device)
    for _ in range(epochs):
        m.train()
        order = rng.choice(tr, len(tr), p=bw / bw.sum()) if BALANCE else rng.permutation(tr)
        for k in range(0, len(order), bs):
            j = order[k:k + bs]
            loss = F.binary_cross_entropy_with_logits(
                logits(j), torch.tensor(d["y"][j], dtype=torch.float32, device=device), pos_weight=pw)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
    m.eval()
    with torch.no_grad():
        return np.array([float(torch.cat([logits(r[k:k + bs]) for k in range(0, len(r), bs)]).mean())
                         for r in (d["rows"][s] for s in te_s)])


def recording_z(d):
    """Per recording and channel z-score over all its windows, x0.2 to LaBraM's 0.1 mV scale."""
    w = np.empty_like(d["win"])
    for s, r in d["rows"].items():
        x = d["win"][r].astype(np.float32)
        mu = x.mean((0, 2), keepdims=True)
        sd = x.std((0, 2), keepdims=True) + 1e-6
        w[r] = ((x - mu) / sd * 0.2).astype(np.float16)
    return w


def lr_base(d, tr_s, te_s):
    iu = np.triu_indices(d["coh"].shape[1], 1)
    sidx = {s: i for i, s in enumerate(d["u"])}
    feat = lambda s: np.concatenate([d["ab"][d["rows"][s]].reshape(len(d["rows"][s]), -1).mean(0),  # noqa: E731
                                     d["coh"][d["rows"][s]][:, iu[0], iu[1], :].reshape(len(d["rows"][s]), -1).mean(0)])
    X = np.stack([feat(s) for s in d["u"]])
    for st in np.unique(d["site"]):  # per-site z-score, no labels
        mm = d["site"] == st
        X[mm] = (X[mm] - X[mm].mean(0)) / (X[mm].std(0) + 1e-8)
    a = [sidx[s] for s in tr_s]
    b = [sidx[s] for s in te_s]
    sw = group_w(d["site"][a], d["ys"][a]) if BALANCE else None
    return lr_probe().fit(X[a], d["ys"][a], logisticregression__sample_weight=sw).predict_proba(X[b])[:, 1]


# ------------------------------------------------------------------ driver


def summarise(d, score):
    res = {}
    for st in list(np.unique(d["site"])) + ["all"]:
        m = np.ones(len(d["u"]), bool) if st == "all" else d["site"] == st
        lo, hi = D.bootstrap_ci(d["ys"][m], score[m], roc_auc_score)
        res[st] = {"auc": float(roc_auc_score(d["ys"][m], score[m])), "ci": [lo, hi],
                   "n_pos": int(d["ys"][m].sum()), "n_neg": int((1 - d["ys"][m]).sum())}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/eeg/raw2")
    ap.add_argument("--tune", default="results/per_disorder/tune.json")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--save", default=None)
    ap.add_argument("--loho", action="store_true", help="leave-one-hospital-out instead of pooled 5-fold")
    ap.add_argument("--balance", action="store_true", help="equal weight per site x label group in training")
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.loads(Path(args.tune).read_text())["best"]["config"]
    models = args.models.split(",")
    global BALANCE
    BALANCE = args.balance
    save = Path(args.save or ("results/pooled_deep_loho.json" if args.loho else
                              "results/pooled_deep_bal.json" if args.balance else "results/pooled_deep.json"))
    report = json.loads(save.read_text()) if save.exists() else {}
    state = None
    if "labram_ft" in models:
        import truststore
        truststore.inject_into_ssl()
        from braindecode.models import Labram
        state = {k: v.clone() for k, v in
                 Labram.from_pretrained("braindecode/labram-pretrained", n_times=2000).state_dict().items()}
    for task in args.tasks.split(","):
        todo = [m for m in models if f"{task}:{m}" not in report]  # resumable
        if not todo:
            continue
        d = load_task(out, task, raw="labram_ft" in todo)
        if d["win"] is not None:
            d["winz"] = recording_z(d)
            d["win"] = None
        fl = folds(d, args.loho)
        g = graphs(d, cfg) if any(m in todo for m in ("gatv2", "gt", "gt_mil", "gt_mil_dro")) else None
        print(f"[{task}] {len(d['u'])} subjects, sites " +
              ", ".join(f"{s} {int((d['site'] == s).sum())}" for s in np.unique(d["site"])), flush=True)
        for m in todo:
            score = np.zeros(len(d["u"]))
            for a, b in fl:
                tr_s, te_s = d["u"][a], d["u"][b]
                if m == "lr_base":
                    score[b] = lr_base(d, tr_s, te_s)
                elif m in ("gatv2", "gt"):
                    score[b] = window_gnn("gatv2" if m == "gatv2" else "transformer", cfg, g, d, tr_s, te_s, device)
                elif m in ("gt_mil", "gt_mil_dro"):
                    score[b] = mil(cfg, g, d, tr_s, te_s, device, dro=m == "gt_mil_dro")
                else:
                    score[b] = labram_ft(d, tr_s, te_s, device, state)
            report[f"{task}:{m}"] = {"per_site": summarise(d, score),
                                     "oof": {str(s): float(v) for s, v in zip(d["u"], score)}}
            tmp = save.with_suffix(".tmp")
            tmp.write_text(json.dumps(report, indent=1))
            os.replace(tmp, save)
            print(f"  {m:<11} " + "  ".join(f"{s} {r['auc']:.3f}" for s, r in report[f"{task}:{m}"]["per_site"].items()),
                  flush=True)


if __name__ == "__main__":
    main()
