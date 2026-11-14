"""Phase 6: which electrodes, connections and bands the GATv2 relies on, per disorder.

Attribution = |gradient x (input - baseline)| of the disorder logit margin, with the
baseline = the training fold's mean features (first-order integrated gradients). Plain
gradient x input ranks neighbouring electrodes first only because their coherence is
large; against a mean baseline, a connection scores by how a subject's deviation from
typical connectivity moves the prediction. Out of fold: each
of 5 subject-wise fold models explains only windows of subjects it never saw.
The tuned GATv2 reads coherence twice - as each node's coherence row ("profile"
node features) and as top-4 edge weights - so a connection's importance sums both.

Two checks, because saliency and attention maps are easily noise:
    stability   mean pairwise Spearman correlation of the 5 fold models' connection maps
    reference   Spearman correlation, over (connection, band), of the attribution with a
                model-free map: |Cohen's d| of subject-mean coherence, patients vs controls

    python -m eeg.experiments.explain --out ~/eeg/raw2 --tasks alzheimers:ds004504 mood:mumtaz ...
Writes results/explain/<task>_<dataset>.json and .png.
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

from eeg.data import park as P  # noqa: E402
from eeg.experiments import per_disorder as D  # noqa: E402
from eeg.data import raw_eeg as R  # noqa: E402

NB, NN = P.N_BANDS, P.N_NODES
IU = np.triu_indices(NN, 1)


def attributions(model, cfg, ab, coh, idx, device, base_ab, base_coh):
    """Summed over idx: power (19, 6) and connection (19, 19, 6) |grad x (input - baseline)|."""
    bx = torch.tensor(P.build_node_features(base_ab, base_coh, cfg["node_features"]), dtype=torch.float32,
                      device=device)
    bc = torch.tensor(base_coh, dtype=torch.float32, device=device)
    loader = DataLoader(D._dataset(cfg, ab, coh, np.zeros(len(ab), dtype=np.int64), idx),
                        batch_size=256, shuffle=False)
    model.eval()
    power, conn = np.zeros((NN, NB)), np.zeros((NN, NN, NB))
    for b in loader:
        b = b.to(device)
        x = b.x.clone().requires_grad_(True)
        e = b.edge_attr.clone().requires_grad_(True)
        z = model(x, b.edge_index, e, b.batch)
        (z[:, 1] - z[:, 0]).sum().backward()
        dx = x - bx.repeat(b.num_graphs, 1)
        ax = (dx * x.grad).abs().detach().cpu().numpy().reshape(-1, NN, 6 + NN * NB)
        power += ax[:, :, :NB].sum(0)
        conn += ax[:, :, NB:].reshape(-1, NN, NN, NB).sum(0)
        s_, d_ = b.edge_index % NN
        ae = ((e - bc[s_, d_]) * e.grad).abs().detach().cpu().numpy()
        src, dst = s_.cpu().numpy(), d_.cpu().numpy()
        np.add.at(conn, (src, dst), ae)
    return power, conn


def sym(c):
    """(19, 19, ...) -> connection i-j importance, both directions, off-diagonal."""
    s = c + c.swapaxes(0, 1)
    s[np.arange(NN), np.arange(NN)] = 0
    return s


def cohen_d_map(coh, subject, y):
    u = np.unique(subject)
    m = np.stack([coh[subject == s].mean(0) for s in u])  # subject mean (19, 19, 6)
    lab = np.array([y[subject == s][0] for s in u])
    a, b = m[lab == 1], m[lab == 0]
    pooled = np.sqrt((a.var(0, ddof=1) + b.var(0, ddof=1)) / 2) + 1e-9
    return (a.mean(0) - b.mean(0)) / pooled  # signed: + = more coherent in patients


def head_plot(conn_map, node_imp, signed_d, title, path, top=15):
    import mne
    pos = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    xy = np.array([pos[c.capitalize() if c.startswith("FP") else c][:2] for c in P.CHANNELS])
    xy = (xy - xy.mean(0)) / np.linalg.norm(xy - xy.mean(0), axis=1).max() * 0.085  # fit inside the head
    fig, ax = plt.subplots(figsize=(5.2, 5.6))
    ax.add_patch(plt.Circle((0, 0), 0.095, fill=False, lw=1.5, color="0.4"))
    ax.plot([-0.012, 0, 0.012], [0.094, 0.108, 0.094], color="0.4", lw=1.5)  # nose
    vals = conn_map[IU]
    order = np.argsort(-vals)[:top]
    vmax = vals[order[0]]
    for k in order[::-1]:
        i, j = IU[0][k], IU[1][k]
        col = "#c0392b" if signed_d[i, j] > 0 else "#2471a3"
        ax.plot(*xy[[i, j]].T, color=col, lw=1 + 5 * vals[k] / vmax, alpha=0.75, solid_capstyle="round")
    size = 80 + 600 * node_imp / node_imp.max()
    ax.scatter(*xy.T, s=size, c="white", edgecolors="0.2", zorder=3)
    for (px, py), c in zip(xy, P.CHANNELS):
        ax.text(px, py, c.capitalize() if c.startswith("FP") else c, ha="center", va="center", fontsize=7, zorder=4)
    ax.set_title(title, fontsize=10)
    ax.text(0, -0.13, f"top {top} connections; width = importance\nred = more coherent in patients, "
            "blue = less; node size = band-power importance", ha="center", fontsize=7, color="0.3")
    ax.set_xlim(-0.12, 0.12)
    ax.set_ylim(-0.145, 0.115)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def explain(task, dataset, out, cfg, device, save):
    pos, neg, _ = R.TASKS[task]
    ab, coh, subject, groups, ds, _ = R.load_features(out, [dataset])
    keep, y = R.task_rows(groups, pos, neg)
    ab, coh, subject = ab[keep], coh[keep], subject[keep]
    rows = np.arange(len(y))
    power, conn, fold_maps = np.zeros((NN, NB)), np.zeros((NN, NN, NB)), []
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=D.SEED).split(rows, y, subject):
        model = D.fit(cfg, ab, coh, y, a, device)
        p, c = attributions(model, cfg, ab, coh, b, device, ab[a].mean(0), coh[a].mean(0))
        power += p
        conn += c
        fold_maps.append(sym(c.sum(-1))[IU])
    cmap = sym(conn.sum(-1))
    d_band = cohen_d_map(coh, subject, y)
    d = d_band.mean(-1)
    stab = float(np.mean([spearmanr(f, g)[0] for f, g in combinations(fold_maps, 2)]))
    ref = float(spearmanr(sym(conn)[IU].ravel(), np.abs(d_band[IU]).ravel())[0])
    band_conn = sym(conn).sum((0, 1)) / 2
    band = {"power": dict(zip(P.BANDS, (power.sum(0) / power.sum()).round(4).tolist())),
            "connectivity": dict(zip(P.BANDS, (band_conn / band_conn.sum()).round(4).tolist()))}
    order = np.argsort(-cmap[IU])[:10]
    name = lambda c: c.capitalize() if c.startswith("FP") else c  # noqa: E731
    top = [{"edge": f"{name(P.CHANNELS[IU[0][k]])}-{name(P.CHANNELS[IU[1][k]])}",
            "importance": round(float(cmap[IU][k] / cmap[IU].max()), 3),
            "cohen_d": round(float(d[IU][k]), 3)} for k in order]
    node = power.sum(1)
    share_power = float(power.sum() / (power.sum() + conn.sum()))
    r = {"task": task, "dataset": dataset, "n_subjects": int(len(np.unique(subject))),
         "fold_stability_spearman": round(stab, 3), "agreement_with_cohen_d_spearman": round(ref, 3),
         "share_band_power_vs_connectivity": round(share_power, 3), "bands": band,
         "top_nodes_power": [name(P.CHANNELS[i]) for i in np.argsort(-node)[:5]], "top_connections": top}
    stem = save / f"{task}_{dataset}"
    stem.with_suffix(".json").write_text(json.dumps(r, indent=2))
    head_plot(cmap, node, d, f"{task.replace('_', ' ')} vs {neg} - {dataset}", stem.with_suffix(".png"))
    print(f"{task:<18} {dataset:<9} stability {stab:.2f}  agree-with-d {ref:.2f}  "
          f"power share {share_power:.2f}  top: {', '.join(t['edge'] for t in top[:4])}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/eeg/raw2")
    ap.add_argument("--tune", default="results/per_disorder/tune.json")
    ap.add_argument("--tasks", nargs="+", default=["alzheimers:ds004504", "ftd:ds004504",
                    "mood:mumtaz", "mood:ds003478", "schizophrenia:aseeg", "parkinsons:ds004584"])
    ap.add_argument("--save", default="results/explain")
    args = ap.parse_args()
    out, save = Path(args.out).expanduser(), Path(args.save)
    save.mkdir(parents=True, exist_ok=True)
    cfg = {**json.loads(Path(args.tune).read_text())["best"]["config"], "epochs": R.RAW_EPOCHS}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for t in args.tasks:
        task, dataset = t.split(":")
        explain(task, dataset, out, cfg, device, save)


if __name__ == "__main__":
    main()
