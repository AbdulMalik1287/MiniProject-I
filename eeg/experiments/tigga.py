"""Re-run of Tigga, Kumari & Alnajjar (Sci Rep 16:34662, 2026) under their protocol and an honest one.

Their released code (github.com/NehaPrerna123/Alzhiemer-s-prediction-using-EEG,
dataset*_hybridgatv2lstmmodel.py) is copied here in model and training settings:
    rows = single time points, 19 channel values each; StandardScaler fit on ALL rows;
    Gaussian noise 0.01; edge_index = complete 19x19 graph minus self-loops, 20% edges dropped;
    HybridGATv2LSTMModel(hidden 64 x 4 heads, LSTM 2x128), full-batch Adam lr 0.005, 100 epochs;
    train_test_split(test_size=0.2, random_state=42) over rows.

What the code actually does (verified by reading it): PyG treats every ROW as a node,
so the 19x19 edge list connects only rows 0-18; every other sample has just its
self-loop and GATv2 acts as a per-row linear map. The LSTM runs across all rows of a
split as one sequence. We do not fix this - we reproduce it, then test it:

    leaky     their protocol: random row split, scaler fit before the split
    honest    rows of held-out SUBJECTS only (5-fold StratifiedGroupKFold), scaler on train
    each with and without the edge list, to show the "graph" contributes nothing.

Data: ds004504 (AD 36 / FTD 23 / control 29), our harmonised 128 Hz windows
(eeg/data/raw_eeg.py features cache, `win`), ROWS_PER_SUBJECT time points drawn per subject.
Their cGAN augmentation is not re-run: their own paper puts synthetic rows in the test set.

    python -m eeg.experiments.tigga --cache ~/eeg/raw2/features/ds004504.npz
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import dense_to_sparse

LABELS = {"control": 0, "ftd": 1, "alzheimers": 2}
ROWS_PER_SUBJECT = 500


class HybridGATv2LSTMModel(nn.Module):  # verbatim structure from their dataset1/2 scripts
    def __init__(self, input_dim, hidden_dim, lstm_hidden_dim, output_dim, num_heads=4):
        super().__init__()
        self.gatv2 = GATv2Conv(input_dim, hidden_dim, heads=num_heads, concat=True)
        self.norm1 = nn.LayerNorm(hidden_dim * num_heads)
        self.multi_lstm = nn.LSTM(hidden_dim * num_heads, lstm_hidden_dim, num_layers=2,
                                  batch_first=True, dropout=0.3)
        self.attention_fusion = nn.Linear(hidden_dim * num_heads + lstm_hidden_dim, hidden_dim * num_heads)
        self.final_fc = nn.Linear(hidden_dim * num_heads, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index):
        g = torch.relu(self.norm1(self.gatv2(x, edge_index)))
        lstm_out, _ = self.multi_lstm(g.unsqueeze(0))
        fused = self.attention_fusion(torch.cat((g, lstm_out.squeeze(0)), dim=1))
        return self.final_fc(self.dropout(torch.relu(fused + g)))


def their_edges(n_nodes, drop=0.2):
    ei, _ = dense_to_sparse(torch.ones(n_nodes, n_nodes) - torch.eye(n_nodes))
    return ei[:, torch.rand(ei.size(1)) > drop]


def train_eval(xtr, ytr, xte, edges, device, epochs=100):
    model = HybridGATv2LSTMModel(xtr.shape[1], 64, 128, 3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    xt, yt = torch.tensor(xtr, dtype=torch.float32, device=device), torch.tensor(ytr, device=device)
    ei = edges.to(device)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        nn.functional.cross_entropy(model(xt, ei), yt).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(xte, dtype=torch.float32, device=device), ei).argmax(1).cpu().numpy()


def rows_from_cache(path, rng):
    z = np.load(path)
    win, subj, grp = z["win"], z["subject"].astype(str), z["group"].astype(str)
    xs, ys, ss = [], [], []
    for s in np.unique(subj):
        w = win[subj == s].astype(np.float32).transpose(0, 2, 1).reshape(-1, win.shape[1])  # (t, 19)
        pick = rng.choice(len(w), size=min(ROWS_PER_SUBJECT, len(w)), replace=False)
        xs.append(w[pick])
        ys += [LABELS[grp[subj == s][0]]] * len(pick)
        ss += [s] * len(pick)
    return np.concatenate(xs), np.array(ys), np.array(ss)


def subject_vote(pred, y, subj):
    """Subject-level label = majority vote of its rows."""
    u = np.unique(subj)
    p = np.array([np.bincount(pred[subj == s], minlength=3).argmax() for s in u])
    t = np.array([y[subj == s][0] for s in u])
    return float((p == t).mean()), float(balanced_accuracy_score(t, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="eeg/data/raw_eeg.py features cache for ds004504 (with win)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="results/tigga.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = {"rows_per_subject": ROWS_PER_SUBJECT, "runs": []}
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        x, y, subj = rows_from_cache(Path(args.cache).expanduser(), rng)
        for edges_on in (True, False):
            edges = their_edges(x.shape[1]) if edges_on else torch.empty(2, 0, dtype=torch.long)
            # leaky: exactly their order of operations
            xl = StandardScaler().fit_transform(x) + 0.01 * rng.standard_normal(x.shape)
            tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42)
            pred = train_eval(xl[tr], y[tr], xl[te], edges, device)
            shared = len(set(subj[tr]) & set(subj[te]))
            r = {"seed": seed, "edges": edges_on, "protocol": "leaky",
                 "row_acc": float((pred == y[te]).mean()),
                 "row_bal_acc": float(balanced_accuracy_score(y[te], pred)),
                 "test_subjects_also_in_train": shared}
            report["runs"].append(r)
            print(json.dumps(r), flush=True)
            # honest: subjects never straddle the split, scaler fit on train rows only
            preds, truth, sids = [], [], []
            for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(x, y, subj):
                assert not set(subj[a]) & set(subj[b])
                sc = StandardScaler().fit(x[a])
                xa = sc.transform(x[a]) + 0.01 * rng.standard_normal(x[a].shape)
                pa = rng.permutation(len(a))  # rows arrive shuffled, as after train_test_split
                pb = rng.permutation(len(b))
                preds.append(train_eval(xa[pa], y[a][pa], sc.transform(x[b])[pb], edges, device))
                truth.append(y[b][pb])
                sids.append(subj[b][pb])
            pred, yt, st = np.concatenate(preds), np.concatenate(truth), np.concatenate(sids)
            sacc, sbal = subject_vote(pred, yt, st)
            r = {"seed": seed, "edges": edges_on, "protocol": "honest",
                 "row_acc": float((pred == yt).mean()),
                 "row_bal_acc": float(balanced_accuracy_score(yt, pred)),
                 "subject_acc": sacc, "subject_bal_acc": sbal}
            report["runs"].append(r)
            print(json.dumps(r), flush=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print("chance (3 classes, balanced accuracy) = 0.333")


if __name__ == "__main__":
    main()
