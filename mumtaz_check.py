"""Mumtaz depression: is the score driven by a group-wise recording difference above 30 Hz?

Found by explain_final.py: 1/f-corrected gamma power separates MDD from controls with
Cohen's d ~ -3, implausibly large. Checks (subject-wise, master-model folds):
    psd        group-mean log10 PSD at 10/25/35/44 Hz (controls >> MDD at 44 Hz = a low-pass
               filter applied to one group only)
    gamma_only AUC of logistic regression on gamma relative + periodic power alone
    band_1_25  AUC using relative band power computed ONLY within 1-25 Hz (six bands, total
               power 1-25 Hz) - immune to anything above 25 Hz

    python mumtaz_check.py --out ~/eeg/raw2
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import pooled_deep as Q
from baselines import lr_probe

F = np.fft.rfftfreq(256, 1 / 128)


def psd(w):
    seg = np.stack([w[..., k:k + 256] for k in range(0, w.shape[-1] - 255, 128)], -2)
    return (np.abs(np.fft.rfft(seg * np.hanning(256), axis=-1)) ** 2).mean(-2)  # (w, 19, f)


def cv_auc(X, ys, u):
    s = np.zeros(len(u))
    for a, b in Q.folds({"u": u, "ys": ys, "site": np.array(["mumtaz"] * len(u))}):
        s[b] = lr_probe().fit(X[a], ys[a]).predict_proba(X[b])[:, 1]
    return float(roc_auc_score(ys, s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/eeg/raw2")
    args = ap.parse_args()
    z = np.load(Path(args.out).expanduser() / "features_v2" / "mumtaz.npz")
    s, g = z["subject"], z["group"]
    u = np.unique(s)
    ys = np.array([int(g[s == x][0] == "mood") for x in u])
    bands = [(1, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 25)]
    low, gam, spec = [], [], {0: [], 1: []}
    for x, y in zip(u, ys):
        p = psd(z["win"][s == x].astype(np.float64))
        tot = p[..., (F >= 1) & (F < 25)].sum(-1)
        low.append(np.median(np.stack([np.log(p[..., (F >= lo) & (F < hi)].sum(-1) / tot) for lo, hi in bands], -1), 0).ravel())
        gam.append(np.r_[np.median(z["ab"][s == x][..., 5], 0), np.median(z["per"][s == x][..., 5], 0)])
        spec[y].append(p.mean((0, 1)))
    res = {"psd_log10": {("mdd" if y else "control"): {f"{hz}Hz": float(np.log10(np.mean(spec[y], 0)[np.argmin(abs(F - hz))]))
                                                        for hz in (10, 25, 35, 44)} for y in (0, 1)},
           "auc_gamma_only": cv_auc(np.array(gam), ys, u),
           "auc_relative_power_1_25Hz": cv_auc(np.array(low), ys, u)}
    Path("results/mumtaz_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
