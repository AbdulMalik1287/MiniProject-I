"""Preprocessing v2, from the literature review (Gemein 2020; Wang 2023 on ds004504; Klepl 2022).

Changes against raw_eeg.py, decided before running:
    windows     drop the first 60 s when >= 30 windows remain after it (settling/eye-opening
                transients, Gemein 2020); keep up to 60 clean 10 s windows spread evenly over
                the rest of the recording instead of the first 30; subject features are the
                MEDIAN over windows (robust to residual artefact windows)
    periodic    aperiodic-adjusted band power: log10 PSD minus the 1/f line fitted on 2-40 Hz
                (alpha 7-14 Hz excluded, as rich_features.py), averaged per band; delta from 2 Hz
    wpli        weighted phase lag index per band (Vinck 2011): |sum Im S| / sum |Im S|
                over 2 s segments and band bins - insensitive to zero-lag volume conduction
Unchanged: channels, notch, 128 Hz, 1-45 Hz FIR, CAR, rejection thresholds.

Evaluation: the master-model protocol of pooled_deep.py (all sites of a disorder, one
StratifiedKFold(5, seed 42) stratified by site x label, per-site z-scoring without labels,
AUC per site). Logistic regression C=0.05 and filter-bank Riemannian (per-site re-centred).

    python prep_v2.py extract --out ~/eeg/raw2
    python prep_v2.py eval --out ~/eeg/raw2
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import per_disorder as D
import raw_eeg as R
import riemann as RM
from baselines import lr_probe
from pooled import recentre

SKIP_WIN = 6        # 60 s of 10 s windows
MAX_WIN = 60
DATASETS = ["ds004504", "brainlat", "mumtaz", "ds003478", "aseeg", "warsaw", "semmelweis"]
TASKS = ["alzheimers", "ftd", "alzheimers_vs_ftd", "mood", "schizophrenia"]
BANDS = [(2, 4), (4, 8), (8, 12), (12, 25), (25, 30), (30, 45)]


# ------------------------------------------------------------------ extraction


def select_windows(segments):
    """[(19, n) harmonised at 128 Hz] -> (w, 19, 1280) per the v2 rule."""
    n = R.WIN_S * R.SF
    wins = [s[:, :s.shape[1] // n * n].reshape(s.shape[0], -1, n).transpose(1, 0, 2) for s in segments]
    w = np.concatenate([x for x in wins if len(x)]) if any(len(x) for x in wins) else np.empty((0, 19, n))
    if len(w) - SKIP_WIN >= 30:
        w = w[SKIP_WIN:]
    bad = (np.ptp(w, axis=2) > R.PTP_UV).any(1) | (w.std(axis=2) < R.FLAT_UV).any(1)
    w = w[~bad]
    if len(w) > MAX_WIN:
        w = w[np.round(np.linspace(0, len(w) - 1, MAX_WIN)).astype(int)]
    return w


def v2_features(w):
    """(w, 19, T) -> periodic (w, 19, 6), wpli (w, 19, 19, 6)."""
    sf, nper = R.SF, 2 * R.SF
    starts = np.arange(0, w.shape[2] - nper + 1, sf)
    X = np.fft.rfft(np.stack([w[:, :, s:s + nper] for s in starts], 2) * np.hanning(nper), axis=-1)  # (w,c,s,f)
    f = np.fft.rfftfreq(nper, 1 / sf)
    psd = (np.abs(X) ** 2).mean(2)
    fit = (f >= 2) & (f <= 40) & ~((f >= 7) & (f <= 14))
    lf, lp = np.log10(f[fit]), np.log10(psd[..., fit] + 1e-30)
    slope = ((lf - lf.mean()) * (lp - lp.mean(-1, keepdims=True))).sum(-1) / ((lf - lf.mean()) ** 2).sum()
    off = lp.mean(-1) - slope * lf.mean()
    per = np.zeros(psd.shape[:2] + (len(BANDS),), np.float32)
    wpli = np.zeros((w.shape[0], w.shape[1], w.shape[1], len(BANDS)), np.float32)
    for b, (lo, hi) in enumerate(BANDS):
        m = (f >= lo) & (f < hi)
        resid = np.log10(psd[..., m] + 1e-30) - (off[..., None] + slope[..., None] * np.log10(f[m]))
        per[..., b] = resid.mean(-1)
        im = np.einsum("wisf,wjsf->wijsf", X[..., m], np.conj(X[..., m])).imag
        wpli[..., b] = np.abs(im.sum((-1, -2))) / (np.abs(im).sum((-1, -2)) + 1e-30)
    return per, wpli


def cmd_extract(args):
    out = args.out / "features_v2"
    out.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        if (out / f"{ds}.npz").exists():
            continue
        keep = {k: [] for k in ("ab", "coh", "per", "wpli", "win", "subject", "group")}
        for sid, group, x, sf, filt in R.recordings(ds):
            segs = x if isinstance(x, list) else [(x, sf)]
            w = select_windows([R.harmonise(s, f_, filt, R.DATASETS[ds]["line"])
                                for s, f_ in segs if s.shape[1] >= f_ * R.WIN_S])
            if len(w) == 0:
                continue
            a, c = R.spectral_features(w)
            p, q = v2_features(w)
            for k, v in (("ab", a), ("coh", c), ("per", p), ("wpli", q), ("win", w.astype(np.float16))):
                keep[k].append(v)
            keep["subject"] += [sid] * len(w)
            keep["group"] += [group] * len(w)
        arrs = {k: (np.concatenate(v) if k not in ("subject", "group") else np.array(v)) for k, v in keep.items()}
        tmp = out / f"{ds}.tmp.npz"
        np.savez_compressed(tmp, **arrs)
        os.replace(tmp, out / f"{ds}.npz")
        n = np.unique(arrs["subject"], return_counts=True)[1]
        print(f"[v2] {ds:<10} {len(n)} subjects, windows/subject median {np.median(n):.0f} (min {n.min()}, max {n.max()})",
              flush=True)


# ------------------------------------------------------------------ evaluation


def subject_table(feat_dir, task, agg):
    pos, neg, sites = R.TASKS[task]
    parts = [np.load(feat_dir / f"{ds}.npz") for ds in sites]
    has_v2 = "per" in parts[0]
    cat = lambda k: np.concatenate([p[k] for p in parts])  # noqa: E731
    groups = np.concatenate([R.cache_groups(ds, p) for ds, p in zip(sites, parts)])
    site_w = np.concatenate([[ds] * len(p["subject"]) for ds, p in zip(sites, parts)])
    rows, y = R.task_rows(groups, pos, neg)
    subject = cat("subject")[rows]
    iu = np.triu_indices(19, 1)
    blocks = {"ab": cat("ab")[rows].reshape(len(rows), -1),
              "coh": cat("coh")[rows][:, iu[0], iu[1], :].reshape(len(rows), -1)}
    if has_v2:
        blocks["per"] = cat("per")[rows].reshape(len(rows), -1)
        blocks["wpli"] = cat("wpli")[rows][:, iu[0], iu[1], :].reshape(len(rows), -1)
    win = cat("win")[rows]
    u = np.unique(subject)
    idx = {s: np.where(subject == s)[0] for s in u}
    f = np.median if agg == "median" else np.mean
    tab = {k: np.stack([f(v[idx[s]], 0) for s in u]) for k, v in blocks.items()}
    ys = np.array([y[idx[s][0]] for s in u])
    site = np.array([site_w[rows][idx[s][0]] for s in u])
    cu, covs = RM.subject_covs(win, subject)
    assert (cu == u).all()
    return u, ys, site, tab, covs


def per_site(x, site, fn):
    x = x.copy()
    for st in np.unique(site):
        x[site == st] = fn(x[site == st])
    return x


def cmd_eval(args):
    from sklearn.model_selection import StratifiedKFold
    save = Path(args.save)
    report = json.loads(save.read_text()) if save.exists() else {}
    z = lambda a: (a - a.mean(0)) / (a.std(0) + 1e-8)  # noqa: E731
    variants = {  # name -> (feature dir, aggregation, LR blocks)
        "v1_lr_base": ("features", "mean", ["ab", "coh"]),
        "v1_riemann": ("features", "mean", None),
        "v2_lr_base": ("features_v2", "median", ["ab", "coh"]),
        "v2_lr_per": ("features_v2", "median", ["ab", "coh", "per"]),
        "v2_lr_wpli": ("features_v2", "median", ["ab", "wpli"]),
        "v2_lr_all": ("features_v2", "median", ["ab", "coh", "per", "wpli"]),
        "v2_riemann": ("features_v2", "median", None),
        # trace-normalised covariances: overall signal scale removed, so the BrainLat
        # patient/control amplitude gap (rs-HEP vs rs files) cannot drive the score
        "v1_riemann_tn": ("features", "mean", "tn"),
        "v2_riemann_tn": ("features_v2", "median", "tn"),
    }
    for task in TASKS:
        cache = {}
        for name, (fd, agg, blocks) in variants.items():
            key = f"{task}:{name}"
            if key in report:
                continue
            if (fd, agg) not in cache:
                cache[(fd, agg)] = subject_table(args.out / fd, task, agg)
            u, ys, site, tab, covs = cache[(fd, agg)]
            if blocks == "tn":
                covs = covs / np.trace(covs, axis1=-2, axis2=-1)[..., None, None]
            x = (recentre_sites(covs, site) if blocks in (None, "tn")
                 else per_site(np.concatenate([tab[b] for b in blocks], 1), site, z))
            strat = np.char.add(site.astype(str), ys.astype(str))
            score = np.zeros(len(u))
            for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(u, strat):
                clf = RM.FBTangentLR() if blocks in (None, "tn") else lr_probe()
                score[b] = clf.fit(x[a], ys[a]).predict_proba(x[b])[:, 1]
            res = {}
            for st in list(np.unique(site)) + ["all"]:
                m = np.ones(len(u), bool) if st == "all" else site == st
                lo, hi = D.bootstrap_ci(ys[m], score[m], roc_auc_score)
                res[st] = {"auc": float(roc_auc_score(ys[m], score[m])), "ci": [lo, hi]}
            report[key] = {"per_site": res, "n": int(len(u))}
            tmp = save.with_suffix(".tmp")
            tmp.write_text(json.dumps(report, indent=1))
            os.replace(tmp, save)
            print(f"{task:<18} {name:<11} n={len(u):<4}" + "  ".join(f"{s} {r['auc']:.3f}" for s, r in res.items()),
                  flush=True)


def recentre_sites(covs, site):
    out = covs.copy()
    for st in np.unique(site):
        out[site == st] = recentre(covs[site == st])
    return out


def selfcheck():
    rng = np.random.default_rng(0)
    t = np.arange(R.WIN_S * R.SF) / R.SF
    src = rng.normal(size=(3, len(t)))
    x = np.stack([src[i % 3] + 0.3 * rng.normal(size=len(t)) for i in range(19)])  # zero-lag mixing
    x[1] = np.roll(x[0], 3)  # channel 1 = lagged copy of channel 0
    w = np.stack([x, x])
    per, wpli = v2_features(w)
    assert wpli[0, 0, 3].mean() < 0.3, wpli[0, 0, 3]   # zero-lag (volume conduction) -> low wPLI
    assert wpli[0, 0, 1].mean() > 0.8, wpli[0, 0, 1]   # true lag -> high wPLI
    alpha = 5 * np.sin(2 * np.pi * 10 * t)
    p2, _ = v2_features(np.stack([x + alpha, x + alpha]))
    assert (p2[0, :, 2] > per[0, :, 2] + 0.5).all()     # alpha peak shows up in periodic alpha
    segs = [np.zeros((19, 100 * R.WIN_S * R.SF)) + rng.normal(size=(19, 100 * R.WIN_S * R.SF))]
    assert len(select_windows(segs)) == MAX_WIN
    assert len(select_windows([segs[0][:, :33 * 1280]])) == 33   # too short to drop 60 s
    print("prep_v2 selfcheck ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "eval", "selfcheck"])
    ap.add_argument("--out", default="~/eeg/raw2")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--save", default="results/prep_v2.json")
    args = ap.parse_args()
    args.out = Path(args.out).expanduser()
    {"extract": cmd_extract, "eval": cmd_eval, "selfcheck": lambda a: selfcheck()}[args.cmd](args)


if __name__ == "__main__":
    main()
