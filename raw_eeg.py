#!/usr/bin/env python3
"""Per-disease GNNs on raw resting-state EEG pooled across hospitals.

Park et al. released only precomputed features, so external raw recordings
cannot be appended to it. This track builds identical features for every raw
dataset with one pipeline, and pools datasets per disease under one rule:
**every dataset contributes its own healthy controls.** Site effects in EEG
(amplifier, reference, filters) are larger than disease effects, so patients
from one hospital with controls from another would teach the model the site.

    python raw_eeg.py features               # load, harmonise, window, cache features
    python raw_eeg.py run --disease schizophrenia
    python raw_eeg.py run --disease mood

Datasets (all CC-BY, eyes-closed resting state, 19 channels, 10-20):
    aseeg   Bagherzadeh et al. 2026, Zenodo 18029536    51 SZ  + 50 controls
    warsaw  Olejarczyk & Jernajczyk 2017, RepOD 0107441 14 SZ  + 14 controls
    mumtaz  Mumtaz et al. 2016, figshare 4244171        30 MDD + 28 controls
            (6 of 64 eyes-closed files are dead links on figshare)

Pipeline, identical for every source:
    notch 50 Hz at native rate (raw sources) -> resample 128 Hz -> 1-45 Hz FIR
    -> common average reference -> 10 s windows -> reject windows with any
    channel peak-to-peak > 300 uV or flat (std < 0.2 uV) -> first 30 clean
    windows per subject (equal 5 min each, so long recordings do not dominate)
Features per window, in park.py's layout so the same graph code applies:
    node: log relative band power (19, 6) - relative, so amplifier gain cancels
    edge: magnitude-squared coherence per band (19, 19, 6) - scale-invariant

ASEEG files carry no channel labels; infer_channel_order.py establishes that
they follow the 10-20 row order used here.

Evaluation is always subject-level (mean window probability per subject), and
windows of one subject never straddle a train/test boundary. The GNN config is
the one tuned on Park (per_disorder.py), so nothing here is selected on these
datasets. Schemes: within each dataset (5-fold, grouped by subject); pooled
across datasets; and cross-hospital transfer (train on one, test on the other),
which directly tests whether a disease signature carries across sites.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import park as P
import per_disorder as D

SF = 128
WIN_S = 10
MAX_WINDOWS = 30
PTP_UV, FLAT_UV = 300.0, 0.2
BAND_EDGES = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 25),
              "highbeta": (25, 30), "gamma": (30, 45)}
CH = [c.upper() for c in P.CHANNELS]
ALIASES = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
DATASET_DISEASE = {"aseeg": "schizophrenia", "warsaw": "schizophrenia", "mumtaz": "mood"}
ROOT = Path(__file__).resolve().parents[3] / "data" / "external"
if not ROOT.exists():  # running from the main checkout rather than a worktree
    ROOT = Path(__file__).resolve().parent / "data" / "external"
RAW_EPOCHS = 20  # ~60k graph visits per fit, comparable to Park's 150 epochs on ~230 graphs


# ------------------------------------------------------------------- loading


def norm_name(ch):
    base = ch.replace("EEG ", "").strip().split("-")[0].upper()
    return ALIASES.get(base, base)


def load_edf(path):
    import mne
    mne.set_log_level("ERROR")
    raw = mne.io.read_raw_edf(path, preload=True)
    names = {}
    for c in raw.ch_names:
        names.setdefault(norm_name(c), c)
    missing = [c for c in CH if c not in names]
    if missing:
        raise ValueError(f"{path.name}: missing channels {missing}")
    raw.pick([names[c] for c in CH])
    return raw.get_data() * 1e6, raw.info["sfreq"]  # volts -> microvolts


def recordings(dataset):
    """Yield (subject_id, label, data uV (19, n), sfreq, already_filtered)."""
    d = ROOT / dataset
    if dataset == "aseeg":
        import h5py
        for group, label in (("normal", 0), ("sz", 1)):
            for fp in sorted((d / "EC" / group).glob("*.mat")):
                with h5py.File(fp, "r") as h:
                    x = np.array(h["preprocessed_data"], dtype=np.float64).T
                yield f"aseeg-{fp.stem}", label, x, 128.0, True
    elif dataset == "mumtaz":
        for fp in sorted(d.glob("*.edf")):
            cls, sid = fp.name.split()[:2]
            # Subject numbers restart per class ("H S1" and "MDD S1" are
            # different people), so the key must include the class.
            yield f"mumtaz-{cls}-{sid}", int(cls == "MDD"), *load_edf(fp), False
    elif dataset == "warsaw":
        for fp in sorted(d.glob("*.edf")):
            yield f"warsaw-{fp.stem}", int(fp.stem.startswith("s")), *load_edf(fp), False
    else:
        raise ValueError(dataset)


def harmonise(x, sf, already_filtered):
    import mne
    mne.set_log_level("ERROR")
    if not already_filtered:
        x = mne.filter.notch_filter(x, sf, 50.0)
    if sf != SF:
        x = mne.filter.resample(x, up=SF, down=sf)
    x = mne.filter.filter_data(x, SF, 1.0, 45.0)
    return x - x.mean(0, keepdims=True)


def clean_windows(x):
    """(19, n) -> (w, 19, 1280) clean windows, plus how many were rejected."""
    n = WIN_S * SF
    k = x.shape[1] // n
    w = x[:, :k * n].reshape(x.shape[0], k, n).transpose(1, 0, 2)
    bad = (np.ptp(w, axis=2) > PTP_UV).any(1) | (w.std(axis=2) < FLAT_UV).any(1)
    return w[~bad][:MAX_WINDOWS], int(bad.sum()), int(k)


# ------------------------------------------------------------------ features


def spectral_features(w):
    """(w, 19, T) -> log relative band power (w, 19, 6), band coherence (w, 19, 19, 6).

    Welch-style: 2 s Hann segments, 50% overlap; coherence is |S_ij|^2 / (S_ii S_jj)
    averaged over the bins of each band. Fully vectorised - pairwise calls to
    scipy.signal.coherence would take ~1M calls for these datasets.
    """
    nper, step = 2 * SF, SF
    starts = np.arange(0, w.shape[2] - nper + 1, step)
    segs = np.stack([w[:, :, s:s + nper] for s in starts], axis=2)  # (w, c, s, nper)
    X = np.fft.rfft(segs * np.hanning(nper), axis=-1)
    freqs = np.fft.rfftfreq(nper, 1 / SF)
    psd = (np.abs(X) ** 2).mean(2)  # (w, c, f)
    csd = np.einsum("wisf,wjsf->wijf", X, np.conj(X)) / len(starts)
    coh_f = np.abs(csd) ** 2 / (psd[:, :, None, :] * psd[:, None, :, :] + 1e-30)

    total = psd[..., (freqs >= 1) & (freqs < 45)].sum(-1)
    ab = np.zeros((w.shape[0], w.shape[1], len(P.BANDS)), np.float32)
    coh = np.zeros((w.shape[0], w.shape[1], w.shape[1], len(P.BANDS)), np.float32)
    for b, name in enumerate(P.BANDS):
        lo, hi = BAND_EDGES[name]
        m = (freqs >= lo) & (freqs < hi)
        ab[..., b] = np.log(psd[..., m].sum(-1) / total + 1e-12)
        coh[..., b] = coh_f[..., m].mean(-1)
    idx = np.arange(w.shape[1])
    coh[:, idx, idx, :] = 1.0
    return ab, np.clip(coh, 0, 1)


def cmd_features(args):
    out = args.out / "features"
    out.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        abs_, cohs, subj, lab, qc = [], [], [], [], []
        for sid, label, x, sf, filt in recordings(ds):
            w, rejected, total = clean_windows(harmonise(x, sf, filt))
            qc.append({"subject": sid, "label": label, "windows_total": total,
                       "rejected": rejected, "kept": int(len(w))})
            if len(w) == 0:
                continue
            a, c = spectral_features(w)
            abs_.append(a)
            cohs.append(c)
            subj += [sid] * len(w)
            lab += [label] * len(w)
        np.savez_compressed(out / f"{ds}.npz", ab=np.concatenate(abs_), coh=np.concatenate(cohs),
                            subject=np.array(subj), label=np.array(lab))
        (out / f"{ds}_qc.json").write_text(json.dumps(qc, indent=1))
        for label, name in ((0, "controls"), (1, "patients")):
            q = [r for r in qc if r["label"] == label]
            rej = sum(r["rejected"] for r in q) / max(sum(r["windows_total"] for r in q), 1)
            print(f"[features] {ds:<7} {name:<9} {len(q):>3} subjects, "
                  f"{sum(r['kept'] for r in q):>5} windows kept, {rej:.1%} rejected", flush=True)
        print(f"[features] {ds}: {sum(r['kept'] == 0 for r in qc)} subjects with no clean window",
              flush=True)


# -------------------------------------------------------------- experiments


def load_features(out, datasets):
    parts = [np.load(out / "features" / f"{ds}.npz") for ds in datasets]
    cat = lambda k: np.concatenate([p[k] for p in parts])  # noqa: E731
    dsname = np.concatenate([[ds] * len(p["label"]) for ds, p in zip(datasets, parts)])
    return cat("ab"), cat("coh"), cat("subject"), cat("label").astype(np.int64), dsname


def subject_scores(subject, y, win_prob, rows):
    """Aggregate window probabilities to one score per subject."""
    s = subject[rows]
    uniq = np.unique(s)
    score = np.array([win_prob[s == u].mean() for u in uniq])
    lab = np.array([y[rows][s == u][0] for u in uniq])
    return uniq, lab, score


def auc_report(lab, score):
    auc = roc_auc_score(lab, score)
    lo, hi = D.bootstrap_ci(lab, score, roc_auc_score)
    return {"auc": float(auc), "ci": [lo, hi], "n_patients": int(lab.sum()),
            "n_controls": int(len(lab) - lab.sum())}


def grouped_oof(cfg, ab, coh, subject, y, rows, strat, device, folds=5):
    """Out-of-fold window probabilities with subjects kept whole within a fold."""
    prob = np.zeros(len(y))
    sgkf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=D.SEED)
    for a, b in sgkf.split(rows, strat[rows], subject[rows]):
        tr, te = rows[a], rows[b]
        assert not set(subject[tr]) & set(subject[te]), "subject leaked across folds"
        model = D.fit(cfg, ab, coh, y, tr, device)
        prob[te] = D.predict_proba(model, cfg, ab, coh, te, device)
    return prob


def fmt(r):
    return (f"AUC {r['auc']:.3f} [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]  "
            f"({r['n_patients']} patients / {r['n_controls']} controls)")


def cmd_run(args, device):
    datasets = [d for d, dis in DATASET_DISEASE.items() if dis == args.disease]
    datasets = [d for d in datasets if (args.out / "features" / f"{d}.npz").exists()]
    if not datasets:
        raise FileNotFoundError(f"no feature caches for {args.disease}; run `features` first")
    tune = Path(args.tune).expanduser()
    base = json.loads(tune.read_text())["best"]["config"] if tune.exists() else D.GRID[-1]
    cfg = {**base, "epochs": RAW_EPOCHS}
    print(f"[run] {args.disease}: datasets {datasets}; GNN {cfg['model']}/{cfg['pool']}/"
          f"topk={cfg['topk']} (tuned on Park, not on these data), {cfg['epochs']} epochs\n")
    ab, coh, subject, y, ds = load_features(args.out, datasets)
    report = {"disease": args.disease, "datasets": datasets, "config": cfg}

    for d in datasets:
        rows = np.where(ds == d)[0]
        prob = grouped_oof(cfg, ab, coh, subject, y, rows, y, device)
        _, lab, score = subject_scores(subject, y, prob, rows)
        report[f"within:{d}"] = auc_report(lab, score)
        print(f"within  {d:<16} {fmt(report[f'within:{d}'])}", flush=True)

    if len(datasets) > 1:
        rows = np.arange(len(y))
        strat = np.array([f"{a}-{b}" for a, b in zip(ds, y)])
        prob = grouped_oof(cfg, ab, coh, subject, y, rows, strat, device)
        _, lab, score = subject_scores(subject, y, prob, rows)
        report["pooled"] = auc_report(lab, score)
        print(f"pooled  {'+'.join(datasets):<16} {fmt(report['pooled'])}", flush=True)
        for d in datasets:
            r = np.where(ds == d)[0]
            _, lab, score = subject_scores(subject, y, prob, r)
            report[f"pooled:{d}"] = auc_report(lab, score)
            print(f"  pooled model scored on {d:<8} {fmt(report[f'pooled:{d}'])}", flush=True)

        for a in datasets:
            for b in datasets:
                if a == b:
                    continue
                tr, te = np.where(ds == a)[0], np.where(ds == b)[0]
                model = D.fit(cfg, ab, coh, y, tr, device)
                prob = np.zeros(len(y))
                prob[te] = D.predict_proba(model, cfg, ab, coh, te, device)
                _, lab, score = subject_scores(subject, y, prob, te)
                report[f"transfer:{a}->{b}"] = auc_report(lab, score)
                print(f"transfer train {a} -> test {b:<8} {fmt(report[f'transfer:{a}->{b}'])}",
                      flush=True)

    print("\nA CI spanning 0.5 means that result is not distinguishable from chance.")
    (args.out / f"raw_{args.disease}.json").write_text(json.dumps(report, indent=2))


# ---------------------------------------------------------------- self-check


def selfcheck():
    rng = np.random.default_rng(0)
    T = WIN_S * SF
    t = np.arange(T) / SF
    w = rng.normal(0, 1, (2, 19, T))
    w[:, 0] += 20 * np.sin(2 * np.pi * 10 * t)       # strong 10 Hz on channel 0
    w[:, 1] = w[:, 2] = rng.normal(0, 5, (2, T))     # channels 1 and 2 identical
    ab, coh = spectral_features(w)
    assert ab.shape == (2, 19, 6) and coh.shape == (2, 19, 19, 6)
    assert P.BANDS[int(np.argmax(ab[0, 0]))] == "alpha", "10 Hz must dominate alpha"
    assert (coh[:, 1, 2, :] > 0.99).all(), "identical channels must be fully coherent"
    assert coh[:, 3, 4, :].mean() < 0.35, "independent noise must be weakly coherent"
    assert np.allclose(coh, coh.transpose(0, 2, 1, 3), atol=1e-5), "coherence must be symmetric"
    assert np.allclose(np.exp(ab).sum(-1), 1.0, atol=0.02), "relative powers must sum to ~1"

    x = rng.normal(0, 10, (19, 45 * SF))
    x[5, 12 * SF] = 900.0                            # artifact in window 1
    x[7, 30 * SF:40 * SF] = 0.0                      # flat channel in window 3
    kept, rejected, total = clean_windows(x)
    assert total == 4 and rejected == 2 and len(kept) == 2, (total, rejected, len(kept))
    many, _, _ = clean_windows(rng.normal(0, 10, (19, 600 * SF)))
    assert len(many) == MAX_WINDOWS, "windows per subject must be capped"

    y = harmonise(rng.normal(0, 10, (19, 20 * 256)), 256.0, False)
    assert y.shape == (19, 20 * SF), y.shape
    assert np.allclose(y.mean(0), 0, atol=1e-9), "common average reference not applied"

    assert norm_name("EEG Fp1-LE") == "FP1" and norm_name("T7") == "T3" and norm_name("EEG Pz-REF") == "PZ"

    subject = np.repeat(np.array([f"s{i}" for i in range(12)]), 5)
    yy = np.repeat(np.arange(12) % 2, 5)
    rows = np.arange(len(yy))
    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=0)
    for a, b in sgkf.split(rows, yy, subject):
        assert not set(subject[a]) & set(subject[b]), "grouped split leaked a subject"
    prob = np.tile([0.2, 0.4, 0.6, 0.8, 1.0], 12)
    uniq, lab, score = subject_scores(subject, yy, prob, rows)
    assert len(uniq) == 12 and np.allclose(score, 0.6), "subject score must be the window mean"
    print("selfcheck ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["features", "run"])
    ap.add_argument("--datasets", nargs="*", default=list(DATASET_DISEASE))
    ap.add_argument("--disease", choices=sorted(set(DATASET_DISEASE.values())))
    ap.add_argument("--out", default="~/eeg/raw")
    ap.add_argument("--tune", default="~/eeg/per_disorder/tune.json")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    args.out = Path(args.out).expanduser()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.command == "features":
        return cmd_features(args)
    if not args.disease:
        ap.error("run needs --disease")
    return cmd_run(args, device)


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else main()
