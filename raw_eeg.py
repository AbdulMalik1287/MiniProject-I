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
    python raw_eeg.py run --disease alzheimers_vs_ftd

Tasks (--disease), each positive group vs negative group from the same hospitals:
    schizophrenia       aseeg, warsaw                     SZ vs controls
    mood                mumtaz                            MDD vs controls
    alzheimers          ds004504                          Alzheimer's vs controls
    ftd                 ds004504                          frontotemporal dementia vs controls
    alzheimers_vs_ftd   ds004504                          Alzheimer's vs FTD - same hospital, so a
                                                          fair test of telling two diseases apart
    parkinsons          ds004584, ds003490, ds002778      Parkinson's vs controls, three hospitals

Datasets (19 channels of the 10-20 system are taken from every cap):
    aseeg     Bagherzadeh 2026, Zenodo 18029536, CC-BY    51 SZ + 50 ctl, eyes closed, Iran
    warsaw    Olejarczyk 2017, RepOD 0107441              14 SZ + 14 ctl, eyes closed, Poland
    mumtaz    Mumtaz 2016, figshare 4244171, CC-BY        30 MDD + 28 ctl, eyes closed, Malaysia
              (6 of 64 eyes-closed files are dead links on figshare)
    ds004504  Miltiadous 2023, OpenNeuro, CC0             36 AD + 23 FTD + 29 ctl, eyes closed, Greece
    ds004584  Singh et al., OpenNeuro, CC0                100 PD + 49 ctl, eyes open, Iowa (Pz reference)
    ds003490  Cavanagh 2021, OpenNeuro, CC0               25 PD (OFF medication) + 25 ctl, New Mexico;
                                                          only the instructed eyes-open minute is used
    ds002778  Rockhill et al., OpenNeuro, CC0             15 PD (ses-off) + 16 ctl, eyes open, San Diego

Eyes-open and eyes-closed recordings never meet inside one task, and psychiatric
and neurological tasks never share a model: their hospitals, ages and recording
conditions differ, so their probabilities are not comparable.

Pipeline, identical for every source:
    notch at the local mains frequency (50 Hz Europe/Asia, 60 Hz US) at native rate
    -> resample 128 Hz -> 1-45 Hz FIR
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
across datasets; cross-hospital transfer (train on one, test on another); and
with three or more hospitals, leave-one-hospital-out. Transfer directly tests
whether a disease signature carries across sites.
"""
import argparse
import csv
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
# line: mains frequency for the notch. eyes: recording condition. patient: the
# group a legacy binary cache's label 1 means (None when a dataset has several).
DATASETS = {
    "aseeg": {"line": 50, "eyes": "closed", "patient": "schizophrenia"},
    "warsaw": {"line": 50, "eyes": "closed", "patient": "schizophrenia"},
    "mumtaz": {"line": 50, "eyes": "closed", "patient": "mood"},
    "ds004504": {"line": 50, "eyes": "closed", "patient": None},
    "ds004584": {"line": 60, "eyes": "open", "patient": "parkinsons"},
    "ds003490": {"line": 60, "eyes": "open", "patient": "parkinsons"},
    "ds002778": {"line": 60, "eyes": "open", "patient": "parkinsons"},
}
TASKS = {  # name -> (positive group, negative group, datasets)
    "schizophrenia": ("schizophrenia", "control", ["aseeg", "warsaw"]),
    "mood": ("mood", "control", ["mumtaz"]),
    "alzheimers": ("alzheimers", "control", ["ds004504"]),
    "ftd": ("ftd", "control", ["ds004504"]),
    "alzheimers_vs_ftd": ("alzheimers", "ftd", ["ds004504"]),
    "parkinsons": ("parkinsons", "control", ["ds004584", "ds003490", "ds002778"]),
}
for _t, (_p, _n, _ds) in TASKS.items():  # a task must never mix eyes-open and eyes-closed data
    assert len({DATASETS[d]["eyes"] for d in _ds}) == 1, f"task {_t} mixes recording conditions"
ROOT = Path(__file__).resolve().parents[3] / "data" / "external"
if not ROOT.exists():  # running from the main checkout rather than a worktree
    ROOT = Path(__file__).resolve().parent / "data" / "external"
RAW_EPOCHS = 20  # ~60k graph visits per fit, comparable to Park's 150 epochs on ~230 graphs


# ------------------------------------------------------------------- loading


def norm_name(ch):
    base = ch.replace("EEG ", "").strip().split("-")[0].upper()
    return ALIASES.get(base, base)


def to_montage(ch_names, data, add_ref=None, label=""):
    """(channels, n) with arbitrary cap labels -> (19, n) in CH order.

    add_ref names an electrode of CH that is the recording reference and so is
    absent from the file. It is inserted as zeros - the reference reads zero
    against itself - and the later average reference reconstructs it exactly.
    """
    idx = {}
    for i, c in enumerate(ch_names):
        idx.setdefault(norm_name(c), i)
    missing = [c for c in CH if c not in idx and c != add_ref]
    if missing:
        raise ValueError(f"{label}: missing channels {missing}")
    x = np.zeros((len(CH), data.shape[1]))
    for j, c in enumerate(CH):
        if c in idx:
            x[j] = data[idx[c]]
    return x


def load_raw(path, add_ref=None, crop=None):
    """Read EDF/BDF/EEGLAB into (19, n) microvolts; crop is (tmin, tmax) seconds."""
    import mne
    mne.set_log_level("ERROR")
    reader = {".edf": mne.io.read_raw_edf, ".bdf": mne.io.read_raw_bdf,
              ".set": mne.io.read_raw_eeglab}[path.suffix.lower()]
    raw = reader(path, preload=True)
    if crop:
        raw.crop(tmin=crop[0], tmax=min(crop[1], raw.times[-1]))
    return to_montage(raw.ch_names, raw.get_data() * 1e6, add_ref, path.name), raw.info["sfreq"]


def read_tsv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def eyes_open_span(events):
    """(tmin, tmax) of the instructed eyes-open minute in a ds003490 rest recording."""
    onsets = [float(e["onset"]) for e in events if e.get("trial_type", "").lower().startswith("eyes open")]
    if not onsets:
        kinds = sorted({e.get("trial_type", "") for e in events})
        raise ValueError(f"no eyes-open events; trial types present: {kinds}")
    return min(onsets), max(onsets) + 1.0


def recordings(dataset):
    """Yield (subject_id, group, data uV (19, n), sfreq, already_filtered)."""
    d = ROOT / dataset
    if dataset == "aseeg":
        import h5py
        for folder, group in (("normal", "control"), ("sz", "schizophrenia")):
            for fp in sorted((d / "EC" / folder).glob("*.mat")):
                with h5py.File(fp, "r") as h:
                    x = np.array(h["preprocessed_data"], dtype=np.float64).T
                yield f"aseeg-{fp.stem}", group, x, 128.0, True
    elif dataset == "mumtaz":
        for fp in sorted(d.glob("*.edf")):
            cls, sid = fp.name.split()[:2]
            # Subject numbers restart per class ("H S1" and "MDD S1" are
            # different people), so the key must include the class.
            yield (f"mumtaz-{cls}-{sid}", "mood" if cls == "MDD" else "control",
                   *load_raw(fp), False)
    elif dataset == "warsaw":
        for fp in sorted(d.glob("*.edf")):
            yield (f"warsaw-{fp.stem}", "schizophrenia" if fp.stem.startswith("s") else "control",
                   *load_raw(fp), False)
    elif dataset == "ds004504":
        groups = {"A": "alzheimers", "F": "ftd", "C": "control"}
        for p in read_tsv(d / "participants.tsv"):
            sid = p["participant_id"]
            fp = d / sid / "eeg" / f"{sid}_task-eyesclosed_eeg.set"
            yield f"{dataset}-{sid}", groups[p["Group"]], *load_raw(fp), False
    elif dataset == "ds004584":
        groups = {"PD": "parkinsons", "Control": "control"}
        for p in read_tsv(d / "participants.tsv"):
            sid = p["participant_id"]
            fp = d / sid / "eeg" / f"{sid}_task-Rest_eeg.set"
            yield f"{dataset}-{sid}", groups[p["GROUP"]], *load_raw(fp, add_ref="PZ"), False
    elif dataset == "ds003490":
        for p in read_tsv(d / "participants.tsv"):
            sid = p["participant_id"]
            if p["Group"] == "CTL":
                ses = "ses-01"
            else:  # patients came twice; use the OFF-medication visit
                ses = "ses-01" if p["sess1_Med"] == "OFF" else "ses-02" if p["sess2_Med"] == "OFF" else None
            if ses is None:
                continue
            base = d / sid / ses / "eeg" / f"{sid}_{ses}_task-Rest"
            span = eyes_open_span(read_tsv(Path(f"{base}_events.tsv")))
            yield (f"{dataset}-{sid}", "control" if p["Group"] == "CTL" else "parkinsons",
                   *load_raw(Path(f"{base}_eeg.set"), crop=span), False)
    elif dataset == "ds002778":
        for fp in sorted(d.glob("sub-*/ses-*/eeg/*_eeg.bdf")):
            sid, ses = fp.parts[-4], fp.parts[-3]
            if ses not in ("ses-hc", "ses-off"):
                continue
            yield f"{dataset}-{sid}", "control" if ses == "ses-hc" else "parkinsons", *load_raw(fp), False
    else:
        raise ValueError(dataset)


def harmonise(x, sf, already_filtered, line=50.0):
    import mne
    mne.set_log_level("ERROR")
    if not already_filtered:
        x = mne.filter.notch_filter(x, sf, float(line))
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
        abs_, cohs, subj, grp, qc = [], [], [], [], []
        for sid, group, x, sf, filt in recordings(ds):
            w, rejected, total = clean_windows(harmonise(x, sf, filt, DATASETS[ds]["line"]))
            qc.append({"subject": sid, "group": group, "windows_total": total,
                       "rejected": rejected, "kept": int(len(w))})
            if len(w) == 0:
                continue
            a, c = spectral_features(w)
            abs_.append(a)
            cohs.append(c)
            subj += [sid] * len(w)
            grp += [group] * len(w)
        np.savez_compressed(out / f"{ds}.npz", ab=np.concatenate(abs_), coh=np.concatenate(cohs),
                            subject=np.array(subj), group=np.array(grp))
        (out / f"{ds}_qc.json").write_text(json.dumps(qc, indent=1))
        for group in sorted({r["group"] for r in qc}):
            q = [r for r in qc if r["group"] == group]
            rej = sum(r["rejected"] for r in q) / max(sum(r["windows_total"] for r in q), 1)
            print(f"[features] {ds:<9} {group:<14} {len(q):>3} subjects, "
                  f"{sum(r['kept'] for r in q):>5} windows kept, {rej:.1%} rejected", flush=True)
        print(f"[features] {ds}: {sum(r['kept'] == 0 for r in qc)} subjects with no clean window",
              flush=True)


# -------------------------------------------------------------- experiments


def cache_groups(ds, part):
    """Group label per window; caches written before group labels existed hold 0/1."""
    if "group" in part:
        return part["group"].astype(str)
    patient = DATASETS[ds]["patient"]
    if patient is None:
        raise ValueError(f"{ds} cache has no group labels; rerun `features`")
    return np.where(part["label"] == 1, patient, "control")


def load_features(out, datasets):
    parts = [np.load(out / "features" / f"{ds}.npz") for ds in datasets]
    cat = lambda k: np.concatenate([p[k] for p in parts])  # noqa: E731
    groups = np.concatenate([cache_groups(ds, p) for ds, p in zip(datasets, parts)])
    dsname = np.concatenate([[ds] * len(p["subject"]) for ds, p in zip(datasets, parts)])
    return cat("ab"), cat("coh"), cat("subject"), groups, dsname


def task_rows(groups, positive, negative):
    """Rows belonging to the task, and 0/1 labels for those rows (1 = positive group)."""
    rows = np.where(np.isin(groups, [positive, negative]))[0]
    return rows, (groups[rows] == positive).astype(np.int64)


def subject_scores(subject, y, win_prob, rows):
    """Aggregate window probabilities to one score per subject."""
    s, p, yr = subject[rows], win_prob[rows], y[rows]  # all three aligned to `rows`
    uniq = np.unique(s)
    score = np.array([p[s == u].mean() for u in uniq])
    lab = np.array([yr[s == u][0] for u in uniq])
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


def cmd_run(args, device):
    positive, negative, wanted = TASKS[args.disease]
    datasets = [d for d in wanted if (args.out / "features" / f"{d}.npz").exists()]
    if not datasets:
        raise FileNotFoundError(f"no feature caches for {args.disease}; run `features` first")
    if len(datasets) < len(wanted):
        print(f"[run] WARNING: missing feature caches for {sorted(set(wanted) - set(datasets))}")
    tune = Path(args.tune).expanduser()
    base = json.loads(tune.read_text())["best"]["config"] if tune.exists() else D.GRID[-1]
    cfg = {**base, "epochs": RAW_EPOCHS}
    print(f"[run] {args.disease}: {positive} vs {negative}, datasets {datasets}; GNN "
          f"{cfg['model']}/{cfg['pool']}/topk={cfg['topk']} (tuned on Park, not on these data), "
          f"{cfg['epochs']} epochs\n")
    ab, coh, subject, groups, ds = load_features(args.out, datasets)
    keep, y = task_rows(groups, positive, negative)
    ab, coh, subject, ds = ab[keep], coh[keep], subject[keep], ds[keep]

    def fmt(r):
        return (f"AUC {r['auc']:.3f} [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]  "
                f"({r['n_patients']} {positive} / {r['n_controls']} {negative})")

    report = {"disease": args.disease, "positive": positive, "negative": negative,
              "datasets": datasets, "eyes": DATASETS[datasets[0]]["eyes"], "config": cfg}

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

    if len(datasets) > 2:  # train on all other hospitals, test on the one left out
        for d in datasets:
            tr, te = np.where(ds != d)[0], np.where(ds == d)[0]
            assert not set(subject[tr]) & set(subject[te]), "subject in both sides of a site split"
            model = D.fit(cfg, ab, coh, y, tr, device)
            prob = np.zeros(len(y))
            prob[te] = D.predict_proba(model, cfg, ab, coh, te, device)
            _, lab, score = subject_scores(subject, y, prob, te)
            report[f"loso:{d}"] = auc_report(lab, score)
            print(f"leave-one-hospital-out, test {d:<9} {fmt(report[f'loso:{d}'])}", flush=True)

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
    # A subset of rows (one dataset out of a pooled array) must index consistently.
    sub = np.arange(20, 45)                               # subjects s4..s8
    marked = prob.copy()
    marked[25:30] = 0.0                                   # subject s5 only
    u2, l2, sc2 = subject_scores(subject, yy, marked, sub)
    assert list(u2) == ["s4", "s5", "s6", "s7", "s8"], list(u2)
    assert np.isclose(sc2[1], 0.0) and np.allclose(sc2[[0, 2, 3, 4]], 0.6), sc2
    assert list(l2) == [0, 1, 0, 1, 0], list(l2)

    # Montage assembly from a 10-10 cap, with the reference electrode missing.
    cap = ["Fp1", "Fz", "F3", "F7", "FC5", "C3", "T7", "P3", "P7", "O1", "Oz", "O2", "P4", "P8",
           "Cz", "C4", "T8", "F4", "F8", "Fp2", "AF3"]  # no Pz: it is the reference
    data = rng.normal(0, 5, (len(cap), 400))
    m = to_montage(cap, data, add_ref="PZ")
    assert m.shape == (19, 400) and np.allclose(m[CH.index("PZ")], 0), "reference must enter as zeros"
    assert np.allclose(m[CH.index("T3")], data[cap.index("T7")]), "T7 must map onto T3"
    assert np.allclose(m[CH.index("FP2")], data[cap.index("Fp2")])
    try:
        to_montage(cap, data)
    except ValueError:
        pass
    else:
        raise AssertionError("a missing non-reference channel must be rejected")
    # Average referencing a zero-reference channel reconstructs it as -mean(others).
    car = m - m.mean(0, keepdims=True)
    others = [i for i, c in enumerate(CH) if c != "PZ"]
    assert np.allclose(car[CH.index("PZ")], -m[others].sum(0) / 19)

    # US recordings (60 Hz mains, 500 Hz) must go through the same chain and land on the common grid.
    y60 = harmonise(rng.normal(0, 10, (19, 20 * 500)), 500.0, False, line=60)
    assert y60.shape == (19, 20 * SF) and np.allclose(y60.mean(0), 0, atol=1e-9)
    assert all(DATASETS[d]["line"] == 60 for d in ("ds004584", "ds003490", "ds002778"))

    events = [{"onset": "10.0", "trial_type": "Eyes Open: Every 1000 ms"},
              {"onset": "69.0", "trial_type": "Eyes Open: Every 1000 ms"},
              {"onset": "90.0", "trial_type": "Eyes Closed: Every 1000 ms"},
              {"onset": "0", "trial_type": "STATUS"}]
    assert eyes_open_span(events) == (10.0, 70.0)
    try:
        eyes_open_span(events[2:])
    except ValueError:
        pass
    else:
        raise AssertionError("missing eyes-open events must be an error, not a silent fallback")

    g = np.array(["control", "alzheimers", "ftd", "control", "ftd", "alzheimers"])
    r_, y_ = task_rows(g, "alzheimers", "ftd")
    assert list(r_) == [1, 2, 4, 5] and list(y_) == [1, 0, 0, 1], (r_, y_)
    legacy = {"label": np.array([0, 1, 1]), "subject": np.array(["a", "b", "c"])}
    assert list(cache_groups("warsaw", legacy)) == ["control", "schizophrenia", "schizophrenia"]
    for t, (p_, n_, ds_) in TASKS.items():
        assert p_ != n_ and all(d in DATASETS for d in ds_), t
    print("selfcheck ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["features", "run"])
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--disease", choices=list(TASKS), help="task to run; see module docstring")
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
