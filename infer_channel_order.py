#!/usr/bin/env python3
"""Recover the undocumented column order of the ASEEG .mat files from the data.

ASEEG (Bagherzadeh et al., Sci Data 2026) ships 19-column matrices with no
channel labels, and neither the paper nor the Zenodo record lists the order.
Guessing wrong would silently scramble node identity in every graph built from
it, so the order is inferred and checked two independent ways:

1. Spatial correlation structure. Mumtaz et al. EDF files carry channel labels.
   After common-average re-referencing, the inter-channel correlation matrix is
   a stable fingerprint of electrode geometry. Candidate orders are scored by
   how well ASEEG's matrix matches Mumtaz's, and an unconstrained search
   (quadratic assignment) checks no unlisted order fits better.
2. Alpha topography. Eyes-closed alpha (8-13 Hz) is strongest over occipital and
   parietal sites; the columns carrying most relative alpha must map there.

Correlation structure is nearly symmetric left/right AND front/back, so (1)
alone cannot tell Fp1 from O1. Physiology resolves front/back: eyes-closed
alpha is posterior, and slow (1-4 Hz) power from residual eye movement is
frontal. An order is accepted only if it matches the fingerprint AND both
physiological markers point the right way. Left/right cannot be resolved from
resting EEG, so the accepted order is a known device export order that fits
exactly, and that residual assumption is printed.
"""
import sys
from pathlib import Path

import h5py
import mne
import numpy as np
from scipy.optimize import quadratic_assignment
from scipy.signal import welch

mne.set_log_level("ERROR")
EXT = Path(__file__).resolve().parents[3] / "data" / "external"
if not EXT.exists():  # running from the main checkout rather than a worktree
    EXT = Path(__file__).resolve().parent / "data" / "external"

CH = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T3", "C3", "CZ",
      "C4", "T4", "T5", "P3", "PZ", "P4", "T6", "O1", "O2"]
CANDIDATES = {
    "10-20 row order": CH,
    "left/right pairs, midline last": ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1",
                                       "O2", "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ"],
    "left chain, midline, right chain": ["FP1", "F3", "C3", "P3", "O1", "F7", "T3", "T5", "FZ",
                                         "FP2", "F4", "C4", "P4", "O2", "F8", "T4", "T6", "CZ", "PZ"],
    "alphabetical": sorted(CH),
}
POSTERIOR = {"O1", "O2", "PZ", "P3", "P4", "T5", "T6"}
FRONTAL = {"FP1", "FP2", "F7", "F3", "FZ", "F4", "F8"}
SF = 128


def corr_fingerprint(x):
    """x (channels, samples) -> Fisher-z channel correlation after average reference."""
    x = x - x.mean(0, keepdims=True)
    r = np.corrcoef(x)
    np.fill_diagonal(r, 0)
    return np.arctanh(np.clip(r, -0.999, 0.999))


def rel_power(x):
    """Relative (alpha 8-13 Hz, delta 1-4 Hz) power per channel."""
    f, p = welch(x, fs=SF, nperseg=SF * 2)
    band = lambda lo, hi: p[:, (f >= lo) & (f < hi)].sum(1)  # noqa: E731
    total = band(1, 45)
    return band(8, 13) / total, band(1, 4) / total


def physiology(order, alpha, delta):
    """(# of top-5 alpha columns that are posterior, # of top-5 delta columns frontal)."""
    top = lambda v: [order[i] for i in np.argsort(-v)[:5]]  # noqa: E731
    return (sum(c in POSTERIOR for c in top(alpha)), sum(c in FRONTAL for c in top(delta)),
            top(alpha), top(delta))


def mumtaz_reference():
    mats = []
    for fp in sorted((EXT / "mumtaz").glob("*.edf")):
        raw = mne.io.read_raw_edf(fp, preload=True)
        names = {c.replace("EEG ", "").split("-")[0].upper(): c for c in raw.ch_names}
        if not all(c in names for c in CH):
            continue
        raw.pick([names[c] for c in CH]).filter(1, 45).resample(SF)
        mats.append(corr_fingerprint(raw.get_data()))
    return np.mean(mats, 0), len(mats)


def aseeg_fingerprint():
    mats, alpha, delta = [], [], []
    for fp in sorted((EXT / "aseeg" / "EC").rglob("*.mat")):
        with h5py.File(fp, "r") as h:
            x = np.array(h["preprocessed_data"], dtype=np.float64).T  # (19, samples)
        mats.append(corr_fingerprint(x))
        a, d = rel_power(x)
        alpha.append(a)
        delta.append(d)
    return np.mean(mats, 0), np.mean(alpha, 0), np.mean(delta, 0), len(mats)


def upper(m):
    return m[np.triu_indices_from(m, 1)]


def main():
    ref, n_ref = mumtaz_reference()
    asg, alpha, delta, n_asg = aseeg_fingerprint()
    print(f"reference: {n_ref} Mumtaz recordings, labelled channels; target: {n_asg} ASEEG recordings\n")

    # Sanity-check the physiological markers on the LABELLED dataset first -
    # if they fail where the truth is known, they prove nothing on ASEEG.
    mz_alpha, mz_delta = [], []
    for fp in sorted((EXT / "mumtaz").glob("H *.edf")):
        raw = mne.io.read_raw_edf(fp, preload=True)
        names = {c.replace("EEG ", "").split("-")[0].upper(): c for c in raw.ch_names}
        if all(c in names for c in CH):
            raw.pick([names[c] for c in CH]).filter(1, 45).resample(SF)
            x = raw.get_data()
            a, d = rel_power(x - x.mean(0, keepdims=True))
            mz_alpha.append(a)
            mz_delta.append(d)
    mp, mf, mta, mtd = physiology(CH, np.mean(mz_alpha, 0), np.mean(mz_delta, 0))
    print(f"marker check on labelled Mumtaz controls: alpha top-5 {mta} ({mp}/5 posterior), "
          f"delta top-5 {mtd} ({mf}/5 frontal)\n")

    print("candidate orders (column j of ASEEG = candidate[j]):")
    scores = {}
    for name, order in CANDIDATES.items():
        perm = [order.index(c) for c in CH]  # ASEEG column index for each CH position
        s = np.corrcoef(upper(asg[np.ix_(perm, perm)]), upper(ref))[0, 1]
        post, front, ta, td = physiology(order, alpha, delta)
        scores[name] = (s, post, front, order)
        print(f"  {name:<34} fingerprint r={s:.3f}  alpha {ta} ({post}/5 post)  "
              f"delta {td} ({front}/5 front)")

    # Unconstrained search: best permutation of ASEEG columns onto labelled CH.
    best, best_score = None, -np.inf
    for seed in range(20):
        res = quadratic_assignment(ref, asg, method="faq",
                                   options={"maximize": True, "rng": seed, "P0": "randomized"})
        perm = res.col_ind
        s = np.corrcoef(upper(asg[np.ix_(perm, perm)]), upper(ref))[0, 1]
        if s > best_score:
            best, best_score = perm, s
    inferred = [None] * 19
    for pos, col in enumerate(best):
        inferred[col] = CH[pos]
    ipost, ifront, ita, itd = physiology(inferred, alpha, delta)
    print(f"\nunconstrained search best fingerprint r={best_score:.3f}")
    print(f"  inferred column order: {inferred}")
    print(f"  its physiology: alpha {ita} ({ipost}/5 post), delta {itd} ({ifront}/5 front)")

    name, (s, post, front, order) = max(scores.items(), key=lambda kv: kv[1][0])
    runner_up = sorted(v[0] for v in scores.values())[-2]
    fingerprint_ok = s - runner_up > 0.1 and s >= best_score - 0.05
    physiology_ok = post >= 4 and front >= 4
    # If the unconstrained optimum fits the geometry slightly better but places
    # alpha frontally, it is a front/back mirror that physiology rules out.
    search_is_mirror = best_score > s and ipost <= 1
    print(f"\nbest candidate: {name}  r={s:.3f} (next candidate r={runner_up:.3f}, "
          f"unconstrained r={best_score:.3f})")
    print(f"  fingerprint clearly separates candidates and is near the optimum: {fingerprint_ok}")
    print(f"  physiology (alpha posterior AND delta frontal): {physiology_ok}")
    if best_score > s:
        print(f"  unconstrained optimum rejected as a front/back mirror: {search_is_mirror}")
    ok = fingerprint_ok and physiology_ok and (best_score <= s or search_is_mirror)
    if ok:
        print(f"VERDICT: ACCEPT '{name}': {order}")
        print("  residual assumption: left/right (e.g. FP1 vs FP2) is not identifiable from "
              "resting EEG; taken from this standard export order.")
    else:
        print("VERDICT: UNRESOLVED - do not use ASEEG until confirmed")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
