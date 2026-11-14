"""Phase 5: richer per-channel EEG markers, computed from the cached 10 s raw windows.

The first-pass features (relative band power + coherence) were picked to be
amplifier-gain invariant for cross-hospital work. Within one dataset gain is
shared, so established absolute and shape markers become usable:

    log absolute band power (6)          slowing in dementia, beta in Parkinson's
    aperiodic exponent, offset (2)       1/f slope of log-PSD vs log-f over 2-40 Hz,
                                         alpha band (7-14 Hz) excluded from the fit
    peak alpha frequency (1)             argmax of PSD in 7-13 Hz; drops in dementia
    theta/alpha, theta/beta (2)          log power ratios
    spectral entropy (1)                 normalised Shannon entropy of the 1-45 Hz PSD
    Hjorth mobility, complexity (2)
    permutation entropy (1)              order 3, normalised (antropy)
-> (windows, 19, 15). Frontal alpha asymmetry is a difference of two channels'
alpha power, which a linear model or GNN can form from these, so it is not added.
"""
import numpy as np

SF = 128
BANDS = [(1, 4), (4, 8), (8, 12), (12, 25), (25, 30), (30, 45)]
NAMES = ([f"logabs_{b}" for b in ("delta", "theta", "alpha", "beta", "highbeta", "gamma")]
         + ["aperiodic_exp", "aperiodic_off", "peak_alpha", "theta_alpha", "theta_beta",
            "spec_entropy", "hjorth_mob", "hjorth_comp", "perm_entropy"])


def rich(win, chunk=512):
    """(w, 19, T) float -> (w, 19, 15) float32."""
    out = np.zeros((len(win), win.shape[1], len(NAMES)), np.float32)
    for k in range(0, len(win), chunk):
        out[k:k + chunk] = _rich(win[k:k + chunk].astype(np.float64))
    return out


def _rich(w):
    import antropy as ant
    nper = 2 * SF
    starts = np.arange(0, w.shape[-1] - nper + 1, SF)
    seg = np.stack([w[..., s:s + nper] for s in starts], -2)  # (w, c, s, nper)
    psd = (np.abs(np.fft.rfft(seg * np.hanning(nper), axis=-1)) ** 2).mean(-2)  # (w, c, f)
    f = np.fft.rfftfreq(nper, 1 / SF)
    bp = np.stack([psd[..., (f >= lo) & (f < hi)].sum(-1) for lo, hi in BANDS], -1) + 1e-12
    feats = [np.log(bp)]
    fit = (f >= 2) & (f <= 40) & ~((f >= 7) & (f <= 14))
    lf, lp = np.log10(f[fit]), np.log10(psd[..., fit] + 1e-12)
    slope = ((lf - lf.mean()) * (lp - lp.mean(-1, keepdims=True))).sum(-1) / ((lf - lf.mean()) ** 2).sum()
    feats += [-slope[..., None], (lp.mean(-1) - slope * lf.mean())[..., None]]
    a = (f >= 7) & (f <= 13)
    feats.append(f[a][psd[..., a].argmax(-1)][..., None])
    feats += [np.log(bp[..., 1] / bp[..., 2])[..., None], np.log(bp[..., 1] / bp[..., 3])[..., None]]
    band = (f >= 1) & (f <= 45)
    p = psd[..., band] / psd[..., band].sum(-1, keepdims=True)
    feats.append((-(p * np.log(p + 1e-12)).sum(-1) / np.log(band.sum()))[..., None])
    d1, d2 = np.diff(w, axis=-1), np.diff(w, n=2, axis=-1)
    v0, v1, v2 = w.var(-1), d1.var(-1), d2.var(-1)
    mob = np.sqrt(v1 / (v0 + 1e-12))
    feats += [mob[..., None], (np.sqrt(v2 / (v1 + 1e-12)) / (mob + 1e-12))[..., None]]
    pe = np.array([[ant.perm_entropy(ch, order=3, normalize=True) for ch in x] for x in w])
    feats.append(pe[..., None])
    return np.concatenate(feats, -1)


if __name__ == "__main__":  # self-check on synthetic 1/f + 10 Hz alpha
    rng = np.random.default_rng(0)
    t = np.arange(10 * SF) / SF
    pink = np.cumsum(rng.normal(size=(4, 19, len(t))), -1)
    pink -= pink.mean(-1, keepdims=True)
    x = pink * 0.1 + 5 * np.sin(2 * np.pi * 10 * t)
    r = rich(x)
    pa = r[..., NAMES.index("peak_alpha")]
    ex = r[..., NAMES.index("aperiodic_exp")]
    assert np.allclose(pa, 10, atol=0.6), pa.mean()
    assert 1.4 < ex.mean() < 2.6, ex.mean()  # brown-ish noise has slope ~2
    print("rich_features selfcheck ok", r.shape)
