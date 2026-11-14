"""What the final system relies on, per disease (electrode x frequency band).

Two independent views, which must agree before a pattern is reported:
    effect size   Cohen's d (patients - controls) of each feature computed WITHIN each site, then
                  averaged over sites; 'agree' = share of sites with the same sign. Model-free and
                  immune to site offsets.
    model         coefficients of the frozen logistic-regression member of final_system.py
                  (relative + 1/f-corrected band power + coherence, z-scored per site), one fit per
                  outer fold. 'stability' = mean pairwise Spearman between the 5 folds' coefficient
                  vectors; 'agreement' = Spearman between mean coefficient and pooled d.
Features shown: relative band power (rel) and 1/f-corrected periodic power (per) per electrode.
Positive = higher in patients (AD-vs-FTD task: higher in AD).

    python explain_final.py --out ~/eeg/raw2
Writes results/explain_final/<task>.json and <task>.png.
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import park as P  # noqa: E402
import pooled_deep as Q  # noqa: E402
import prep_v2 as V  # noqa: E402
from baselines import lr_probe  # noqa: E402

TASKS = {"alzheimers": "Alzheimer's vs control", "ftd": "FTD vs control", "alzheimers_vs_ftd": "Alzheimer's vs FTD",
         "mood": "Depression vs control", "schizophrenia": "Schizophrenia vs control"}
CH = [c.capitalize() if c.startswith("FP") else c for c in P.CHANNELS]
NB = len(P.BANDS)


def cohen_d(x, y):
    a, b = x[y == 1], x[y == 0]
    sp = np.sqrt(((len(a) - 1) * a.var(0, ddof=1) + (len(b) - 1) * b.var(0, ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean(0) - b.mean(0)) / (sp + 1e-12)


def names(kind):
    return [f"{kind} {b} {c}" for c in CH for b in P.BANDS]  # (19, 6) flattened row-major


def topo_row(axes, vals, title, vmax, info):
    import mne
    for b, ax in enumerate(axes):
        mne.viz.plot_topomap(vals[:, b], info, axes=ax, show=False, cmap="RdBu_r", vlim=(-vmax, vmax),
                             contours=0, sensors=True)
        ax.set_title(f"{title}\n{P.BANDS[b]}", fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/eeg/raw2")
    ap.add_argument("--save", default="results/explain_final")
    args = ap.parse_args()
    import mne
    mne.set_log_level("ERROR")
    info = mne.create_info(CH, 128, "eeg")
    info.set_montage("standard_1020")
    out, save = Path(args.out).expanduser(), Path(args.save)
    save.mkdir(parents=True, exist_ok=True)
    for task, title in TASKS.items():
        u, ys, site, tab, _ = V.subject_table(out / "features_v2", task, "median")
        nrel, nper, ncoh = tab["ab"].shape[1], tab["per"].shape[1], tab["coh"].shape[1]
        z = lambda a: (a - a.mean(0)) / (a.std(0) + 1e-8)  # noqa: E731
        X = V.per_site(np.concatenate([tab["ab"], tab["coh"], tab["per"]], 1), site, z)
        # model view: frozen LR member, one fit per outer fold (same folds as final_system)
        coefs = np.stack([lr_probe().fit(X[a], ys[a])[-1].coef_[0]
                          for a, _ in Q.folds({"u": u, "ys": ys, "site": site})])
        keep = np.r_[np.arange(nrel), nrel + ncoh + np.arange(nper)]  # rel + per, drop coherence for display
        cf = coefs[:, keep]
        stability = float(np.mean([spearmanr(cf[i], cf[j])[0] for i, j in combinations(range(len(cf)), 2)]))
        # effect-size view: within-site Cohen's d, averaged
        F = np.concatenate([tab["ab"], tab["per"]], 1)
        ds = np.stack([cohen_d(F[site == s], ys[site == s]) for s in np.unique(site)])
        d, agree = ds.mean(0), (np.sign(ds) == np.sign(ds.mean(0))).mean(0)
        agreement = float(spearmanr(cf.mean(0), d)[0])
        labels = names("rel") + names("per")
        top = np.argsort(-np.abs(d))[:12]
        res = {"title": title, "n": int(len(u)), "sites": sorted(set(site.tolist())),
               "fold_stability_spearman": stability, "model_vs_effect_spearman": agreement,
               "top_effects": [{"feature": labels[i], "cohen_d": float(d[i]), "site_sign_agreement": float(agree[i]),
                                "lr_coef_mean": float(cf.mean(0)[i]), "lr_coef_sign_folds": float((np.sign(cf[:, i]) == np.sign(cf.mean(0)[i])).mean())}
                               for i in top],
               "band_mean_abs_d": {k: {b: float(np.abs(d[o:o + nrel].reshape(19, NB)[:, j]).mean())
                                       for j, b in enumerate(P.BANDS)} for k, o in (("rel", 0), ("per", nrel))}}
        (save / f"{task}.json").write_text(json.dumps(res, indent=1))
        vmax = max(0.3, float(np.abs(d).max()))
        fig, axes = plt.subplots(2, NB, figsize=(12, 4.6))
        topo_row(axes[0], d[:nrel].reshape(19, NB), "relative power", vmax, info)
        topo_row(axes[1], d[nrel:].reshape(19, NB), "1/f-corrected", vmax, info)
        fig.suptitle(f"{title}: Cohen's d within site, averaged (red = higher in "
                     f"{'AD' if task == 'alzheimers_vs_ftd' else 'patients'}); |d| scale ±{vmax:.2f}\n"
                     f"LR fold stability {stability:.2f}, model-vs-effect agreement {agreement:.2f}", fontsize=10)
        fig.tight_layout()
        fig.savefig(save / f"{task}.png", dpi=130)
        plt.close(fig)
        print(f"{task:<18} stability {stability:.2f} agreement {agreement:.2f} | top: "
              + "; ".join(f"{labels[i]} d={d[i]:+.2f} ({agree[i]:.0%} sites)" for i in top[:5]), flush=True)


if __name__ == "__main__":
    # self-check of cohen_d sign and scale
    rng = np.random.default_rng(0)
    yy = np.r_[np.ones(500), np.zeros(500)].astype(int)
    xx = rng.normal(size=(1000, 1)) + yy[:, None] * 0.8
    assert abs(cohen_d(xx, yy)[0] - 0.8) < 0.15
    main()
