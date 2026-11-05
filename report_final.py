"""Final-phase sections for make_report.py: everything run on the local GPU on 2026-11-05.

Reads only results/*.json written by ensemble.py, baselines.py, riemann.py,
labram_probe.py, pooled.py, sharpened.py, tigga.py, site_probe.py, explain.py
and results/final/raw_*.json; missing files drop their section.
"""
import json
import statistics as st
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
ORDER = ["alzheimers:ds004504", "ftd:ds004504", "alzheimers_vs_ftd:ds004504", "mood:mumtaz",
         "mood:ds003478", "schizophrenia:aseeg", "schizophrenia:warsaw", "schizophrenia:semmelweis",
         "parkinsons:ds004584", "parkinsons:ds003490", "parkinsons:ds002778", "parkinsons:ds007526"]
NAME = {"alzheimers:ds004504": "Alzheimer's vs control (Thessaloniki)",
        "ftd:ds004504": "FTD vs control (Thessaloniki)",
        "alzheimers_vs_ftd:ds004504": "Alzheimer's vs FTD (Thessaloniki)",
        "mood:mumtaz": "Depression (Mumtaz, Malaysia)", "mood:ds003478": "Depression (UNM, New Mexico)",
        "schizophrenia:aseeg": "Schizophrenia (ASEEG, Iran)", "schizophrenia:warsaw": "Schizophrenia (Warsaw)",
        "schizophrenia:semmelweis": "Schizophrenia (Semmelweis, Budapest)",
        "parkinsons:ds004584": "Parkinson's (Iowa)", "parkinsons:ds003490": "Parkinson's (New Mexico)",
        "parkinsons:ds002778": "Parkinson's (San Diego)", "parkinsons:ds007526": "Parkinson's (Tel Aviv)"}
MEMBERS = [("gatv2", "GATv2"), ("msbp", "MSBP-GNN"), ("lr_base", "LR"), ("lr_rich", "LR rich"),
           ("riemann", "Riemann"), ("labram", "LaBraM")]


def _load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def summary_bullets():
    e = _load("ensemble_v2.json")
    if not e:
        return []
    a = {k: e[k]["ensemble"]["auc"] for k in ORDER if k in e}
    hi = [NAME[k].split(",")[0].replace(" (", " - ").rstrip(")") + f" {v:.2f}" for k, v in a.items() if v >= 0.85]
    return [
        "Final within-dataset result (12 tasks on 10 open datasets, 4 disorders): a pre-declared ensemble of six "
        "methods (two GNNs, two logistic regressions, Riemannian covariance, frozen LaBraM) reaches subject-level "
        f"AUC {min(a.values()):.2f}-{max(a.values()):.2f}; above 0.85 for {', '.join(hi)}. Every number is from "
        "subject-wise cross-validation: no window of a test subject is ever trained on.",
        "Model choice matters less than the data: logistic regression on the same features matches the GNNs, and "
        "the gains came from new information (Riemannian covariance, 1/f and alpha-peak markers) and from "
        "combining methods. The Alzheimer's/FTD numbers sit at or above a 2026 benchmark of nine frozen "
        "representations on the same dataset (AUC 0.79-0.90 AD, 0.69-0.83 FTD).",
        "A published GATv2-LSTM (Tigga et al., Sci Rep 2026) was re-run from its released code: its edge list "
        "connects only rows 0-18 so the graph has no effect, and its random row split inflates accuracy.",
    ]


def add_sections(doc, h):
    """h: dict of make_report helpers - table, figure, bullets, ci, dot_whisker, FIG, S1, S2."""
    table, figure, bullets, ci = h["table"], h["figure"], h["bullets"], h["ci"]
    e = _load("ensemble_v2.json")
    if not e:
        return
    keys = [k for k in ORDER if k in e]

    doc.add_heading("5.6 Final within-dataset results (local GPU, 2026-11-05)", 2)
    doc.add_paragraph(
        "Each dataset is evaluated on its own subjects with 5-fold stratified cross-validation over subjects "
        "(seed 42); all six methods see the same folds. The headline is the ensemble: the mean of the six "
        "methods' within-fold rank-normalised subject scores. Its members were fixed before results were seen "
        "(every method built, not the ones that scored well), so it involves no selection on the test folds. "
        "Picking the best single method per row would; those numbers are shown only for reference.")
    rows = []
    for k in keys:
        r = e[k]
        best = max((m for m, _ in MEMBERS), key=lambda m: r[m]["auc"])
        rows.append((NAME[k], ci(r["ensemble"]), *(f"{r[m]['auc']:.3f}" for m, _ in MEMBERS),
                     dict(MEMBERS)[best]))
    table(doc, ["dataset", "ensemble AUC [95% CI]", *(n for _, n in MEMBERS), "best member"], rows,
          note="LR = logistic regression on relative band power + coherence; LR rich = on 1/f exponent/offset, "
               "peak alpha, absolute power, ratios, entropies, Hjorth; Riemann = filter-bank tangent space; "
               "LaBraM = frozen foundation-model embeddings (block 6) + logistic regression.")
    h["dot_whisker"](h["FIG"] / "final_auc.png", [NAME[k] for k in keys],
                     [("ensemble", h["S1"], [(e[k]["ensemble"]["auc"], *e[k]["ensemble"]["ci"]) for k in keys])],
                     "subject-level AUC with 95% bootstrap CI",
                     "Final within-dataset results (pre-declared six-method ensemble)", xlim=(0.3, 1.0))
    figure(doc, h["FIG"] / "final_auc.png", "Figure 10. Final results. Grey line: chance (0.5).")

    doc.add_heading("5.7 What improved the numbers", 2)
    b, r_, rm, lb = (_load(f) for f in ("baselines_base.json", "baselines_rich.json", "riemann.json", "labram.json"))
    bullets(doc, [
        "Architecture was not the bottleneck: logistic regression on the GNNs' own features matched them "
        "(Alzheimer's 0.898 both). A spatio-temporal GATv2-LSTM over consecutive windows also did not beat "
        "the plain GATv2.",
        "Filter-bank Riemannian covariance was the strongest single method on schizophrenia and Parkinson's "
        "(e.g. Warsaw 0.85, San Diego 0.80).",
        "Handcrafted markers (aperiodic 1/f exponent, peak alpha frequency, absolute band power) lifted "
        "Parkinson's Iowa from 0.66 to 0.75 and Mumtaz depression from 0.91 to 0.96 with logistic regression.",
        "A frozen foundation model (LaBraM) was at parity overall and best only on UNM depression.",
        "Combining all six methods gave a steady +0.02 to +0.07 over the GATv2 alone.",
    ])
    if b and r_ and rm and lb:
        table(doc, ["dataset", "LR", "LR rich", "Riemann", "LaBraM block 6"],
              [(NAME[k], f"{b[k]['logreg'][0]:.3f}", f"{r_[k]['logreg'][0]:.3f}", f"{rm[k]['riemann_fb'][0]:.3f}",
                f"{lb[k]['block6'][0]:.3f}") for k in keys if k in b and k in rm and k in lb],
              note="Mean AUC over 5 repeats of 5-fold subject-wise CV (different folds from the ensemble table).")

    pooled = _load("pooled.json")
    if pooled:
        doc.add_heading("5.8 Pooled multi-site training", 2)
        doc.add_paragraph(
            "Test folds come from one dataset; training adds every subject of the other same-condition datasets "
            "of that disorder. Site offsets are removed without labels (per-site z-scoring; per-site Riemannian "
            "re-centring). It helps Parkinson's and hurts depression and schizophrenia, where other hospitals "
            "mostly add site noise - consistent with the site probe in 5.10.")
        table(doc, ["target dataset", "LR", "LR rich", "Riemann"],
              [(NAME[k], *(f"{pooled[k][m][0]:.3f}" for m in ("lr_base", "lr_rich", "riemann")))
               for k in ORDER if k in pooled])

    tg = _load("tigga.json")
    if tg:
        doc.add_heading("5.9 Re-running a published GATv2-LSTM: leakage and an inert graph", 2)
        doc.add_paragraph(
            "Tigga, Kumari & Alnajjar (Sci Rep 16:34662, 2026; Christ University and BIT Mesra) report 76-89% on "
            "the same Alzheimer's/FTD dataset. Their released code was re-run with its exact model and training "
            "settings. Two findings: (1) each input row is a single time point, and PyG treats rows as nodes, so "
            "their 19x19 edge list connects only rows 0-18 - removing it changes nothing; (2) their random row "
            "split puts every subject in both train and test. Their 76% could not be reached even with their "
            "split, because their EEG-to-CSV step is not released.")
        runs = tg["runs"]
        out = []
        for p in ("leaky", "honest"):
            for edges in (True, False):
                g = [x for x in runs if x["protocol"] == p and x["edges"] == edges]
                out.append((f"{p} split, {'with' if edges else 'without'} edges",
                            f"{st.mean(x['row_bal_acc'] for x in g):.3f}",
                            f"{st.mean(x['subject_bal_acc'] for x in g):.3f}" if p == "honest" else "-"))
        table(doc, ["protocol", "row balanced accuracy", "subject balanced accuracy"], out,
              note="3 seeds; 3 classes (AD/FTD/control), chance 0.333.")

    sp = _load("site_probe.json")
    if sp:
        doc.add_heading("5.10 Why cross-hospital transfer fails: the hospital is easier than the disease", 2)
        doc.add_paragraph(
            "A linear model on subject-mean features names the hospital far better than the disease - even from "
            "healthy controls alone. Removing site offsets (ComBat, Euclidean alignment) or training a "
            "domain-adversarial GATv2 did not make schizophrenia transfer between hospitals (results/final/); "
            "this line of work was put on hold.")
        table(doc, ["task", "hospital (all)", "hospital (controls only)", "disease"],
              [(t, *(f"{v[k]['bal_acc']:.2f} (chance {v[k]['chance']:.2f})" for k in ("site", "site_controls", "disease")))
               for t, v in sp.items()], note="Balanced accuracy, 5-fold CV over subjects.")

    ex = sorted((RES / "explain").glob("*.json")) if (RES / "explain").exists() else []
    if ex:
        doc.add_heading("5.11 What the GATv2 relies on", 2)
        doc.add_paragraph(
            "Gradient x (input - training mean) attribution, out of fold. Maps are stable across fold models "
            "(Spearman 0.67-0.84) but agree only weakly with simple patient-control coherence differences, so "
            "the models use feature combinations. Schizophrenia leans on frontal-pole (Fp1-Fp2) delta coherence, "
            "where eye movements appear; part of that signal may be ocular, not cortical.")
        rows = []
        for f in ex:
            r = json.loads(f.read_text())
            rows.append((f"{r['task']} ({r['dataset']})", f"{r['fold_stability_spearman']:.2f}",
                         ", ".join(t["edge"] for t in r["top_connections"][:4])))
        table(doc, ["task", "fold stability", "top connections"], rows)
        for name in ("alzheimers_ds004504", "schizophrenia_aseeg"):
            png = RES / "explain" / f"{name}.png"
            if png.exists():
                figure(doc, png, f"Figure. Attribution map, {name.replace('_', ' ')}.", width=3.6)

    sh = _load("sharpened.json")
    if sh:
        doc.add_heading("5.12 Secondary: two hard tasks at sharper clinical contrasts", 2)
        doc.add_paragraph(
            "These ask easier questions than the main tasks and do not replace them. Cut-offs are published "
            "clinical bands fixed before running: UNM depression as BDI-II >= 20 (moderate/severe) vs controls "
            "(BDI < 7), dropping mild cases; Tel Aviv Parkinson's as MDS-UPDRS III >= 33 (moderate motor "
            "severity) vs controls. Same six-method ensemble, run once.")
        main = _load("ensemble_v2.json")
        rows = []
        for name, r in sh.items():
            base = main.get(f"{r['task']}:{r['dataset']}", {}).get("ensemble", {}).get("auc")
            rows.append((name.replace("_", " "), str(r["n"]), f"{base:.3f}" if base else "-", ci(r["ensemble"])))
        table(doc, ["sharpened task", "subjects", "main-task ensemble AUC", "sharpened ensemble AUC [95% CI]"], rows)


def limitations():
    return [
        "Within-dataset AUCs carry 95% CIs of roughly +/-0.1 (30-150 subjects per dataset); differences smaller "
        "than that between methods are trends, not established improvements.",
        "Mumtaz depression results may be optimistic: subject-disjoint splits on that dataset swing widely "
        "between folds in independent work (Zenodo 20630665).",
        "Semmelweis gives only 2 minutes of eyes-closed EEG per subject; Tel Aviv patients lost more windows "
        "to artefacts than controls (8.8% vs 3.9%), a mild confound.",
        "ASZED (Nigerian schizophrenia) was examined and excluded: its rest recordings are eyes-open, ~20 s per "
        "subject, and only 13 windows survived artefact rejection, with patients losing far more than controls.",
        "Final-phase results (5.6-5.12) were computed on a local RTX 2060 GPU; results there are from the same "
        "code and data as the earlier sections but newer library versions, so re-runs differ within CI.",
    ]


REPRODUCE = [
    "python fetch_neuro.py --datasets ds003478 ds007526      # UNM depression, Tel Aviv Parkinson's",
    "python raw_eeg.py features --datasets ... --out <dir>     # caches features + raw windows",
    "python rich_features.py; python baselines.py --features base|rich|both",
    "python riemann.py; python labram_probe.py; python ensemble.py   # final table (ensemble_v2.json)",
    "python pooled.py; python sharpened.py; python tigga.py --cache <ds004504.npz>",
    "python site_probe.py; python explain.py",
]
