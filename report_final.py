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
    fs = _load("final_system.json")
    first = []
    if fs:
        g = lambda t, s: fs[t]["ensemble"]["per_site"][s]["auc"]  # noqa: E731
        first = [
            "Final frozen system (2026-11-13, focus on four disorders): five scale-free members (logistic regression, "
            "trace-normalised Riemannian, GATv2, Graph Transformer, fine-tuned LaBraM) trained per disease on all of "
            "its datasets, mean-rank ensemble, calibrated, with abstention. Subject-level AUC: Alzheimer's "
            f"{g('alzheimers', 'ds004504'):.2f}, FTD {g('ftd', 'ds004504'):.2f} (Thessaloniki); Alzheimer's vs FTD "
            f"{g('alzheimers_vs_ftd', 'ds004504'):.2f} / {g('alzheimers_vs_ftd', 'brainlat'):.2f} (Thessaloniki / "
            f"BrainLat); depression {g('mood', 'mumtaz'):.2f} (Mumtaz; 0.95 within 1-25 Hz); schizophrenia "
            f"{g('schizophrenia', 'aseeg'):.2f} / {g('schizophrenia', 'warsaw'):.2f} / {g('schizophrenia', 'semmelweis'):.2f} "
            "(ASEEG / Warsaw / Semmelweis). On its most confident half of subjects accuracy is 84-97% for "
            "Alzheimer's, depression and schizophrenia.",
            "Two recording confounds were found and handled: BrainLat patients and controls come from different file "
            "types (amplitude gap; scale-free inputs only), and Mumtaz controls carry ~55x more 44 Hz power than "
            "patients (filter difference; 1-25 Hz check).",
        ]
    return first + [
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

    add_sections_v2(doc, h)


SITE = {"ds004504": "Thessaloniki", "brainlat": "BrainLat", "mumtaz": "Mumtaz", "aseeg": "ASEEG",
        "warsaw": "Warsaw", "semmelweis": "Semmelweis"}
FINAL_TASKS = [("alzheimers", "Alzheimer's vs control"), ("ftd", "FTD vs control"),
               ("alzheimers_vs_ftd", "Alzheimer's vs FTD"), ("mood", "Depression vs control"),
               ("schizophrenia", "Schizophrenia vs control")]


def add_sections_v2(doc, h):
    """Sections 5.13-5.18 (2026-11-12/13): AD/FTD audits, BrainLat, master models, preprocessing v2, final system."""
    table, figure, bullets, ci = h["table"], h["figure"], h["bullets"], h["ci"]
    tc, dt, ga = _load("three_class.json"), _load("dtgcn.json"), _load("graph_ablation.json")
    if tc and dt and ga:
        doc.add_heading("5.13 Three-class AD/FTD/control, the DT-GCN audit and the graph ablation", 2)
        e = tc["methods"]["ensemble"]
        doc.add_paragraph(
            f"Three-class (88 Thessaloniki subjects, 5x5-fold subject-wise CV): the seven-member ensemble reaches "
            f"accuracy {e['accuracy'][0]:.3f}, macro-F1 {e['macro_f1'][0]:.3f}, macro one-vs-rest AUC "
            f"{e['macro_auc_ovr'][0]:.3f}; FTD is mostly confused with AD. DT-GCN (Liu & Meng 2026) reports "
            f"{dt['paper_reported']['accuracy']} accuracy; its released code selects the best epoch on the held-out "
            f"fold. Ported and re-run: paper protocol {dt['paper_protocol_dynamic']['accuracy_best_epoch'][0]:.3f}, "
            f"epoch chosen on an inner validation split {dt['inner_val_dynamic']['accuracy'][0]:.3f}, last epoch "
            f"{dt['strict_dynamic']['accuracy'][0]:.3f}.")
        graphs = ["none", "spatial", "pearson", "plv", "coherence", "hybrid"]
        table(doc, ["graph", "GCN F1", "GraphSAGE F1", "GATv2 F1"],
              [(g, *(f"{ga[f'{g}:{m}']['macro_f1'][0]:.3f}" for m in ("gcn", "sage", "gatv2"))) for g in graphs
               if f"{g}:gcn" in ga],
              note="Band-power-only nodes so connectivity reaches the model only through edges. 'none' = self-loops "
                   "only: it scores the same as every graph, so the edges add nothing measurable.")

    cs = _load("cross_site.json")
    if cs:
        doc.add_heading("5.14 BrainLat: a second dementia cohort and its recording confound", 2)
        doc.add_paragraph(
            "BrainLat (Argentina and Chile; Synapse) adds AD, bvFTD and controls; 14 Chilean controls have no signal "
            "file on Synapse. Its patients were exported as rs-HEP files and its controls as rs files, with ~4x "
            "lower amplitude in controls and no overlap: any amplitude-sensitive method separates them perfectly "
            "(within-BrainLat AUC 1.00, invalid). Published careful analyses that normalise each recording and use "
            "scale-free features reach ~0.68-0.70 for AD vs control. We therefore use only scale-free inputs "
            "(relative power, coherence, trace-normalised covariances, per-recording z-scored windows) on BrainLat "
            "and report patient-vs-control there with this caveat. AD vs FTD is unaffected (same file type).")
        k1, k2 = "alzheimers_vs_ftd:ds004504->brainlat:raw", "alzheimers_vs_ftd:brainlat->ds004504:raw"
        table(doc, ["AD vs FTD transfer", "LR", "Riemann", "LaBraM", "ensemble"],
              [(lab, *(f"{cs[k][m]['auc']:.3f}" for m in ("lr_base", "riemann", "labram", "ensemble")))
               for lab, k in (("Thessaloniki -> BrainLat", k1), ("BrainLat -> Thessaloniki", k2))],
              note="Train on every subject of one cohort, test on every subject of the other.")

    pdp = _load("pooled_deep.json")
    if pdp:
        doc.add_heading("5.15 Master models per disease and the transformer components", 2)
        doc.add_paragraph(
            "One model per disease trained on all of its datasets together (one StratifiedKFold(5) over subjects, "
            "stratified by site x label; AUC reported per site). Components from the team's architecture plan were "
            "built and compared on identical folds: Graph Transformer (TransformerConv), attention-based MIL over a "
            "subject's windows, GroupDRO over site x label groups, and LaBraM with its last two blocks fine-tuned. "
            "None beat the simple references: the Graph Transformer tied, LaBraM fine-tuning was mixed, and MIL / "
            "GroupDRO were worse on every task. Weighting sites equally (results/pooled_deep_bal.json) changed "
            "nothing consistently. Pooling clearly helped AD vs FTD.")
        names = [("lr_base", "LR"), ("gatv2", "GATv2"), ("gt", "Graph Transformer"), ("gt_mil", "MIL"),
                 ("gt_mil_dro", "MIL+DRO"), ("labram_ft", "LaBraM-FT"), ("ensemble_all6", "ensemble")]
        rows = []
        for t, tn in FINAL_TASKS:
            if f"{t}:lr_base" not in pdp or t == "mood":
                continue
            for s in pdp[f"{t}:lr_base"]["per_site"]:
                if s != "all":
                    rows.append((f"{tn} - {SITE[s]}", *(f"{pdp[f'{t}:{m}']['per_site'][s]['auc']:.3f}" for m, _ in names)))
        table(doc, ["task - site", *(n for _, n in names)], rows,
              note="Depression rows omitted: that run pooled UNM, which was later dropped (questionnaire labels; "
                   "11 of 46 'depressed' students had interview-confirmed MDD).")

    pv = _load("prep_v2.json")
    if pv:
        doc.add_heading("5.16 Preprocessing v2 and scale-free covariances", 2)
        doc.add_paragraph(
            "A literature review found that ICA/ASR artefact removal and re-referencing do not improve EEG decoding "
            "(including a study on ds004504). Three cheap changes were tested instead on the master-model folds: "
            "(1) skip the first 60 s when >= 30 windows remain and keep up to 60 clean windows spread over the "
            "recording, aggregating by median; (2) 1/f-corrected (periodic) band power; (3) wPLI. (1) and (2) "
            "gave small gains in the same direction across datasets and were adopted; wPLI was worse and dropped. "
            "Raw filter-bank covariances still carried BrainLat's amplitude gap (AD vs control 0.99, invalid); "
            "dividing each covariance by its trace removed it (0.70) and also helped Thessaloniki FTD.")
        cols = [("v1_lr_base", "LR v1"), ("v2_lr_base", "LR v2 windows"), ("v2_lr_per", "+ 1/f power"),
                ("v2_lr_wpli", "wPLI"), ("v1_riemann", "Riemann raw"), ("v2_riemann_tn", "Riemann scale-free")]
        rows = []
        for t, tn in FINAL_TASKS:
            if f"{t}:v1_lr_base" not in pv:
                continue
            for s in pv[f"{t}:v1_lr_base"]["per_site"]:
                if s != "all":
                    rows.append((f"{tn} - {SITE[s]}", *(f"{pv[f'{t}:{c}']['per_site'][s]['auc']:.3f}"
                                                         if f"{t}:{c}" in pv else "-" for c, _ in cols)))
        table(doc, ["task - site", *(n for _, n in cols)], rows,
              note="Raw Riemann on BrainLat patient-vs-control is invalid (amplitude confound) and shown only to "
                   "document it.")

    fs = _load("final_system.json")
    if fs:
        doc.add_heading("5.17 Final system", 2)
        doc.add_paragraph(
            "Members fixed before running, all on scale-free v2 inputs: logistic regression (relative + 1/f-"
            "corrected power + coherence), trace-normalised filter-bank Riemannian, GATv2, Graph Transformer and "
            "LaBraM fine-tuned. Ensemble = mean of member ranks. Probabilities come from cross-fitted Platt "
            "scaling (each subject's calibrator is fitted on the other folds), and the system may abstain on its "
            "least confident subjects. Master-model folds for every task; depression uses Mumtaz only.")
        rows = []
        for t, tn in FINAL_TASKS:
            e = fs[t]["ensemble"]
            for s, r in e["per_site"].items():
                if s != "all":
                    rows.append((f"{tn} - {SITE[s]}", ci(r), *(f"{fs[t]['per_site_' + m][s]['auc']:.3f}"
                                                               for m in ("lr", "riemann_tn", "gatv2", "gt", "labram_ft"))))
        table(doc, ["task - site", "ensemble AUC [95% CI]", "LR", "Riemann (tn)", "GATv2", "Graph Tr.", "LaBraM-FT"],
              rows, note="BrainLat patient-vs-control carries the rs-HEP/rs caveat (5.14); Mumtaz carries the filter "
                         "caveat (5.18).")
        table(doc, ["task", "Brier", "ECE", *(f"acc @ {c}" for c in ("100%", "90%", "80%", "70%", "60%", "50%"))],
              [(tn, f"{fs[t]['ensemble']['brier']:.3f}", f"{fs[t]['ensemble']['ece']:.3f}",
                *(f"{fs[t]['ensemble']['coverage'][c]['accuracy']:.2f}" for c in ("100%", "90%", "80%", "70%", "60%", "50%")))
               for t, tn in FINAL_TASKS],
              note="Coverage = share of subjects the system answers (most confident first); threshold 0.5.")

    exd = RES / "explain_final"
    mc = _load("mumtaz_check.json")
    if exd.exists():
        doc.add_heading("5.18 What the final system relies on", 2)
        doc.add_paragraph(
            "Within-site Cohen's d of relative and 1/f-corrected band power per electrode, averaged over sites, "
            "cross-checked against the logistic-regression member's fold coefficients. Alzheimer's: theta up and "
            "alpha/beta down everywhere (EEG slowing), same sign at both sites. FTD: alpha and beta down, temporal "
            "and frontal. AD vs FTD: more theta in AD. Schizophrenia: right-temporal delta up, alpha down (d ~0.5).")
        rows = []
        for t, tn in FINAL_TASKS:
            f = exd / f"{t}.json"
            if f.exists():
                r = json.loads(f.read_text())
                rows.append((tn, f"{r['fold_stability_spearman']:.2f}", f"{r['model_vs_effect_spearman']:.2f}",
                             "; ".join(f"{x['feature']} ({x['cohen_d']:+.2f})" for x in r["top_effects"][:3])))
        table(doc, ["task", "fold stability", "model vs effect", "largest effects (Cohen's d)"], rows)
        for t in ("alzheimers", "mood"):
            if (exd / f"{t}.png").exists():
                figure(doc, exd / f"{t}.png", f"Figure. Within-site effect sizes, {dict(FINAL_TASKS)[t]}.", width=6.3)
        if mc:
            doc.add_paragraph(
                f"Depression check: Mumtaz's largest effect is 1/f-corrected gamma (d ~ -3), implausibly large. Mean "
                f"log10 power at 44 Hz is {mc['psd_log10']['control']['44Hz']:.2f} in controls vs "
                f"{mc['psd_log10']['mdd']['44Hz']:.2f} in MDD (35 Hz: {mc['psd_log10']['control']['35Hz']:.2f} vs "
                f"{mc['psd_log10']['mdd']['35Hz']:.2f}), i.e. a low-pass filter applied to one group. Gamma alone "
                f"gives AUC {mc['auc_gamma_only']:.3f}; relative power computed only within 1-25 Hz still gives "
                f"{mc['auc_relative_power_1_25Hz']:.3f}, so the depression result stands at ~0.95 with this caveat.")


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
        "Mumtaz controls and patients were filtered differently above ~40 Hz (5.18); the 1-25 Hz-only score (0.95) "
        "is the defensible depression number, and depression rests on this single dataset since UNM was dropped "
        "for questionnaire-based labels.",
        "BrainLat patient-vs-control remains confounded by export file type even with scale-free inputs; only AD vs "
        "FTD on BrainLat is clean.",
        "Final-phase results (5.6-5.18) were computed on a local RTX 2060 GPU; results there are from the same "
        "code and data as the earlier sections but newer library versions, so re-runs differ within CI.",
    ]


REPRODUCE = [
    "python fetch_neuro.py --datasets ds003478 ds007526      # UNM depression, Tel Aviv Parkinson's",
    "python raw_eeg.py features --datasets ... --out <dir>     # caches features + raw windows",
    "python rich_features.py; python baselines.py --features base|rich|both",
    "python riemann.py; python labram_probe.py; python ensemble.py   # final table (ensemble_v2.json)",
    "python pooled.py; python sharpened.py; python tigga.py --cache <ds004504.npz>",
    "python site_probe.py; python explain.py",
    "python three_class.py; python dtgcn.py; python graph_ablation.py; python cross_site.py",
    "python pooled_deep.py [--balance]                     # master models + transformer components",
    "python prep_v2.py extract; python prep_v2.py eval     # preprocessing v2",
    "python final_system.py; python explain_final.py; python mumtaz_check.py",
]
