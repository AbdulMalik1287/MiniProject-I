"""Team update PDF (plain-language project reset + full work history).

    python docs/make_team_update.py        -> docs/Team_Update_2026-11-13.pdf
Master-model numbers are read from results/pooled_deep.json; everything else is
transcribed from the results files named in each table.
"""
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "Team_Update_2026-11-13.pdf"

pdfmetrics.registerFont(TTFont("Arial", "C:/Windows/Fonts/arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", "C:/Windows/Fonts/arialbd.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Italic", "C:/Windows/Fonts/ariali.ttf"))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic",
                              boldItalic="Arial-Bold")

INK, MUTED, ACCENT, LINE = colors.HexColor("#1d2433"), colors.HexColor("#5b6475"), colors.HexColor("#2f5d8a"), colors.HexColor("#d5dae3")
GOOD, BAD, MID = colors.HexColor("#e3f1e6"), colors.HexColor("#f8e3e1"), colors.HexColor("#fbf1dc")
HEAD_BG, ZEBRA = colors.HexColor("#e9eef5"), colors.HexColor("#f7f9fc")

S = {
    "title": ParagraphStyle("t", fontName="Arial-Bold", fontSize=22, leading=27, textColor=INK, spaceAfter=4),
    "sub": ParagraphStyle("s", fontName="Arial", fontSize=11, leading=15, textColor=MUTED, spaceAfter=14),
    "h1": ParagraphStyle("h1", fontName="Arial-Bold", fontSize=15, leading=19, textColor=ACCENT, spaceBefore=10, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="Arial-Bold", fontSize=11.5, leading=15, textColor=INK, spaceBefore=8, spaceAfter=3),
    "p": ParagraphStyle("p", fontName="Arial", fontSize=9.6, leading=13.6, textColor=INK, spaceAfter=5),
    "small": ParagraphStyle("sm", fontName="Arial", fontSize=8.2, leading=11, textColor=MUTED, spaceAfter=4),
    "cell": ParagraphStyle("c", fontName="Arial", fontSize=8.4, leading=11, textColor=INK),
    "cellb": ParagraphStyle("cb", fontName="Arial-Bold", fontSize=8.4, leading=11, textColor=INK),
    "box": ParagraphStyle("bx", fontName="Arial", fontSize=9.6, leading=13.6, textColor=INK),
    "big": ParagraphStyle("bg", fontName="Arial-Bold", fontSize=18, leading=21, textColor=ACCENT, alignment=TA_CENTER),
    "bigl": ParagraphStyle("bl", fontName="Arial", fontSize=8, leading=10, textColor=MUTED, alignment=TA_CENTER),
}
W = A4[0] - 36 * mm


def P(t, s="p"):
    return Paragraph(t, S[s])


def bullets(items):
    return [P(f"•&nbsp;&nbsp;{t}") for t in items]


def table(rows, widths, shade=None, head=True):
    """rows: list of lists of str; shade: {row_index: color} for verdict rows."""
    data = [[P(c, "cellb" if (head and i == 0) else "cell") for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * W for w in widths], repeatRows=1 if head else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
          ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5)]
    if head:
        st.append(("BACKGROUND", (0, 0), (-1, 0), HEAD_BG))
    for i in range(1 if head else 0, len(rows)):
        if shade and i in shade:
            st.append(("BACKGROUND", (0, i), (-1, i), shade[i]))
        elif i % 2 == 0:
            st.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    t.setStyle(TableStyle(st))
    return t


def verdict_table(rows, widths):
    """Last column is the verdict: Worked / Partly / Did not."""
    col = {"Worked": GOOD, "Partly": MID, "Did not": BAD, "Done": GOOD, "Running": MID, "Next": MID, "Dropped": BAD}
    shade = {i: col[r[-1]] for i, r in enumerate(rows) if i and r[-1] in col}
    return table(rows, widths, shade)


def box(flow, bg=HEAD_BG):
    t = Table([[flow]], colWidths=[W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                           ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return t


def stats(items):
    cells = [[P(v, "big") for v, _ in items], [P(l, "bigl") for _, l in items]]
    t = Table(cells, colWidths=[W / len(items)] * len(items))
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.5, LINE), ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE),
                           ("BACKGROUND", (0, 0), (-1, -1), ZEBRA), ("TOPPADDING", (0, 0), (-1, -1), 6),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return t


def footer(c, doc):
    c.saveState()
    c.setFont("Arial", 7.5)
    c.setFillColor(MUTED)
    c.drawString(18 * mm, 10 * mm, "Brain Signal Processing - IIIT Nagpur CSE mini project - team update, 13 Nov 2026")
    c.drawRightString(A4[0] - 18 * mm, 10 * mm, f"page {doc.page}")
    c.restoreState()


def master_rows():
    r = json.loads((ROOT / "results" / "pooled_deep.json").read_text())
    names = {"lr_base": "Logistic regression", "gatv2": "GATv2 (graph)", "gt": "Graph Transformer",
             "gt_mil": "Attention-MIL", "gt_mil_dro": "Attention-MIL + GroupDRO", "labram_ft": "LaBraM fine-tuned",
             "ensemble_all6": "<b>Ensemble (all 6)</b>"}
    tasks = [("alzheimers", "AD vs control", ["ds004504", "brainlat"]),
             ("ftd", "FTD vs control", ["ds004504", "brainlat"]),
             ("alzheimers_vs_ftd", "AD vs FTD", ["ds004504", "brainlat"]),
             ("schizophrenia", "Schizophrenia", ["aseeg", "warsaw", "semmelweis"])]
    label = {"ds004504": "Thessaloniki", "brainlat": "BrainLat", "aseeg": "ASEEG", "warsaw": "Warsaw",
             "semmelweis": "Semmelweis"}
    out = []
    for t, tname, sites in tasks:
        rows = [["Model"] + [label[s] for s in sites] + ["All"]]
        for m in names:
            ps = r[f"{t}:{m}"]["per_site"]
            rows.append([names[m]] + [f"{ps[s]['auc']:.2f}" for s in sites + ["all"]])
        out.append((tname, rows))
    return out


def build():
    st = []
    # ---------------------------------------------------------------- cover / summary
    st += [P("Brain Signal Processing: where we are", "title"),
           P("Detecting brain and mental disorders from resting-state EEG with graph neural networks · "
             "team update and full work history · 13 November 2026 · supervisor Dr. Nileshchandra Pikle", "sub")]
    st.append(box(P("<b>In one paragraph.</b> We built and honestly tested about 30 models on 12 public EEG datasets. "
                    "Our best scores are strong for Alzheimer's (AUC 0.92) and depression (0.97 on one dataset), decent "
                    "for FTD (0.85) and schizophrenia (0.68-0.83 depending on hospital), and weak for telling Alzheimer's "
                    "apart from FTD (0.68-0.86). The main lesson: <b>a bigger or fancier model does not help</b> - simple "
                    "logistic regression ties our graph networks and transformers. What limits us is the data: small "
                    "datasets, each hospital leaving a strong 'fingerprint' in its recordings, noisy labels in one "
                    "dataset, and a recording mismatch in another. We also showed that several published high scores "
                    "come from flawed testing.", "box")))
    st.append(Spacer(1, 8))
    st.append(stats([("0.92", "Alzheimer's vs healthy<br/>(Thessaloniki)"), ("0.85", "FTD vs healthy<br/>(Thessaloniki)"),
                     ("0.97", "Depression vs healthy<br/>(Mumtaz)"), ("0.68-0.83", "Schizophrenia vs healthy<br/>(3 hospitals)"),
                     ("0.68 / 0.86", "Alzheimer's vs FTD<br/>(Thessaloniki / BrainLat)")]))
    st.append(P("Numbers are AUC (see 'How to read the numbers'). Each comes from people the model never saw during training.", "small"))

    st.append(P("How to read the numbers", "h1"))
    st.append(table([
        ["Term", "Plain meaning"],
        ["<b>AUC</b>", "Chance that the model ranks a random patient as 'more ill' than a random healthy person. "
                       "0.5 = coin flip, 0.7 = useful, 0.8 = good, 0.9+ = very good."],
        ["<b>Subject-wise testing</b>", "A person's recording is either all in training or all in testing, never split. "
                                        "Splitting one person's 10-second windows across both (common in papers) inflates scores, "
                                        "because the model just recognises the person."],
        ["<b>Leakage</b>", "Any way the test answers sneak into training or model choice, e.g. picking the best training "
                           "epoch by looking at test accuracy. Gives scores that do not hold up."],
        ["<b>Hospital effect</b>", "Each hospital's equipment and procedure leave a fingerprint. Our models could tell "
                                   "the hospital from healthy people alone with AUC 0.88-1.00 - easier than the disease."],
        ["<b>Ensemble</b>", "Several models vote (we average their rankings). Members are fixed before seeing results."],
        ["<b>Master model</b>", "One model per disease trained on people from all hospitals mixed together, tested on "
                                "held-out people from every hospital."],
        ["<b>95% CI</b>", "Range the true score probably lies in. With 30-100 people per dataset these ranges are wide "
                          "(often ±0.10), so small differences between models mean nothing."],
    ], [0.22, 0.78]))

    # ---------------------------------------------------------------- scope + data
    st.append(PageBreak())
    st.append(P("Current scope", "h1"))
    st.append(P("We narrowed to the four disorders with the best and most defensible data: <b>Alzheimer's (AD), "
                "frontotemporal dementia (FTD), depression and schizophrenia</b>. Parkinson's was explored "
                "(4 datasets, AUC 0.73-0.80) and parked to keep focus."))
    st.append(table([
        ["Disorder", "Datasets used", "People", "Notes"],
        ["AD, FTD", "Thessaloniki (OpenNeuro ds004504)", "36 AD, 23 FTD, 29 healthy", "Clean; one hospital"],
        ["", "BrainLat (Argentina + Chile, Synapse)", "AD, FTD, healthy (86 usable)",
         "Patients and healthy people were recorded into different file types - see caveats"],
        ["Depression", "Mumtaz (Malaysia, figshare)", "30 depressed, 28 healthy", "Clinical diagnosis; now our only depression set"],
        ["Schizophrenia", "ASEEG, Warsaw, Semmelweis", "101 / 28 / 71", "Three hospitals"],
    ], [0.16, 0.32, 0.24, 0.28]))
    st.append(P("Dropped datasets", "h2"))
    st.append(table([
        ["Dataset", "Why dropped"],
        ["UNM depression (ds003478)", "'Depressed' meant a questionnaire score ≥13. Of 46 such students only 11 had depression "
                                      "confirmed by clinical interview; 9 were interviewed and did not meet criteria. "
                                      "Label noise capped scores at ~0.6-0.7 whatever we did."],
        ["ASZED schizophrenia", "Eyes-open and only ~20 s per person (other sets are eyes-closed, minutes long)."],
        ["14 Chilean BrainLat controls", "Their signal files do not exist on Synapse (header files only)."],
        ["Parkinson's sets (4)", "Out of focus, kept in the repo."],
    ], [0.3, 0.7]))

    st.append(P("Shared processing pipeline", "h2"))
    st.append(P("Every recording goes through the same steps: 19 standard scalp electrodes → mains-noise filter → "
                "resample to 128 Hz → keep 1-45 Hz → average reference → cut into 10-second windows → throw away "
                "windows with extreme or flat signal → up to 30 clean windows per person. From each window we compute "
                "power in 6 frequency bands per electrode and how strongly each pair of electrodes moves together "
                "(coherence). These become the nodes and edges of the brain graph."))

    # ---------------------------------------------------------------- history
    st.append(P("Work history, in order", "h1"))
    hist = [
        ("1. Proposal and baseline",
         ["Rejected the suggested RL-based paper (IEEE 10977776) as a baseline - not a GNN-on-EEG method.",
          "Reproduced <b>EEG-GCNN</b> (released checkpoints, 478 held-out people) as our reference GNN."]),
        ("2. Literature plan",
         ["Reviewed 6 papers. Chose <b>MSBP-GNN</b> as the main model to rebuild, <b>Tigga et al.</b> as an example of "
          "flawed testing, and the GATv2→LSTM idea (spatio-temporal) as an extension."]),
        ("3. Data and first models",
         ["Built the raw-EEG pipeline and collected 12 public datasets.",
          "Trained GATv2 and GCN per disorder, rebuilt MSBP-GNN, built GATv2-LSTM. All landed in the same range; "
          "the LSTM added nothing."]),
        ("4. Cross-hospital attempt (paused)",
         ["Tried three standard fixes for hospital differences: ComBat, Euclidean alignment, domain-adversarial "
          "training (DANN). Transfer to an unseen hospital stayed near chance (e.g. schizophrenia).",
          "Site probe: hospital identifiable at AUC 0.88-1.00 from healthy people alone. Paused, then dropped."]),
        ("5. Pushing the scores up honestly",
         ["Added richer EEG markers (1/f slope, alpha peak, entropy, Hjorth), a Riemannian model, the pretrained "
          "LaBraM transformer, and more datasets.",
          "Built <b>Ensemble v2</b> (6 models, fixed in advance) - our headline within-dataset numbers.",
          "Target '0.85 everywhere' was not reachable honestly; instead we added clinically sharper secondary tasks "
          "(e.g. moderate Parkinson's 0.96)."]),
        ("6. Teammate plan 1 (rigorous AD/FTD experiments)",
         ["<b>3-class AD / FTD / healthy</b>: accuracy 0.65, macro-F1 0.62 - FTD is mostly confused with AD.",
          "<b>DT-GCN audit</b> (paper claims 82%): their code picks the best epoch on the test fold. Same model with "
          "honest selection: 0.52 accuracy; strict last-epoch: 0.38.",
          "<b>Graph ablation</b> (6 graph types × GCN/GraphSAGE/GATv2): a model with <b>no edges</b> scores the same "
          "as every graph (F1 0.48-0.55). Edges add nothing measurable.",
          "Re-ran Tigga et al.: edges inert; their leaky protocol 0.48 vs honest 0.44."]),
        ("7. Focus on top 4 and BrainLat",
         ["Narrowed to AD, FTD, depression, schizophrenia. Downloaded BrainLat (Synapse).",
          "Found that BrainLat patients and healthy people come from different file types (4× amplitude gap) - "
          "patient-vs-healthy scores of 1.00 were fake. Reviewed how others used BrainLat: careful studies get ~0.68-0.70.",
          "AD vs FTD on BrainLat is clean (same file type): 0.74 within BrainLat; 0.70 / 0.66 when trained on one "
          "dataset and tested on the other."]),
        ("8. Teammate plan 2 (transformers, master model)",
         ["Built the 'new' components from the teammate table: Graph Transformer, attention-MIL, GroupDRO, LaBraM "
          "fine-tuning - all trained as one <b>master model per disease</b> on all hospitals mixed.",
          "Result: Graph Transformer ties the simple models, LaBraM fine-tuning is mixed, attention-MIL and GroupDRO "
          "are worse. Weighting hospitals equally changed nothing. <b>AD vs FTD improved clearly</b> with the master model.",
          "Dropped leave-one-hospital-out and UNM."]),
        ("9. Preprocessing review (running)",
         ["Literature: ICA / ASR artifact removal and reference changes do not improve classification (one study on "
          "our own Thessaloniki data). Testing three cheap changes now: windows spread across the whole recording, "
          "1/f-corrected band power, and wPLI connectivity."]),
    ]
    for head, items in hist:
        st.append(KeepTogether([P(head, "h2")] + bullets(items)))

    # ---------------------------------------------------------------- what worked
    st.append(P("What worked and what did not", "h1"))
    st.append(verdict_table([
        ["What we tried", "Result", "Verdict"],
        ["Ensemble v2 (6 models voting)", "Best within-dataset scores", "Worked"],
        ["Logistic regression / Riemannian / rich markers", "Match or beat the graph networks", "Worked"],
        ["Master model per disease (all hospitals mixed)", "Big gain for AD vs FTD; small losses elsewhere", "Partly"],
        ["Honest audits of published models (DT-GCN, Tigga)", "Showed their scores rely on leaky testing", "Worked"],
        ["GATv2 / GCN / MSBP-GNN", "All similar, none beats simple models", "Partly"],
        ["Graph edges (6 graph types)", "No better than no edges", "Did not"],
        ["GATv2-LSTM (time sequence)", "No gain", "Did not"],
        ["Graph Transformer", "Ties the simple models", "Partly"],
        ["LaBraM pretrained transformer (frozen / fine-tuned)", "Useful ensemble member, not a winner", "Partly"],
        ["Attention-MIL, GroupDRO", "Worse on every task", "Did not"],
        ["ComBat, Euclidean alignment, DANN (hospital fixes)", "Unseen hospital still near chance", "Did not"],
        ["Hospital-balanced training", "No consistent change", "Did not"],
        ["Stricter depression cut-off (UNM)", "No improvement; dataset dropped", "Did not"],
        ["3-class AD / FTD / healthy", "65% accuracy; FTD confused with AD", "Partly"],
        ["Preprocessing v2", "Results pending", "Running"],
    ], [0.45, 0.4, 0.15]))

    st.append(P("Why the fancy models do not win", "h2"))
    st += bullets([
        "<b>Too few people.</b> 28-130 people per dataset. A transformer with thousands of parameters memorises "
        "individuals; a simple model cannot.",
        "<b>Hospital fingerprints are stronger than disease.</b> Any flexible model happily learns the hospital.",
        "<b>The information is mostly in band power</b> (how much slow vs fast activity). Graph structure and "
        "time order add little on top - the ablation proved it for edges.",
    ])

    # ---------------------------------------------------------------- results
    st.append(PageBreak())
    st.append(P("Best results per disorder", "h1"))
    st.append(table([
        ["Task", "Within one dataset (Ensemble v2)", "Master model, all hospitals (ensemble)"],
        ["AD vs healthy", "Thessaloniki <b>0.92</b>", "Thessaloniki 0.91 · BrainLat 0.74*"],
        ["FTD vs healthy", "Thessaloniki <b>0.85</b>", "Thessaloniki 0.81 · BrainLat 0.65*"],
        ["AD vs FTD", "Thessaloniki 0.58 · BrainLat 0.74", "<b>Thessaloniki 0.68 · BrainLat 0.86</b>"],
        ["Depression", "Mumtaz <b>0.97</b>", "- (single dataset now)"],
        ["Schizophrenia", "ASEEG 0.83 · Warsaw 0.70 · Semmelweis 0.68", "0.81 · 0.67 · 0.67"],
        ["3-class AD/FTD/healthy", "accuracy 0.65 · macro-F1 0.62", "not run yet"],
    ], [0.2, 0.4, 0.4]))
    st.append(P("* BrainLat patient-vs-healthy carries the file-type caveat. Within-dataset and master numbers use "
                "different splits, so compare them loosely.", "small"))

    st.append(P("Master model, every model (AUC per hospital)", "h2"))
    st.append(P("Results file: results/pooled_deep.json. All models used identical splits; differences under ~0.10 "
                "are within noise.", "small"))
    for tname, rows in master_rows():
        n = len(rows[0])
        st.append(KeepTogether([P(tname, "h2"), table(rows, [0.34] + [0.66 / (n - 1)] * (n - 1))]))

    # ---------------------------------------------------------------- teammate table
    st.append(P("Status of the teammate's architecture table", "h1"))
    st.append(verdict_table([
        ["Component", "Status", "Verdict"],
        ["GATv2, GCN, MSBP-GNN, GATv2+LSTM", "Built and benchmarked", "Done"],
        ["Riemannian, LaBraM, ComBat, Euclidean alignment, DANN", "Built and benchmarked", "Done"],
        ["Graph Transformer", "Built (master model) - ties simple models", "Done"],
        ["LaBraM deeper fine-tuning", "Built (last 2 blocks) - mixed", "Done"],
        ["Attention-based MIL", "Built - worse", "Done"],
        ["GroupDRO", "Built - worse", "Done"],
        ["Uncertainty-aware prediction", "Next step (calibration + 'unsure' output)", "Next"],
        ["Spectral / hierarchical temporal transformer, cross-attention", "Skipped: need far more data; LaBraM covers the idea", "Dropped"],
        ["Supervised contrastive learning", "Skipped for now: little to contrast with ~100 people", "Dropped"],
    ], [0.42, 0.43, 0.15]))

    # ---------------------------------------------------------------- caveats
    st.append(P("Honest caveats to keep in the report", "h1"))
    st += bullets([
        "<b>Depression is single-site</b> (Mumtaz, 58 people). 0.97 cannot be claimed to generalise. Option: request "
        "access to MODMA (clinically diagnosed depression, needs a signed request).",
        "<b>BrainLat patient-vs-healthy</b> is confounded by recording file type. We remove the amplitude gap, but other "
        "differences may remain. AD vs FTD on BrainLat is clean.",
        "<b>Small datasets</b> mean wide confidence intervals; many model differences are noise.",
        "<b>Generalising to a brand-new hospital is not solved</b> - we tried and documented it.",
    ])

    # ---------------------------------------------------------------- next
    st.append(P("What we do next", "h1"))
    st.append(table([
        ["#", "Task", "How", "Time"],
        ["1", "Finish preprocessing v2", "Running now. Adopt only if it helps consistently beyond the noise.", "~1 h"],
        ["2", "Freeze the final system", "Master model for AD, FTD, AD-vs-FTD, schizophrenia; within-dataset for "
                                         "depression. Members fixed - no more picking by results.", "~30 min"],
        ["3", "Calibrated probabilities + 'unsure'", "Per disease, fitted on training data only. Curves like "
                                                      "'answers 70% of people at X% accuracy'.", "1-2 h"],
        ["4", "Explainability", "Which electrodes and frequency bands drive each disease, checked for stability "
                                "across folds.", "~1 h"],
        ["5", "Update report, results page, model registry", "Add everything above.", "1-2 h"],
        ["6", "GitHub", "Push branch gcn-comparison and open a PR to master (after team sign-off).", "5 min"],
        ["7", "Optional: 3-class with BrainLat", "Same 3-class test on the master model.", "~30 min"],
    ], [0.05, 0.27, 0.56, 0.12]))
    st.append(P("Decisions for the team", "h2"))
    st += bullets([
        "Is single-site depression acceptable, or do we request MODMA?",
        "Agree the headline framing: an honest benchmark showing that simple models match GNNs/transformers on "
        "current public EEG, plus audits of leaky published results - rather than chasing one big number.",
    ])
    st.append(Spacer(1, 6))
    st.append(P("Code: github.com/AbdulMalik1287/MiniProject-I, branch gcn-comparison (latest commits local until "
                "pushed). Key files: pooled_deep.py (master models), ensemble.py, three_class.py, dtgcn.py, "
                "graph_ablation.py, prep_v2.py, docs/MODEL_REGISTRY.md.", "small"))

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm, title="Brain Signal Processing - team update",
                            author="Brain Signal Processing team")
    doc.build(st, onFirstPage=footer, onLaterPages=footer)
    print(OUT)


if __name__ == "__main__":
    build()
