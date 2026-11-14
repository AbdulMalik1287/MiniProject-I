"""Team update PDF (plain-language project reset + full work history).

    python -m eeg.reporting.team_update        -> docs/Team_Update_2026-11-13.pdf
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
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from eeg.paths import REPO as ROOT  # noqa: E402
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


FIN = None


def fin():
    global FIN
    if FIN is None:
        FIN = json.loads((ROOT / "results" / "final_system.json").read_text())
    return FIN


TASK_NAMES = [("alzheimers", "AD vs healthy"), ("ftd", "FTD vs healthy"), ("alzheimers_vs_ftd", "AD vs FTD"),
              ("mood", "Depression"), ("schizophrenia", "Schizophrenia")]
SITE = {"ds004504": "Thessaloniki", "brainlat": "BrainLat", "mumtaz": "Mumtaz", "aseeg": "ASEEG",
        "warsaw": "Warsaw", "semmelweis": "Semmelweis"}


def final_rows():
    rows = [["Task", "Hospital", "AUC", "95% CI", "Before (master v1)"]]
    old = json.loads((ROOT / "results" / "pooled_deep.json").read_text())
    for t, name in TASK_NAMES:
        ps = fin()[t]["ensemble"]["per_site"]
        for k, s in enumerate([x for x in ps if x != "all"]):
            mark = "*" if s == "brainlat" and t != "alzheimers_vs_ftd" else ("†" if s == "mumtaz" else "")
            prev = old.get(f"{t}:ensemble_all6", {}).get("per_site", {}).get(s)
            rows.append([name if k == 0 else "", SITE[s] + mark, f"<b>{ps[s]['auc']:.2f}</b>",
                         f"{ps[s]['ci'][0]:.2f}-{ps[s]['ci'][1]:.2f}",
                         f"{prev['auc']:.2f}" if prev and t != "mood" else "-"])
    return rows


def coverage_rows():
    rows = [["Task", "all", "90%", "80%", "70%", "60%", "50%"]]
    for t, name in TASK_NAMES:
        c = fin()[t]["ensemble"]["coverage"]
        rows.append([name] + [f"{c[k]['accuracy']:.0%}" for k in ("100%", "90%", "80%", "70%", "60%", "50%")])
    return rows


def member_rows():
    names = {"lr": "Logistic regression", "riemann_tn": "Riemannian (scale-free)", "gatv2": "GATv2",
             "gt": "Graph Transformer", "labram_ft": "LaBraM fine-tuned"}
    cols = [(t, s) for t, _ in TASK_NAMES for s in fin()[t]["ensemble"]["per_site"] if s != "all"]
    short = {"alzheimers": "AD", "ftd": "FTD", "alzheimers_vs_ftd": "AD/FTD", "mood": "Dep", "schizophrenia": "SZ"}
    rows = []
    for chunk in (cols[:6], cols[6:]):
        pad = [""] * (6 - len(chunk))
        rows.append(["<b>Model</b>"] + [f"<b>{short[t]} {SITE[s]}</b>" for t, s in chunk] + pad)
        for m, mn in names.items():
            rows.append([mn] + [f"{fin()[t]['per_site_' + m][s]['auc']:.2f}" for t, s in chunk] + pad)
        rows.append(["<b>Ensemble</b>"] + [f"<b>{fin()[t]['ensemble']['per_site'][s]['auc']:.2f}</b>" for t, s in chunk] + pad)
    return rows


def build():
    st = []
    # ---------------------------------------------------------------- cover / summary
    st += [P("Brain Signal Processing: where we are", "title"),
           P("Detecting brain and mental disorders from resting-state EEG with graph neural networks · "
             "team update and full work history · 13 November 2026 · supervisor Dr. Nileshchandra Pikle", "sub")]
    st.append(box(P("<b>In one paragraph.</b> We built and honestly tested about 30 models on 12 public EEG datasets "
                    "and froze a final system per disease: 5 complementary models voting, trained on all hospitals of "
                    "a disease together, with calibrated probabilities and an 'unsure' option. Final scores: Alzheimer's "
                    "0.94, FTD 0.89, depression 0.95-0.97 (one dataset), schizophrenia 0.67-0.84 depending on hospital, "
                    "Alzheimer's vs FTD 0.71-0.86. The main lesson: <b>a bigger or fancier model does not help</b> - "
                    "simple logistic regression ties our graph networks and transformers; better preprocessing and "
                    "combining models helped more. What limits us is the data: small datasets, hospital 'fingerprints', "
                    "and recording differences between patient and control groups in two datasets (found and handled). "
                    "We also showed that several published high scores come from flawed testing.", "box")))
    st.append(Spacer(1, 8))
    st.append(stats([("0.94", "Alzheimer's vs healthy<br/>(Thessaloniki)"), ("0.89", "FTD vs healthy<br/>(Thessaloniki)"),
                     ("0.95", "Depression vs healthy<br/>(Mumtaz, 1-25 Hz)"), ("0.67-0.84", "Schizophrenia vs healthy<br/>(3 hospitals)"),
                     ("0.71 / 0.86", "Alzheimer's vs FTD<br/>(Thessaloniki / BrainLat)")]))
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
                "windows with extreme or flat signal → skip the first minute when the recording is long enough → keep up to 60 clean windows spread across the whole recording. From each window we compute "
                "power in 6 frequency bands per electrode and how strongly each pair of electrodes moves together "
                "(coherence), plus band power with the background 1/f slope removed. These become the nodes and edges of the brain graph."))

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
        ("9. Preprocessing review",
         ["Literature: ICA / ASR artifact removal and reference changes do not improve classification (one study on "
          "our own Thessaloniki data).",
          "Tested three cheap changes. <b>Kept:</b> windows spread across the whole recording (median over windows) and "
          "1/f-corrected band power - small but consistent gains (e.g. Warsaw 0.72 → 0.82, Mumtaz 0.92 → 0.96). "
          "<b>Dropped:</b> wPLI connectivity (worse).",
          "Caught a second confound: Riemannian covariances kept BrainLat's amplitude gap (fake 0.99). Dividing each "
          "covariance by its trace fixed it (0.70, in line with careful prior work) and even helped Thessaloniki FTD."]),
        ("10. Final frozen system",
         ["5 members fixed before running: logistic regression, scale-free Riemannian, GATv2, Graph Transformer, "
          "fine-tuned LaBraM - all on scale-free inputs. Mean-rank ensemble, cross-fitted calibration, 'unsure' option.",
          "Beats or matches every earlier result (see 'Final system results')."]),
        ("11. Explainability and a last check",
         ["Head maps per disease from within-hospital effect sizes, cross-checked with the model's weights.",
          "Found that Mumtaz's healthy group has ~55× more 44 Hz power than its depressed group (a filter difference). "
          "Using only 1-25 Hz, depression still scores 0.95, so the result stands with a caveat."]),
    ]
    for head, items in hist:
        st.append(KeepTogether([P(head, "h2")] + bullets(items)))

    # ---------------------------------------------------------------- what worked
    st.append(CondPageBreak(90 * mm))
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
        ["Preprocessing v2: spread-out windows + 1/f-corrected power", "Small consistent gains", "Worked"],
        ["wPLI connectivity instead of coherence", "Worse", "Did not"],
        ["Scale-free (trace-normalised) Riemannian", "Removed BrainLat fake 0.99; helped FTD", "Worked"],
        ["Final frozen ensemble + calibration + 'unsure'", "Best numbers; 84-97% accuracy on confident half", "Worked"],
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
    st.append(P("Final system results", "h1"))
    st.append(P("One frozen system per disease (members fixed before running; master-model splits; every person "
                "tested by models that never saw them). AUC with 95% confidence interval. Results file: "
                "results/final_system.json.", "small"))
    st.append(table(final_rows(), [0.22, 0.18, 0.2, 0.2, 0.2]))
    st.append(P("* BrainLat patient-vs-healthy: scale-free inputs, but still carries the file-type caveat. "
                "† Mumtaz: 0.95 when restricted to 1-25 Hz (filter caveat).", "small"))
    st.append(P("Answer only when confident", "h2"))
    st.append(P("The system can say 'unsure' for the people it is least confident about. Accuracy on the people it "
                "does answer:", "p"))
    st.append(table(coverage_rows(), [0.28] + [0.12] * 6))
    st.append(P("Every member, per hospital", "h2"))
    st.append(table(member_rows(), [0.22, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13]))
    st.append(PageBreak())
    st.append(P("Earlier master model (before preprocessing v2), every model", "h2"))
    st.append(P("Results file: results/pooled_deep.json. All models used identical splits; differences under ~0.10 "
                "are within noise.", "small"))
    for tname, rows in master_rows():
        n = len(rows[0])
        st.append(KeepTogether([P(tname, "h2"), table(rows, [0.34] + [0.66 / (n - 1)] * (n - 1))]))

    # ---------------------------------------------------------------- teammate table
    st.append(PageBreak())
    st.append(P("What the system looks at", "h1"))
    st.append(P("Maps show, for each frequency band, where patients differ from healthy people (red = higher in "
                "patients, blue = lower), measured inside each hospital and then averaged. They are cross-checked "
                "against the logistic-regression member's weights: 'stability' = do the 5 training folds agree, "
                "'agreement' = does the model use what the data shows."))
    st += bullets([
        "<b>Alzheimer's:</b> more slow theta everywhere, less alpha/beta - the classic 'EEG slowing' of dementia. "
        "Same direction at both hospitals.",
        "<b>FTD:</b> less beta and alpha, strongest over temporal and frontal electrodes.",
        "<b>Alzheimer's vs FTD:</b> Alzheimer's shows more theta than FTD.",
        "<b>Schizophrenia:</b> right-temporal (T4) delta up, alpha down; weaker effects (d ≈ 0.5).",
        "<b>Depression (Mumtaz):</b> dominated by a 44 Hz filter difference between groups - see caveats; the "
        "1-25 Hz pattern alone still separates the groups.",
    ])
    for t, cap in (("alzheimers", "Alzheimer's vs healthy"), ("ftd", "FTD vs healthy"),
                   ("alzheimers_vs_ftd", "Alzheimer's vs FTD"), ("schizophrenia", "Schizophrenia vs healthy"),
                   ("mood", "Depression vs healthy (Mumtaz)")):
        st.append(KeepTogether([P(cap, "h2"), Image(str(ROOT / "results" / "explain_final" / f"{t}.png"),
                                                    width=W, height=W * 4.6 / 12)]))
    st.append(CondPageBreak(90 * mm))
    st.append(P("Status of the teammate's architecture table", "h1"))
    st.append(verdict_table([
        ["Component", "Status", "Verdict"],
        ["GATv2, GCN, MSBP-GNN, GATv2+LSTM", "Built and benchmarked", "Done"],
        ["Riemannian, LaBraM, ComBat, Euclidean alignment, DANN", "Built and benchmarked", "Done"],
        ["Graph Transformer", "Built (master model) - ties simple models", "Done"],
        ["LaBraM deeper fine-tuning", "Built (last 2 blocks) - mixed", "Done"],
        ["Attention-based MIL", "Built - worse", "Done"],
        ["GroupDRO", "Built - worse", "Done"],
        ["Uncertainty-aware prediction", "Built: calibrated probabilities + 'unsure' option", "Done"],
        ["Spectral / hierarchical temporal transformer, cross-attention", "Skipped: need far more data; LaBraM covers the idea", "Dropped"],
        ["Supervised contrastive learning", "Skipped for now: little to contrast with ~100 people", "Dropped"],
    ], [0.42, 0.43, 0.15]))

    # ---------------------------------------------------------------- caveats
    st.append(P("Honest caveats to keep in the report", "h1"))
    st += bullets([
        "<b>Depression is single-site</b> (Mumtaz, 58 people). 0.97 cannot be claimed to generalise. Option: request "
        "access to MODMA (clinically diagnosed depression, needs a signed request).",
        "<b>Mumtaz has a filter difference between groups</b> (controls ~55× more 44 Hz power). Restricted to 1-25 Hz "
        "the score is 0.95, so quote 0.95 with this caveat, not 0.97.",
        "<b>BrainLat patient-vs-healthy</b> is confounded by recording file type. We remove the amplitude gap, but other "
        "differences may remain. AD vs FTD on BrainLat is clean.",
        "<b>Small datasets</b> mean wide confidence intervals; many model differences are noise.",
        "<b>Generalising to a brand-new hospital is not solved</b> - we tried and documented it.",
    ])

    # ---------------------------------------------------------------- next
    st.append(P("What is left", "h1"))
    st.append(table([
        ["#", "Task", "Status"],
        ["1", "Experiments, final system, calibration, explainability", "Done"],
        ["2", "Report, results page, model registry", "Updated with these results"],
        ["3", "GitHub", "Merged into master; code reorganised into the eeg/ package"],
        ["4", "Optional: 3-class AD/FTD/healthy on the final system", "Not started (~30 min)"],
        ["5", "Optional: second depression dataset (MODMA)", "Needs a signed access request - team decision"],
    ], [0.05, 0.55, 0.4]))
    st.append(P("Decisions for the team", "h2"))
    st += bullets([
        "Is single-site depression acceptable, or do we request MODMA?",
        "Agree the headline framing: an honest benchmark showing that simple models match GNNs/transformers on "
        "current public EEG, plus audits of leaky published results - rather than chasing one big number.",
    ])
    st.append(Spacer(1, 6))
    st.append(P("Code: github.com/AbdulMalik1287/MiniProject-I (master). Start at the README; the final system is "
                "eeg/experiments/final_system.py, its explanation eeg/experiments/explain_final.py, and every model is "
                "listed in docs/MODEL_REGISTRY.md.", "small"))

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm, title="Brain Signal Processing - team update",
                            author="Brain Signal Processing team")
    doc.build(st, onFirstPage=footer, onLaterPages=footer)
    print(OUT)


if __name__ == "__main__":
    build()
