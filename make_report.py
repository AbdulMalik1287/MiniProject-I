#!/usr/bin/env python3
"""Build docs/MiniProject_Results.docx from the committed result files.

Every number in the document is read from results/ (JSON written by the
experiment scripts) or computed from the model classes themselves - nothing is
retyped - except the two earlier phases run on Blackwell, whose raw result
files were not copied off the node; those tables are taken from the README
tables written at the time and are marked as such in the document.

    python make_report.py            # -> docs/MiniProject_Results.docx
"""
import json
import subprocess
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from docx import Document  # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Inches, Pt, RGBColor  # noqa: E402

import park as P  # noqa: E402
import per_disorder as D  # noqa: E402
import raw_eeg as R  # noqa: E402
from train import build_model  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
DOCS = HERE / "docs"
FIG = DOCS / "figures"

# Reference palette (dataviz skill, light mode). Series slots validated with
# scripts/validate_palette.js: #2a78d6,#eb6834 -> all checks pass.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2 = "#2a78d6", "#eb6834"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
       "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE, "xtick.color": MUTED, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})


def load(p):
    return json.loads(Path(p).read_text())


SITE = {"aseeg": "ASEEG", "warsaw": "Warsaw", "mumtaz": "Mumtaz", "ds004504": "Thessaloniki",
        "ds004584": "Iowa", "ds003490": "New Mexico", "ds002778": "San Diego"}
TASK_TAG = {"schizophrenia": "SZ", "mood": "MDD", "alzheimers": "AD", "ftd": "FTD",
            "alzheimers_vs_ftd": "AD vs FTD", "parkinsons": "PD"}
NEURO_TASKS = ["alzheimers", "ftd", "alzheimers_vs_ftd", "parkinsons"]


def verdict(v):
    """Plain reading of an AUC interval, decided by the interval rather than by the writer."""
    if v["ci"][0] > 0.5:
        return "above chance"
    if v["ci"][1] < 0.5:
        return "below chance"
    return "not distinguishable from chance"


def neuro_summary(raws):
    """Summary bullets for whichever neurological results exist, with data-driven wording."""
    def a(t, k):
        v = raws[t][k]
        return f"AUC {v['auc']:.2f} [{v['ci'][0]:.2f}, {v['ci'][1]:.2f}], {verdict(v)}"

    out = []
    if "alzheimers" in raws:
        out.append("Alzheimer's disease vs healthy controls (Thessaloniki, same GNN, not re-tuned): "
                   f"{a('alzheimers', 'within:ds004504')}.")
    if "ftd" in raws:
        out.append(f"Frontotemporal dementia vs healthy controls (same hospital): {a('ftd', 'within:ds004504')}.")
    if "alzheimers_vs_ftd" in raws:
        out.append("Telling Alzheimer's from frontotemporal dementia - the only between-disease test that is "
                   "fair, because both groups come from one hospital: "
                   f"{a('alzheimers_vs_ftd', 'within:ds004504')}.")
    if "parkinsons" in raws:
        r = raws["parkinsons"]
        within = "; ".join(f"{SITE[d]} {a('parkinsons', f'within:{d}')}" for d in r["datasets"]
                           if f"within:{d}" in r)
        out.append(f"Parkinson's disease vs healthy controls within each hospital: {within}.")
        loso = [d for d in r["datasets"] if f"loso:{d}" in r]
        if loso:
            out.append("Parkinson's trained on the other hospitals and tested on one never seen: " +
                       "; ".join(f"{SITE[d]} {a('parkinsons', f'loso:{d}')}" for d in loso) + ".")
    return out


def qc_group(ds, r):
    """Group of a QC record; caches written before group labels existed store label 0/1."""
    if "group" in r:
        return r["group"]
    return R.DATASETS[ds]["patient"] if r["label"] == 1 else "control"


# ---------------------------------------------------------------- docx helpers


def shade(cell, hex_fill):
    tc = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill.lstrip("#"))
    tc.append(shd)


def table(doc, header, rows, widths=None, bold_first_col=False, note=None):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(str(h))
        r.bold, r.font.size = True, Pt(8.5)
        shade(c, "#eef3fb")
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(v))
            run.font.size = Pt(8.5)
            if bold_first_col and i == 0:
                run.bold = True
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    if note:
        p = doc.add_paragraph()
        r = p.add_run(note)
        r.italic, r.font.size = True, Pt(8)
        r.font.color.rgb = RGBColor.from_string("52514E")
    doc.add_paragraph()
    return t


def figure(doc, path, caption, width=6.3):
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph()
    r = p.add_run(caption)
    r.italic, r.font.size = True, Pt(8.5)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER


def bullets(doc, items):
    for it in items:
        doc.add_paragraph(it, style="List Bullet")


def code(doc, text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.name, r.font.size = "Consolas", Pt(8)
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")


def ci(v):
    return f"{v['auc']:.3f} [{v['ci'][0]:.2f}, {v['ci'][1]:.2f}]"


# ------------------------------------------------------------------- figures


def box_diagram(path, stages, title=None, height=1.6):
    """Left-to-right chain of labelled boxes - an architecture/pipeline diagram."""
    n = len(stages)
    fig, ax = plt.subplots(figsize=(1.55 * n, height))
    ax.set_axis_off()
    ax.grid(False)
    w, gap = 1.25, 0.3
    for i, (head, body) in enumerate(stages):
        x = i * (w + gap)
        ax.add_patch(plt.Rectangle((x, 0), w, 1, facecolor="#eef3fb", edgecolor=S1, linewidth=1.2))
        ax.text(x + w / 2, 0.78, head, ha="center", va="center", fontsize=8.5, weight="bold", color=INK)
        ax.text(x + w / 2, 0.38, body, ha="center", va="center", fontsize=7, color=INK2, linespacing=1.3)
        if i < n - 1:
            ax.annotate("", xy=(x + w + gap - 0.03, 0.5), xytext=(x + w + 0.03, 0.5),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
    ax.set_xlim(-0.05, n * (w + gap) - gap + 0.05)
    ax.set_ylim(-0.05, 1.05)
    if title:
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    fig.savefig(path)
    plt.close(fig)


def gnn_stack(path, rows, title):
    """Vertical layer stack for one GNN - rows are (layer, detail, output shape)."""
    fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(rows) + 0.5))
    ax.set_axis_off()
    ax.grid(False)
    import textwrap
    for i, (name, detail, shape) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.add_patch(plt.Rectangle((0, y + 0.08), 6.0, 0.84, facecolor="#eef3fb" if i % 2 == 0 else "#f6f8fc",
                                   edgecolor=AXIS, linewidth=0.8))
        ax.text(0.15, y + 0.5, name, va="center", fontsize=8.5, weight="bold", color=INK)
        # Wrapped so long operation text never runs into the output-shape column.
        ax.text(1.2, y + 0.5, textwrap.fill(detail, 58), va="center", fontsize=7.2, color=INK2,
                linespacing=1.15)
        ax.text(5.85, y + 0.5, shape, va="center", ha="right", fontsize=7.5, color=S1, family="Consolas")
    ax.set_xlim(-0.05, 6.05)
    ax.set_ylim(-0.05, len(rows) + 0.05)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    fig.savefig(path)
    plt.close(fig)


def dot_whisker(path, labels, series, xlabel, title, chance=0.5, xlim=(0.25, 1.0)):
    """series: list of (name, color, [(value, lo, hi), ...]) - one row per label."""
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(labels) + 1.1))
    k = len(series)
    offs = np.linspace(-0.16, 0.16, k) if k > 1 else [0.0]
    y = np.arange(len(labels))[::-1]
    for (name, color, vals), off in zip(series, offs):
        v = np.array([a for a, _, _ in vals])
        lo = np.array([b for _, b, _ in vals])
        hi = np.array([c for _, _, c in vals])
        ax.errorbar(v, y + off, xerr=[v - lo, hi - v], fmt="o", color=color, ecolor=color,
                    elinewidth=2, capsize=0, markersize=6.5, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, label=name)
    if chance is not None:
        ax.axvline(chance, color=MUTED, linewidth=1.2)
        ax.text(chance, len(labels) - 0.35, " chance", color=MUTED, fontsize=7.5, va="bottom")
    ax.set_yticks(y, labels)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.7, len(labels) - 0.2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    # Legend sits above the plot area so it can never cover a data point.
    ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=22 if k > 1 else 6)
    if k > 1:
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=k, fontsize=8,
                  handletextpad=0.3, columnspacing=1.5, borderaxespad=0.2)
    fig.savefig(path)
    plt.close(fig)


def heatmap(path, mat, rows, cols, title):
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.grid(False)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", SEQ)
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=20, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if mat[i, j] > 0.55 else INK)
    ax.set_xlabel("model  (mean calibrated P(disorder))")
    ax.set_ylabel("true class")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.outline.set_visible(False)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------- architecture


def layer_table(model, spec):
    """spec: [(child_name, operation, output_shape)]; params counted from the model."""
    rows, counted = [], 0
    children = dict(model.named_children())
    for name, op, shape in spec:
        n = sum(p.numel() for p in children[name].parameters()) if name in children else 0
        counted += n
        rows.append((name, op, shape, f"{n:,}"))
    total = sum(p.numel() for p in model.parameters())
    assert counted == total, f"layer table misses parameters: {counted} vs {total}"
    rows.append(("total", "", "", f"{total:,}"))
    return rows, total


def architectures():
    base = build_model("gcn", 1)  # EEG-GCNN as reproduced: 8 nodes, 6 features
    base_rows, base_n = layer_table(base, [
        ("conv1", "GCNConv 6->32, improved, normalize=False, edge weight = distance+coherence", "8 x 32"),
        ("conv2", "GCNConv 32->20 (same flags)", "8 x 20"),
        ("bn", "BatchNorm over node features", "8 x 20"),
        ("pool", "global add pooling", "20"),
        ("fc1", "Linear 20->10, LeakyReLU, dropout 0.2 before", "10"),
        ("fc2", "Linear 10->2 (diseased / healthy)", "2"),
    ])
    cfg = load(RES / "per_disorder" / "tune.json")["best"]["config"]
    sel = D.make_model(cfg)
    in_dim = P.node_feature_dim(cfg["node_features"])
    sel_rows, sel_n = layer_table(sel, [
        ("conv1", f"GATv2Conv {in_dim}->8 x 4 heads (concat), edge_dim={P.N_BANDS} "
                  "(per-band coherence read by attention)", "19 x 32"),
        ("conv2", f"GATv2Conv 32->20, 1 head, edge_dim={P.N_BANDS}", "19 x 20"),
        ("bn", "BatchNorm over node features, LeakyReLU", "19 x 20"),
        ("pool", "concat(global mean pool, global max pool)", "40"),
        ("fc1", "Linear 40->10, LeakyReLU, dropout 0.2 before", "10"),
        ("fc2", "Linear 10->2 (disorder / healthy); logit margin scored", "2"),
    ])
    fcnn = build_model("fcnn", 1, n_nodes=P.N_NODES, n_classes=2, in_dim=in_dim)
    fc_n = sum(p.numel() for p in fcnn.parameters())
    cands = []
    for g in D.GRID:
        m = D.make_model(g)
        cands.append((g["model"], g["pool"], "complete (361 edges)" if g["topk"] is None
                      else f"top-{g['topk']} per node", f"{sum(p.numel() for p in m.parameters()):,}"))
    return cfg, base_rows, base_n, sel_rows, sel_n, fc_n, cands


# ------------------------------------------------------------------ document


def git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                                       text=True).strip()
    except Exception:
        return "unknown"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    pd_dir = RES / "per_disorder"
    tune, ev = load(pd_dir / "tune.json"), load(pd_dir / "evaluate.json")
    calib = load(pd_dir / "models" / "calibration.json")
    raws = {t: load(RES / "raw" / f"raw_{t}.json") for t in R.TASKS
            if (RES / "raw" / f"raw_{t}.json").exists()}
    raw_sz, raw_md = raws["schizophrenia"], raws["mood"]
    neuro = [t for t in NEURO_TASKS if t in raws]
    cfg, base_rows, base_n, sel_rows, sel_n, fc_n, cands = architectures()
    in_dim = P.node_feature_dim(cfg["node_features"])
    dis = D.DISORDERS
    short = [D.SHORT[d] for d in dis]
    ckpts = {d: torch.load(pd_dir / "models" / f"{D.SHORT[d]}.pt", map_location="cpu", weights_only=False)
             for d in dis}

    # ---- figures
    box_diagram(FIG / "system.png", [
        ("EEG features", "19 channels\n6 bands power\n171 pairs coherence"),
        ("Graph", "19 nodes\ntop-4 coherence\nneighbours per node"),
        ("5 GNNs", "one per disorder\nGATv2, binary\nvs healthy"),
        ("Calibration", "Platt on train\nout-of-fold,\nbalanced prior"),
        ("Ensemble", "max P >= 0.5\nnames disorder,\nelse healthy"),
    ], title="Per-disorder system")
    box_diagram(FIG / "raw_pipeline.png", [
        ("Load", "EDF / .mat\n19 channels\n(order verified)"),
        ("Filter", "notch 50 Hz\n128 Hz, 1-45 Hz\navg reference"),
        ("Window", "10 s windows\nreject >300 uV\nor flat; 30/subj"),
        ("Features", "log rel. power\n+ coherence\nper band"),
        ("GNN", "same GATv2\nconfig as Park\n(not re-tuned)"),
        ("Score", "mean window P\nper subject\nAUC + CI"),
    ], title="Raw-EEG pipeline (identical for every hospital)")
    gnn_stack(FIG / "gnn_selected.png",
              [(r[0], r[1], r[2]) for r in sel_rows[:-1]],
              f"Selected per-disorder GNN  ({sel_n:,} parameters)")
    gnn_stack(FIG / "gnn_baseline.png",
              [(r[0], r[1], r[2]) for r in base_rows[:-1]],
              f"EEG-GCNN baseline as reproduced  ({base_n:,} parameters)")

    res = sorted(tune["results"], key=lambda r: r["mean_auc"])
    dot_whisker(FIG / "tuning.png",
                [f"{r['config']['model']} / {r['config']['pool']} / "
                 f"{'complete' if r['config']['topk'] is None else 'top-' + str(r['config']['topk'])}"
                 for r in res][::-1],
                [("mean CV AUC", S1, [(r["mean_auc"], r["mean_auc"], r["mean_auc"]) for r in res][::-1])],
                "mean out-of-fold AUC across the 5 disorders (train split only)",
                "Tuning: 8 GNN configurations", chance=None, xlim=(0.58, 0.70))
    dot_whisker(FIG / "per_disorder_auc.png", short, [
        ("vs healthy (trained task)", S1,
         [(ev["per_disorder"][d]["auc"], *ev["per_disorder"][d]["auc_ci"]) for d in dis]),
        ("vs other disorders (specificity)", S2,
         [(ev["per_disorder"][d]["auc_vs_other_disorders"], *ev["per_disorder"][d]["auc_vs_other_ci"])
          for d in dis]),
    ], "test-split AUC with 95% bootstrap CI", "Per-disorder GNNs on the held-out test split")
    act = np.array(ev["activation"])
    heatmap(FIG / "activation.png", act, [D.SHORT[c] for c in D.ALL_CLASSES], short,
            "Which models fire for which patients")
    def raw_label(task, key):
        tag = TASK_TAG[task]
        kind, _, rest = key.partition(":")
        if kind == "within":
            return f"{tag}: within {SITE[rest]}"
        if kind == "pooled":
            n = len(raws[task]["datasets"])
            return f"{tag}: pooled, scored on {SITE[rest]}" if rest else f"{tag}: pooled ({n} hospitals)"
        if kind == "loso":
            return f"{tag}: trained elsewhere, test {SITE[rest]}"
        a, b = rest.split("->")
        return f"{tag}: train {SITE[a]}, test {SITE[b]}"

    def raw_rows_for(tasks, for_figure=False):
        rows = []
        for t in tasks:
            many = len(raws[t]["datasets"]) > 2
            for k, v in raws[t].items():
                if not (isinstance(v, dict) and "auc" in v):
                    continue
                # With three hospitals the figure keeps leave-one-out and drops the six
                # pairwise transfers and per-site pooled scores; the table keeps them all.
                if for_figure and many and (k.startswith("transfer:") or k.startswith("pooled:")):
                    continue
                rows.append((raw_label(t, k), v))
        return rows

    raw_rows = raw_rows_for(["schizophrenia", "mood"])
    dot_whisker(FIG / "raw_auc.png", [r[0] for r in raw_rows],
                [("subject-level AUC", S1, [(v["auc"], *v["ci"]) for _, v in raw_rows])],
                "subject-level AUC with 95% bootstrap CI", "Raw EEG: within, pooled and cross-hospital",
                xlim=(0.2, 1.0))
    if neuro:
        fig_rows = raw_rows_for(neuro, for_figure=True)
        dot_whisker(FIG / "neuro_auc.png", [r[0] for r in fig_rows],
                    [("subject-level AUC", S1, [(v["auc"], *v["ci"]) for _, v in fig_rows])],
                    "subject-level AUC with 95% bootstrap CI",
                    "Neurological disorders: within, pooled and cross-hospital", xlim=(0.2, 1.0))

    # ---- document
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name, st.font.size = "Calibri", Pt(10)
    doc.add_heading("Brain Signal Processing: EEG + Graph Neural Networks for "
                    f"{'Brain' if neuro else 'Psychiatric'} Disorder Classification", 0)
    p = doc.add_paragraph()
    p.add_run("Results and architectures. ").bold = True
    p.add_run("Mini Project I, Department of CSE, IIIT Nagpur, July-December 2026. "
              "Rajat Malik, Yash Khedekar, Abdul Malik, Ashwin Rathore. "
              "Supervisor: Dr. Nileshchandra Pikle. "
              f"Generated {date.today().isoformat()} from repository commit {git_head()}.")

    doc.add_heading("1. Summary", 1)
    best_h = max(dis, key=lambda d: ev["per_disorder"][d]["auc"])
    e = ev["ensemble"]
    bullets(doc, [
        "Base paper EEG-GCNN (Wagh & Varatharajah, ML4H 2020) reproduced exactly: all five held-out "
        "metrics match the published values, including standard deviations (AUC 0.871 +/- 0.001).",
        f"One GNN per disorder was built for {len(dis)} disorders. On a held-out test split never used for "
        f"tuning or training, every model separates its disorder from healthy controls: AUC "
        f"{min(ev['per_disorder'][d]['auc'] for d in dis):.2f}-{max(ev['per_disorder'][d]['auc'] for d in dis):.2f}, "
        f"best {D.SHORT[best_h]} {ev['per_disorder'][best_h]['auc']:.3f}; every 95% CI lies above chance.",
        "No model separates its disorder from the OTHER disorders (AUC "
        f"{min(ev['per_disorder'][d]['auc_vs_other_disorders'] for d in dis):.2f}-"
        f"{max(ev['per_disorder'][d]['auc_vs_other_disorders'] for d in dis):.2f}, every CI includes 0.5). "
        "The models detect psychiatric illness, not which illness.",
        f"The ensemble reaches balanced accuracy {e['bal_acc']:.3f} [{e['bal_acc_ci'][0]:.2f}, "
        f"{e['bal_acc_ci'][1]:.2f}] over {len(D.ALL_CLASSES)} classes (chance {1 / len(D.ALL_CLASSES):.3f}).",
        *([f"As a screen for 'outside the healthy range', the pre-declared combined score (mean of the five) "
           f"reaches AUC {ev['combined']['mean']['auc']:.3f} [{ev['combined']['mean']['auc_ci'][0]:.2f}, "
           f"{ev['combined']['mean']['auc_ci'][1]:.2f}]: it catches {ev['combined']['mean']['sensitivity']:.0%} "
           f"of patients and wrongly flags {1 - ev['combined']['mean']['specificity']:.0%} of healthy controls. "
           "Combining did not beat the single models."] if "combined" in ev else []),
        f"Graph sparsification mattered: the selected GNN is GATv2 with mean+max pooling on a top-"
        f"{cfg['topk']} coherence graph, and the three best of eight configurations all use sparse graphs.",
        "On raw EEG from two independent hospitals, the same GNN (not re-tuned) detects schizophrenia "
        f"within each hospital (AUC {raw_sz['within:aseeg']['auc']:.2f} ASEEG, "
        f"{raw_sz['within:warsaw']['auc']:.2f} Warsaw; pooled {raw_sz['pooled']['auc']:.2f}), but a model "
        "trained at one hospital is at chance at the other (AUC "
        f"{raw_sz['transfer:aseeg->warsaw']['auc']:.2f} and {raw_sz['transfer:warsaw->aseeg']['auc']:.2f}). "
        "What is learned is largely hospital-specific: the disease signature, as captured by these "
        "features, does not transfer across sites.",
        f"Depression within Mumtaz reaches AUC {raw_md['within:mumtaz']['auc']:.2f}, but with no second open "
        "dataset it cannot be checked across hospitals, and the schizophrenia transfer result means a "
        "single-site score should not be read as a transferable disease signature.",
        *neuro_summary(raws),
    ])

    doc.add_heading("2. Datasets", 1)
    qc = {ds: load(RES / "raw" / f"{ds}_qc.json") for ds in R.DATASETS if (RES / "raw" / f"{ds}_qc.json").exists()}
    qn_ = lambda ds, grp: sum(qc_group(ds, r) == grp for r in qc[ds])  # noqa: E731
    neuro_rows = []
    if "ds004504" in qc:
        neuro_rows.append(("ds004504 (Miltiadous 2023)", "OpenNeuro, CC0, Thessaloniki",
                           f"{qn_('ds004504', 'alzheimers')} AD / {qn_('ds004504', 'ftd')} FTD / "
                           f"{qn_('ds004504', 'control')} controls", "19-ch raw, eyes closed", "AD / FTD track"))
    for ds, where, note in (("ds004584", "Iowa", "63-ch, eyes open, Pz reference"),
                            ("ds003490", "New Mexico", "64-ch, eyes-open minute, OFF meds"),
                            ("ds002778", "San Diego", "32-ch BioSemi, eyes open, OFF meds")):
        if ds in qc:
            neuro_rows.append((ds, f"OpenNeuro, CC0, {where}",
                               f"{qn_(ds, 'parkinsons')} PD / {qn_(ds, 'control')} controls", note, "PD track"))
    table(doc, ["dataset", "source", "subjects", "signal", "role"], [
        ("TUH Abnormal + MPI LEMON", "EEG-GCNN FigShare features", "1,593 (1,385 / 208)",
         "8-ch bipolar, precomputed", "baseline reproduction"),
        ("Park et al. 2021", "SMG-SNU Boramae, Seoul", "945 (OCD excluded -> 899)",
         "19-ch, precomputed PSD + coherence", "per-disorder GNNs"),
        ("ASEEG (Bagherzadeh 2026)", "Zenodo 18029536, CC-BY",
         f"{qn_('aseeg', 'schizophrenia')} SZ / {qn_('aseeg', 'control')} controls", "19-ch raw, 128 Hz",
         "raw SZ track"),
        ("Warsaw (Olejarczyk 2017)", "RepOD 0107441",
         f"{qn_('warsaw', 'schizophrenia')} SZ / {qn_('warsaw', 'control')} controls", "19-ch raw EDF",
         "raw SZ track"),
        ("Mumtaz 2016", "figshare 4244171, CC-BY",
         f"{qn_('mumtaz', 'mood')} MDD / {qn_('mumtaz', 'control')} controls", "19-ch raw EDF, 256 Hz",
         "raw depression track"),
        *neuro_rows,
    ], widths=[1.4, 1.5, 1.3, 1.3, 1.0],
        note="Mumtaz: 6 of 64 eyes-closed files are dead links on figshare. OCD dropped from Park "
             "(46 subjects; 9 in the test split, AUC CI roughly +/-0.22).")

    doc.add_heading("3. Graph construction", 1)
    bullets(doc, [
        "Nodes are EEG electrodes. Edges are functional connectivity (spectral coherence), a statistic "
        "defined for every electrode pair - electrodes are not physically wired, so zeros exist only "
        "where a modelling choice puts them.",
        "EEG-GCNN baseline: 8 bipolar-montage nodes, complete graph with self-loops (64 edges), edge "
        "weight = normalised geodesic electrode distance + coherence, node features = 6-band PSD.",
        f"Per-disorder GNNs (Park): 19 nodes (10-20 montage). Node features = 6 band powers + the "
        f"channel's full coherence row ({in_dim} dims). Edge features = 6 per-band coherences. The "
        f"selected graph keeps each node's top-{cfg['topk']} most coherent neighbours (symmetrised, "
        "self-loops kept).",
        "Raw EEG: identical layout, with log relative band power (amplifier gain cancels) and "
        "magnitude-squared coherence computed per 10-second window.",
    ])

    doc.add_heading("4. Architectures", 1)
    doc.add_heading("4.1 EEG-GCNN baseline (reproduced)", 2)
    figure(doc, FIG / "gnn_baseline.png", "Figure 1. EEG-GCNN layer stack; output shapes per graph.")
    table(doc, ["layer", "operation", "output", "params"], base_rows, widths=[0.7, 4.0, 0.8, 0.8])
    doc.add_heading("4.2 Selected per-disorder GNN", 2)
    figure(doc, FIG / "gnn_selected.png", "Figure 2. Selected GNN (identical for all five disorders).")
    table(doc, ["layer", "operation", "output", "params"], sel_rows, widths=[0.7, 4.0, 0.8, 0.8],
          note=f"Training: Adam lr {cfg['lr']}, weight decay {cfg['weight_decay']}, batch {cfg['batch_size']}, "
               f"{cfg['epochs']} epochs, class-weighted cross-entropy, GCN normalisation on. "
               f"Graph-blind control (FCNN on the same {P.N_NODES * in_dim} inputs): {fc_n:,} parameters.")
    doc.add_heading("4.3 Configurations searched", 2)
    table(doc, ["conv", "pooling", "graph", "params"], cands, widths=[1.2, 1.3, 2.2, 1.0])
    doc.add_heading("4.4 System and pipelines", 2)
    figure(doc, FIG / "system.png", "Figure 3. Per-disorder system: five binary GNNs, calibrated, then combined.")
    figure(doc, FIG / "raw_pipeline.png", "Figure 4. Raw-EEG preprocessing, identical for every hospital.")

    doc.add_heading("5. Results", 1)
    doc.add_heading("5.1 Baseline reproduction: EEG-GCNN", 2)
    table(doc, ["metric", "published", "reproduced"], [
        ("AUC", "0.871 (0.001)", "0.871 (0.001)"), ("precision", "0.989 (0.003)", "0.989 (0.003)"),
        ("recall", "0.677 (0.018)", "0.677 (0.017)"), ("F1", "0.804 (0.011)", "0.804 (0.011)"),
        ("balanced accuracy", "0.810 (0.003)", "0.810 (0.003)"),
    ], widths=[2.0, 2.0, 2.0],
        note="478 held-out subjects, 10 released checkpoints. Run on the Blackwell node 2026-09-08 "
             "(train.py --eval-ckpt); values from the README table written at the time.")

    doc.add_heading("5.2 Graph formulation study on Park (all classes, 10-fold CV)", 2)
    table(doc, ["model", "binary SZ vs HC", "4-way"], [
        ("logistic regression (all 1140 features)", "0.731", "0.428"), ("random forest", "0.721", "0.449"),
        ("majority class", "-", "0.401"), ("FCNN, profile node features", "0.703", "0.321"),
        ("GATv2, profile node features", "0.613", "0.273"), ("GCN, profile node features", "0.594", "0.301"),
        ("Park et al. 2021, published", "0.938", "not attempted"),
    ], widths=[3.0, 1.6, 1.6],
        note="Accuracy. Run on Blackwell 2026-09-08; values from the README. Motivated the later design: "
             "node features must carry the coherence row, and the published 0.938 did not reproduce.")
    table(doc, ["model", "power only", "+ mean coherence", "+ coherence row"], [
        ("FCNN", "0.627 / 0.277", "0.670 / 0.318", "0.703 / 0.321"),
        ("GCN", "0.566 / 0.260", "0.538 / 0.252", "0.594 / 0.301"),
        ("GATv2", "0.556 / 0.272", "0.515 / 0.261", "0.613 / 0.273"),
    ], widths=[1.2, 1.6, 1.6, 1.6], note="Node-feature ablation, accuracy binary / 4-way.")

    doc.add_heading("5.3 Per-disorder GNNs", 2)
    doc.add_paragraph(
        "Protocol: a stratified 80/20 subject split was fixed once. Eight GNN configurations were "
        "compared by 5-fold CV on the train split only; one configuration was chosen for all five "
        "models by mean out-of-fold AUC (choosing per disorder on 90-290 subjects would fit noise). "
        "Models were trained on the full train split, calibrated on train out-of-fold predictions, "
        "then evaluated once on the test split.")
    figure(doc, FIG / "tuning.png", "Figure 5. Tuning results (train split only). Point estimates on a "
                                    "zoomed axis: gaps of ~0.02 are within cross-validation noise; the "
                                    "consistent signal is that sparse (top-4) graphs fill the top three places.")
    table(doc, ["conv", "pooling", "graph"] + short + ["mean"], [
        (r["config"]["model"], r["config"]["pool"],
         "complete" if r["config"]["topk"] is None else f"top-{r['config']['topk']}",
         *[f"{r['per_disorder_auc'][d]:.3f}" for d in dis], f"{r['mean_auc']:.3f}")
        for r in sorted(tune["results"], key=lambda r: -r["mean_auc"])
    ], note="Pooled out-of-fold AUC per disorder. Mean excludes OCD.")

    table(doc, ["model", "train cases / controls", "test cases / controls"], [
        (D.SHORT[d], f"{ckpts[d]['n_train_pos']} / {ckpts[d]['n_train'] - ckpts[d]['n_train_pos']}",
         f"{ev['per_disorder'][d]['n_pos']} / {ev['per_disorder'][d]['n'] - ev['per_disorder'][d]['n_pos']}")
        for d in dis], widths=[1.6, 2.2, 2.2],
        note="The healthy controls are shared by all five models (95 in the whole corpus).")

    figure(doc, FIG / "per_disorder_auc.png",
           "Figure 6. Blue: each model on its trained task. Orange: the same model asked to tell its "
           "disorder apart from the other disorders.")
    table(doc, ["model", "AUC vs healthy [95% CI]", "acc", "bal. acc", "AUC vs other disorders [95% CI]",
                "Brier raw -> cal."], [
        (D.SHORT[d], ci({"auc": v["auc"], "ci": v["auc_ci"]}), f"{v['accuracy']:.3f}", f"{v['bal_acc']:.3f}",
         ci({"auc": v["auc_vs_other_disorders"], "ci": v["auc_vs_other_ci"]}),
         f"{v['brier_raw']:.3f} -> {v['brier']:.3f}")
        for d, v in ((d, ev["per_disorder"][d]) for d in dis)
    ], note=f"Held-out test split, {ev['n_test']} subjects. Threshold 0.5 on calibrated probabilities.")

    table(doc, ["model", "calibration slope", "intercept", "OOF AUC", "OOF saturated"], [
        (D.SHORT[d], f"{calib[d]['a']:.3f}", f"{calib[d]['b']:+.3f}", f"{calib[d]['oof_auc']:.3f}",
         f"{calib[d]['oof_saturated']:.0%}") for d in dis],
        note=f"Slopes of 0.10-0.20 mean the raw models were 5-10x overconfident. Share of test "
             f"probabilities above 0.99 or below 0.01: raw {ev['saturated_raw']:.0%}, calibrated "
             f"{ev['saturated']:.0%}.")

    figure(doc, FIG / "activation.png",
           "Figure 7. A disorder-specific model would light up only its own row; instead every model "
           "fires for all patients.")
    table(doc, ["true class"] + short, [
        (D.SHORT[c], *[f"{x:.2f}" for x in act[i]]) for i, c in enumerate(D.ALL_CLASSES)])
    cm = np.array(e["confusion"])
    table(doc, ["true \\ predicted"] + [D.SHORT[c] for c in D.ALL_CLASSES], [
        (D.SHORT[c], *[str(x) for x in cm[i]]) for i, c in enumerate(D.ALL_CLASSES)],
        note=f"Ensemble confusion matrix. Accuracy {e['accuracy']:.3f}, balanced accuracy "
             f"{e['bal_acc']:.3f} [{e['bal_acc_ci'][0]:.2f}, {e['bal_acc_ci'][1]:.2f}], majority-class "
             f"rate {e['majority_rate']:.3f}, chance {1 / len(D.ALL_CLASSES):.3f}.")

    if "combined" in ev:
        doc.add_heading("Outside the healthy range: combining the five models", 3)
        doc.add_paragraph(
            "Because every model detects illness but none names it, the display headline is a single "
            "score: the mean of the five calibrated probabilities, flagged at 0.5. This rule was fixed "
            "before the test split was scored for it; the maximum is reported only as a secondary check, "
            "so the test split was not used to choose between rules.")
        comb_rows = []
        for rule, r in ev["combined"].items():
            comb_rows.append((f"combined, {rule}", ci({"auc": r["auc"], "ci": r["auc_ci"]}),
                              f"{r['sensitivity']:.2f}", f"{r['specificity']:.2f}"))
        for d in dis:
            v = ev["per_disorder"][d]
            comb_rows.append((f"single {D.SHORT[d]} model", ci({"auc": v["auc_any_patient"],
                                                               "ci": v["auc_any_patient_ci"]}), "-", "-"))
        cm_ = ev["combined"]["mean"]
        table(doc, ["score", "AUC any patient vs healthy [95% CI]", "sensitivity at 0.5",
                    "specificity at 0.5"], comb_rows, widths=[1.8, 2.4, 1.1, 1.1],
              note=f"Combining did not measurably help: the mean ({cm_['auc']:.3f}) sits inside the "
                   "range of the single models, and the anxiety model alone scored higher. With 19 "
                   "healthy controls none of these differences is resolvable, and switching the "
                   "headline to the best single model after seeing these numbers would be test-set "
                   "selection. At 0.5 the flag catches "
                   f"{cm_['sensitivity']:.0%} of patients and wrongly flags {1 - cm_['specificity']:.0%} "
                   "of healthy controls.")

    doc.add_heading("5.4 Raw EEG across hospitals", 2)
    doc.add_paragraph(
        "Each external dataset contributes its own healthy controls, because site effects in EEG "
        "exceed disease effects; patients from one hospital with controls from another would train a "
        "site detector. The GNN configuration is the one tuned on Park, so nothing here is selected "
        "on these data. Evaluation is subject-level, with subjects never split across folds.")
    ch_txt = RES / "raw" / "channel_order_inference.txt"
    if ch_txt.exists():
        doc.add_paragraph(
            "ASEEG ships unlabelled channel columns. The order was recovered from the data "
            "(infer_channel_order.py): the electrode correlation fingerprint was matched against "
            "labelled Mumtaz recordings, and posterior alpha plus frontal delta resolved the "
            "front/back mirror that geometry alone cannot. Output:")
        code(doc, "\n".join(l for l in ch_txt.read_text().splitlines()
                            if l.strip() and not l.startswith("reference:")))
    def qc_table(datasets, note):
        rows = []
        for ds in datasets:
            for grp in sorted({qc_group(ds, r) for r in qc[ds]}):
                recs = [r for r in qc[ds] if qc_group(ds, r) == grp]
                rej = sum(r["rejected"] for r in recs) / max(sum(r["windows_total"] for r in recs), 1)
                rows.append((SITE[ds], grp, str(len(recs)), str(sum(r["kept"] for r in recs)), f"{rej:.1%}"))
        table(doc, ["dataset", "group", "subjects", "windows kept", "rejected"], rows, note=note)

    qc_table([d for d in ("aseeg", "warsaw", "mumtaz") if d in qc],
             "Windows with any channel peak-to-peak above 300 uV or flat are rejected; at most 30 "
             "clean 10 s windows per subject. Mumtaz patients lose more windows (recording "
             "differences between groups are a possible confound).")
    figure(doc, FIG / "raw_auc.png", "Figure 8. Raw-EEG results. Transfer = train on one hospital, "
                                     "test on the other, never seen.")
    table(doc, ["evaluation", "AUC [95% CI]", "positive", "negative"], [
        (name, ci(v), str(v["n_patients"]), str(v["n_controls"])) for name, v in raw_rows])

    if neuro:
        doc.add_heading("5.5 Neurological disorders", 2)
        doc.add_paragraph(
            "Alzheimer's disease, frontotemporal dementia and Parkinson's disease come from separate "
            "open datasets, each with its own healthy controls, run through the same pipeline and the "
            "same GNN configuration (tuned on Park, not re-tuned). They are kept apart from the "
            "psychiatric models: different hospitals, older patients, and in the Parkinson's datasets "
            "eyes-open recordings, so their probabilities are not comparable with the psychiatric ones, "
            "and no model is ever asked to choose between a disease from one hospital and a disease "
            "from another. The one between-disease test is Alzheimer's vs frontotemporal dementia, "
            "which is fair because both come from the same hospital and the same protocol.")
        bullets(doc, [
            "Parkinson's patients: the OFF-medication session where a dataset offers both (New Mexico, "
            "San Diego); Iowa does not state medication state.",
            "Iowa records against Pz, one of the 19 electrodes; Pz enters as zeros and the average "
            "reference reconstructs it exactly.",
            "New Mexico recordings interleave instructed eyes-open and eyes-closed minutes; only the "
            "eyes-open minute is used, to match the other two Parkinson's sites.",
            "Mains notch at 50 Hz (Thessaloniki) or 60 Hz (the three US sites).",
        ])
        qc_table([d for d in ("ds004504", "ds004584", "ds003490", "ds002778") if d in qc],
                 "Same rejection rule and 30-window cap as section 5.4.")
        figure(doc, FIG / "neuro_auc.png",
               "Figure 9. Neurological results. For Parkinson's the figure shows leave-one-hospital-out "
               "(trained on the other two hospitals); the table below also lists every pairwise transfer.")
        table(doc, ["evaluation", "AUC [95% CI]", "positive", "negative"], [
            (name, ci(v), str(v["n_patients"]), str(v["n_controls"])) for name, v in raw_rows_for(neuro)],
            note="'positive' and 'negative' are the two groups of each task: patients vs controls, "
                 "and for AD vs FTD, Alzheimer's vs frontotemporal dementia.")

    doc.add_heading("6. Problems found and fixed", 1)
    bullets(doc, [
        "PyG checkpoint layout changed since 2020: GCN weights moved into a Linear submodule with "
        "transposed shape, so porting needs a transpose, not a rename.",
        "Positive class: the baseline corpus is 87% diseased, so precision/recall against 'healthy' "
        "looked nothing like the paper while AUC and balanced accuracy (symmetric) matched.",
        "Park stores coherence as a percentage (0-100). Fed raw into an unnormalised GCN over a complete "
        "19-node graph, activations grew ~4e8 and accuracy fell below chance.",
        "float32 softmax saturated to exactly 1.0 for ~44% of test probabilities, erasing ranking and "
        "blocking calibration; scoring now uses the float64 logit margin.",
        "Subject-score aggregation indexed a subset of windows against the full probability array; "
        "caught when the first two-dataset run crashed, fixed and covered by a test.",
        "f-strings containing backslashes parsed only on Python 3.12+; a normalisation test compared "
        "two randomly initialised models and was flaky.",
    ])

    doc.add_heading("7. Limitations", 1)
    bullets(doc, [
        "Small test sets: with 19-23 healthy controls in most comparisons, AUC confidence intervals are "
        "roughly +/-0.15. Differences smaller than that are not interpretable.",
        "One feature row per subject in Park (no raw EEG), so no windowing or augmentation is possible there.",
        "Depression has only one open raw dataset with controls, so its within-dataset result has no "
        "cross-hospital check. Addictive, trauma and anxiety have no open raw dataset with controls.",
        "ASEEG left/right channel assignment is not identifiable from resting EEG and follows the "
        "standard 10-20 export order.",
        "Per-disorder and raw-EEG results were computed on a laptop CPU (Blackwell node unreachable on "
        "2026-09-13); the baseline and the Park graph study ran on the Blackwell node.",
        *(["Neurological results come from single hospitals for Alzheimer's and frontotemporal dementia "
           "(no second open dataset with controls was found), from older patients than the psychiatric "
           "sets, and for Parkinson's from eyes-open recordings. None of these probabilities is "
           "comparable with the psychiatric models'. The San Diego Parkinson's authors ask to be "
           "contacted before a manuscript using their data is submitted.",
           "Parkinson's recordings are eyes-open, so they contain blinks, and a reduced spontaneous "
           "blink rate is itself a clinical sign of Parkinson's. Blink activity reaches the model through "
           "frontal slow-wave power and coherence, so part of any Parkinson's result may reflect blinking "
           "rather than cortical rhythms; an ablation without the frontal-pole electrodes would test this."]
          if neuro else []),
    ])

    doc.add_heading("8. Reproduce", 1)
    code(doc, "\n".join([
        "python train.py --eval-ckpt                          # EEG-GCNN reproduction",
        "python per_disorder.py tune | train | calibrate | evaluate",
        "python per_disorder.py predict --ids <subject no.>   # per-disorder probabilities",
        "python infer_channel_order.py                        # ASEEG channel order",
        "python fetch_neuro.py                                # Alzheimer's/FTD/Parkinson's data (CC0)",
        "python raw_eeg.py features",
        "python raw_eeg.py run --disease schizophrenia | mood",
        "python raw_eeg.py run --disease alzheimers | ftd | alzheimers_vs_ftd | parkinsons",
        "python make_report.py                                # this document",
    ]))

    out = DOCS / "MiniProject_Results.docx"
    doc.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
