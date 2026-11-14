"""Figures for the final project report, drawn from the committed result files.

    python -m eeg.reporting.report_figures      # -> docs/figures/report_*.png
"""
import json

import matplotlib.pyplot as plt
import numpy as np

from eeg.paths import DOCS, RESULTS
from eeg.reporting.make_report import AXIS, INK, INK2, MUTED, S1, S2, box_diagram, dot_whisker

FIG = DOCS / "figures"
SITE = {"ds004504": "Thessaloniki", "brainlat": "BrainLat", "mumtaz": "Mumtaz", "aseeg": "ASEEG",
        "warsaw": "Warsaw", "semmelweis": "Semmelweis"}
TASKS = [("alzheimers", "AD vs healthy"), ("ftd", "FTD vs healthy"), ("alzheimers_vs_ftd", "AD vs FTD"),
         ("mood", "Depression vs healthy"), ("schizophrenia", "Schizophrenia vs healthy")]


def load(name):
    return json.loads((RESULTS / name).read_text())


def system():
    box_diagram(FIG / "report_system.png", [
        ("Recording", "resting EEG\n19 channels\n(any hospital)"),
        ("Preprocess", "notch, 128 Hz\n1-45 Hz, avg ref\n10 s windows"),
        ("Select", "skip 60 s\nup to 60 clean\nwindows, spread"),
        ("Features", "relative + 1/f\nband power,\ncoherence, cov."),
        ("5 members", "LR, Riemann,\nGATv2, Graph Tr.,\nLaBraM-FT"),
        ("Ensemble", "mean rank\n+ Platt\ncalibration"),
        ("Output", "probability\nor 'unsure'"),
    ], title="Final per-disorder system", height=1.7)


def final_auc():
    fs = load("final_system.json")
    labels, vals = [], []
    for t, name in TASKS:
        for s, r in fs[t]["ensemble"]["per_site"].items():
            if s != "all":
                labels.append(f"{name} - {SITE[s]}")
                vals.append((r["auc"], *r["ci"]))
    dot_whisker(FIG / "report_final_auc.png", labels, [("final system", S1, vals)],
                "subject-level AUC with 95% bootstrap CI", "Final system, AUC per hospital", xlim=(0.3, 1.0))


def coverage():
    fs = load("final_system.json")
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    cov = [100, 90, 80, 70, 60, 50]
    for (t, name), mk in zip(TASKS, ["o", "s", "^", "D", "v"]):
        c = fs[t]["ensemble"]["coverage"]
        ax.plot(cov, [100 * c[f"{k}%"]["accuracy"] for k in cov], marker=mk, linewidth=1.6, markersize=5, label=name)
    ax.invert_xaxis()
    ax.set_xlabel("share of people the system answers (%)")
    ax.set_ylabel("accuracy on answered people (%)")
    ax.set_title("Accuracy when the system may say 'unsure'", fontsize=9.5, color=INK, loc="left")
    ax.legend(fontsize=7.5, loc="lower right")
    fig.savefig(FIG / "report_coverage.png")
    plt.close(fig)


def leakage():
    lk = load("leakage_demo.json")
    names = {"alzheimers": "AD vs healthy", "ftd": "FTD vs healthy", "alzheimers_vs_ftd": "AD vs FTD",
             "mood": "Depression", "schizophrenia": "Schizophrenia"}
    labels = [f"{names[k.split(':')[0]]} - {SITE[k.split(':')[1]]}" for k in lk]
    dot_whisker(FIG / "report_leakage.png", labels,
                [("window-level split (leaky)", S2, [(v["leaky"]["auc"],) * 3 for v in lk.values()]),
                 ("person-level split (honest)", S1, [(v["honest"]["auc"],) * 3 for v in lk.values()])],
                "window-level AUC", "Same model, two ways of splitting the data", xlim=(0.4, 1.02))


def ablation():
    ga = load("graph_ablation.json")
    graphs = ["none", "spatial", "pearson", "plv", "coherence", "hybrid"]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(len(graphs))
    for i, (m, name) in enumerate([("gcn", "GCN"), ("sage", "GraphSAGE"), ("gatv2", "GATv2")]):
        ax.bar(x + (i - 1) * 0.26, [ga[f"{g}:{m}"]["macro_f1"][0] for g in graphs], 0.26,
               yerr=[ga[f"{g}:{m}"]["macro_f1"][1] for g in graphs], label=name, capsize=2,
               error_kw={"elinewidth": 0.8})
    ax.axhline(1 / 3, color=MUTED, linewidth=1)
    ax.text(len(graphs) - 0.5, 1 / 3 + 0.01, "chance", color=MUTED, fontsize=7.5, ha="right")
    ax.set_xticks(x, graphs)
    ax.set_ylabel("macro-F1 (AD / FTD / healthy)")
    ax.set_ylim(0, 0.7)
    ax.set_title("Graph construction does not matter: 'none' = no edges", fontsize=9.5, color=INK, loc="left")
    ax.legend(fontsize=7.5, ncol=3, loc="upper left")
    fig.savefig(FIG / "report_ablation.png")
    plt.close(fig)


def dtgcn():
    d = load("dtgcn.json")
    names = ["reported in paper", "their protocol\n(best test epoch)", "epoch chosen on\ninner validation",
             "last epoch\n(no selection)"]
    vals = [0.82, d["paper_protocol_dynamic"]["accuracy_best_epoch"][0], d["inner_val_dynamic"]["accuracy"][0],
            d["strict_dynamic"]["accuracy"][0]]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    bars = ax.bar(names, vals, color=[MUTED, S2, S1, S1], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=8, color=INK2)
    ax.axhline(1 / 3, color=AXIS, linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("3-class accuracy")
    ax.set_title("DT-GCN re-run: the reported accuracy depends on test-fold selection", fontsize=9.5,
                 color=INK, loc="left")
    fig.savefig(FIG / "report_dtgcn.png")
    plt.close(fig)


def main():
    for f in (system, final_auc, coverage, leakage, ablation, dtgcn):
        f()
        print("drew", f.__name__)


if __name__ == "__main__":
    main()
