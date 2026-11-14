# Results index

Every file here is written by a script in `eeg/` (run from the repository root). Newest first.

## Final system (2026-11-13)

| File | Written by | What it holds |
|---|---|---|
| `final_system.json` | `eeg.experiments.final_system` | Per task: out-of-fold scores of the 5 members, AUC per site, ensemble AUC, calibrated probabilities, Brier, ECE, accuracy at each coverage |
| `explain_final/<task>.json, .png` | `eeg.experiments.explain_final` | Within-site Cohen's d per electrode x band, LR fold stability, head maps |
| `leakage_demo.json` | `eeg.experiments.leakage_demo` | Same model with window-level (leaky) vs person-level (honest) splits |
| `mumtaz_check.json` | `eeg.experiments.mumtaz_check` | Mumtaz group-wise spectrum at 10/25/35/44 Hz; gamma-only and 1-25 Hz-only AUC |
| `prep_v2.json` | `eeg.data.prep_v2 eval` | Preprocessing v1 vs v2, 1/f power, wPLI, raw vs trace-normalised Riemannian |
| `pooled_deep.json`, `pooled_deep_bal.json` | `eeg.experiments.pooled_deep [--balance]` | Master models per disease (LR, GATv2, Graph Transformer, MIL, GroupDRO, LaBraM-FT) |

## AD/FTD audits and BrainLat (2026-11-12)

| File | Written by | What it holds |
|---|---|---|
| `three_class.json`, `three_class_brainlat.json` | `eeg.experiments.three_class` | 3-class AD / FTD / control, 7 methods + ensemble |
| `dtgcn.json` | `eeg.experiments.dtgcn` | DT-GCN port under paper, inner-validation and strict protocols |
| `graph_ablation.json` | `eeg.experiments.graph_ablation` | 6 graph types x GCN / GraphSAGE / GATv2 |
| `cross_site.json` | `eeg.experiments.cross_site` | Thessaloniki <-> BrainLat transfer |

## Within-dataset phase (2026-11-05)

| File | Written by | What it holds |
|---|---|---|
| `ensemble_v2.json`, `ensemble.json` | `eeg.experiments.ensemble` | Six-member (v2) and four-member ensembles per dataset |
| `baselines_{base,rich,both}.json` | `eeg.experiments.baselines` | LR / SVM / gradient boosting |
| `riemann.json` | `eeg.models.riemann` | Filter-bank Riemannian |
| `labram.json` | `eeg.experiments.labram_probe` | Frozen LaBraM probe |
| `pooled.json` | `eeg.experiments.pooled` | Pooled multi-site classical models |
| `sharpened.json` | `eeg.experiments.sharpened` | Clinically sharper secondary tasks |
| `site_probe.json` | `eeg.experiments.site_probe` | Hospital vs disease predictability |
| `tigga.json` | `eeg.experiments.tigga` | Re-run of Tigga et al. GATv2-LSTM |
| `explain/` | `eeg.experiments.explain` | GATv2 attribution maps |
| `final/raw_*.json` | `eeg.data.raw_eeg run --model ...` | Raw-EEG GNN, MSBP, GATv2-LSTM, ComBat, alignment, DANN |
| `subject_counts.json` | `eeg.experiments.ensemble` | Subjects per task |

## First phases (2026-09)

| Path | Written by | What it holds |
|---|---|---|
| `per_disorder/` | `eeg.experiments.per_disorder` | Park corpus per-disorder GNNs: split, tuning, models, test predictions |
| `per_disorder_gcn/` | `eeg.experiments.per_disorder` | Same with the GCN config |
| `raw/` | `eeg.data.raw_eeg run` | First-pass raw-EEG runs (September) |
