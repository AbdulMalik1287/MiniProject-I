# Brain Signal Processing

**Detecting brain and mental disorders from resting-state EEG with graph neural networks, honestly evaluated.**

Mini project, IIIT Nagpur CSE, Jul–Dec 2026 · Supervisor: Dr. Nileshchandra Pikle ·
Team: Abdul Malik, Ashwin Rathore, Yash Khedekar, Rajat Malik

We built about 30 models (graph networks, transformers, Riemannian and classical baselines) on 12 public
EEG datasets and froze one system per disorder. Every number comes from people the model never saw
during training: splits are by subject, and nothing is tuned on test folds.

## Final results

One frozen system per disorder: five scale-free models (logistic regression, Riemannian, GATv2, Graph
Transformer, fine-tuned LaBraM) vote. Each system is trained on all datasets of its disorder together,
then calibrated, and it can abstain when unsure.

| Disorder | Dataset (hospital) | AUC [95% CI] | Accuracy, all / most confident 50% |
|---|---|---|---|
| Alzheimer's vs healthy | Thessaloniki | **0.94** [0.87–0.99] | 77% / 91% (both sites) |
| | BrainLat* | 0.75 [0.62–0.86] | |
| FTD vs healthy | Thessaloniki | **0.89** [0.78–0.97] | 73% / 79% |
| | BrainLat* | 0.64 [0.49–0.79] | |
| Alzheimer's vs FTD | Thessaloniki / BrainLat | 0.71 / **0.86** | 68% / 82% |
| Depression vs healthy | Mumtaz† | **0.97** [0.91–1.00] | 93% / 97% |
| Schizophrenia vs healthy | ASEEG / Warsaw / Semmelweis | 0.84 / 0.82 / 0.67 | 70% / 84% |

\* BrainLat patients and controls were exported as different file types; we use scale-free inputs only,
but treat these rows with caution. † Mumtaz groups were filtered differently above ~40 Hz; using only
1–25 Hz the score is 0.95, the number we quote.

## What we found

- **Bigger models do not help at this data size.** Logistic regression ties the GNNs and transformers. A
  graph with no edges scores the same as six graph constructions. Attention-MIL and GroupDRO were worse.
- **Better inputs and combining models do help.** Spreading windows over the whole recording,
  1/f-corrected band power and a five-model ensemble gave the final gains.
- **Hospital fingerprints dominate.** A linear model identifies the hospital from healthy people alone at
  AUC 0.88–1.00, so transfer to an unseen hospital stays near chance despite ComBat, alignment and DANN.
- **Published high scores often leak.** Letting a person's windows into both training and test gives ~99%
  on every task, even Alzheimer's vs FTD, where the honest score is 0.57: the model recognises people,
  not disease. DT-GCN's reported 82% falls to 52% once epochs are no longer picked on the test fold, and
  Tigga et al.'s graph has no effect while their split mixes subjects.
- **Two recording confounds found and handled:** BrainLat's file-type amplitude gap (a fake AUC of 0.99
  that drops to 0.70 when inputs are scale-free) and Mumtaz's group-wise filter difference.
- **The models see known physiology:** Alzheimer's shows theta "slowing" with alpha/beta loss, the same at
  both hospitals; FTD shows reduced alpha/beta; schizophrenia shows right-temporal delta increases.

## Documents

| | |
|---|---|
| [Team update (PDF)](docs/Team_Update_2026-11-13.pdf) | Plain-language summary and full work history |
| [Results report (docx)](docs/MiniProject_Results.docx) | Every experiment, table and figure (sections 5.1–5.18) |
| [Readout page](docs/eeg_readout.html) | Final results per hospital and an interactive per-subject readout |
| [Model registry](docs/MODEL_REGISTRY.md) | All 33 models and schemes with hyperparameters and parameter counts |
| [Project history](docs/HISTORY.md) | Lab notebook of the first phases (EEG-GCNN reproduction, Park corpus) |
| [Bibliography](literature/REFERENCES.md) | References (`literature/fetch_pdfs.py` downloads the open-access PDFs) |
| [Results index](results/README.md) | Which script wrote which result file |

## Repository layout

```
eeg/
  data/          dataset loaders and the EEG pipeline
                 raw_eeg.py (filter, window, features), prep_v2.py (final preprocessing),
                 rich_features.py, park.py, fetch_neuro.py
  models/        train.py (EEG-GCNN, GATv2, Graph Transformer), msbp.py (MSBP-GNN), st_gnn.py (GATv2-LSTM),
                 site_invariant.py (ComBat, alignment, DANN), riemann.py
  experiments/   every evaluation in the report; final_system.py is the frozen final system,
                 explain_final.py its explanation, pooled_deep.py the master models
  reporting/     make_report.py (docx), readout.py (HTML page), team_update.py (PDF)
  paths.py       repository and data locations
results/         result files (JSON, figures) written by the experiments
docs/            report, PDF, readout page, registry, history
literature/      bibliography
tests/           smoke test
```

## Reproducing

Python 3.11 with a CUDA GPU (an RTX 2060 with 6 GB is enough). Run everything from the repository root.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python tests/smoke_test.py                       # imports every module and runs the built-in checks
```

Data goes in `data/external/<dataset>/` (36 GB in total). OpenNeuro sets download with
`python -m eeg.data.fetch_neuro`. Mumtaz, ASEEG, Warsaw, Semmelweis and BrainLat (Synapse; needs an
account) download from their sources, listed in the report. Then:

```bash
OUT=~/eeg/raw2                                                   # feature cache
python -m eeg.data.raw_eeg features --out $OUT --datasets ds004504 brainlat mumtaz aseeg warsaw semmelweis
python -m eeg.data.prep_v2 extract --out $OUT                    # final preprocessing
python -m eeg.experiments.final_system --out $OUT                # ~1 h on a 6 GB GPU
python -m eeg.experiments.explain_final --out $OUT
python -m eeg.reporting.make_report                              # docs/MiniProject_Results.docx
python -m eeg.reporting.readout                                  # docs/eeg_readout.html
python -m eeg.reporting.team_update                              # docs/Team_Update_2026-11-13.pdf
```

Every other result has its command in the docstring at the top of its script, and the report's
"Reproduce" section lists them in order.

## Caveats

- Datasets have 28–145 people each, so confidence intervals are wide (often ±0.10) and most differences
  between models are noise.
- Depression rests on one dataset (Mumtaz). UNM was dropped because its labels come from a questionnaire:
  only 11 of 46 "depressed" students had depression confirmed by clinical interview.
- Generalising to a hospital the model has never seen is not solved.
- The San Diego Parkinson's authors ask to be contacted before a manuscript using their data is submitted.
- Research code, not a diagnostic tool.
