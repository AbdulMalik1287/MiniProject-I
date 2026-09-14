# Brain Signal Processing — EEG + GNN disorder classification

Mini project, IIIT Nagpur CSE, Jul–Dec 2026. Supervisor: Dr. Nileshchandra Pikle.

**Base paper:** Wagh & Varatharajah, *EEG-GCNN: Augmenting EEG-based Neurological
Disease Diagnosis using a Domain-guided Graph Convolutional Neural Network*,
ML4H @ NeurIPS 2020. [arXiv:2011.12107](https://arxiv.org/abs/2011.12107) ·
[code](https://github.com/neerajwagh/eeg-gcnn)

## Why this paper

Selected over the originally suggested [EEG_RL-Net](https://ieeexplore.ieee.org/document/10977776/),
which was rejected on two grounds: reinforcement learning is the core of the
method (not removable), and its task is motor-imagery BCI decoding, not disorder
identification — it does not match the synopsis.

Selected over [ExPANet](https://arxiv.org/abs/2511.05537) (Hazra & Ghosh, IIT
Kharagpur, 2025) — better topical fit and Indian-authored, but its repo is not
released yet (404, README says "will be"), and at 97.5% reported accuracy it
leaves no headroom. It is the **target to beat**, cited in related work.

EEG-GCNN wins on being reproducible now: released checkpoints, precomputed
features on FigShare, and — checked in the source, not assumed — **correct
subject-wise splitting** with subject-level metric aggregation.

## The graph

Verified against `EEGGraphDataset.py`, not taken from the paper text:

- **8 nodes** — bipolar montage `F7-F3, F8-F4, T7-C3, T8-C4, P7-P3, P8-P4, O1-P3, O2-P4`
- **64 edges** — `product(8, 8)`, i.e. complete graph *including self-loops*, 100% density
- **edge weight** = min-max normalised geodesic electrode distance **+** spectral coherence
  (summed into one scalar upstream; this harness can keep them as 2 channels)
- **node features** = 6-band power spectral density, shape `(8, 6)`

## Reproduction status: DONE (2026-09-08, milestone was 15 Sep)

`python train.py --eval-ckpt` on the 478 held-out subjects, 10 released
checkpoints. Every metric matches the published figures including the standard
deviations:

| metric | published | reproduced |
|---|---|---|
| AUC | 0.871 (0.001) | **0.871 (0.001)** |
| precision | 0.989 (0.003) | **0.989 (0.003)** |
| recall | 0.677 (0.018) | **0.677 (0.017)** |
| F1 | 0.804 (0.011) | **0.804 (0.011)** |
| balanced accuracy | 0.810 (0.003) | **0.810 (0.003)** |

Dataset as loaded: **225,334 windows, 1,593 subjects**, 87% diseased / 13%
healthy (1385 / 208 subjects). Held-out split is 478 subjects at seed 42.

Two portability issues had to be solved to get there, both worth knowing:

1. **Checkpoint layout changed.** 2020-era PyG stored `conv1.weight` as
   `(in, out)` and computed `x @ W`; current PyG keeps it in an `nn.Linear`
   submodule as `conv1.lin.weight` with shape `(out, in)` and computes
   `x @ Wᵀ`. Porting needs a **transpose**, not just a rename — see
   `port_state_dict`.
2. **Positive class is `diseased`, not `healthy`.** The corpus is 87% diseased,
   so precision/recall/F1 computed against `healthy` describe detection of the
   rare class and look nothing like the paper. AUC and balanced accuracy are
   symmetric under that flip — which is why they matched while the others did
   not. If you ever see those two agree and the rest disagree, check the
   positive class first.

## Phase 2: multi-disorder on Park et al. (in progress)

`park.py` — 945 subjects, 19 channels, single site (SMG-SNU Boramae, Seoul).
Verified against the paper: Mood 266, Addictive 186, Trauma 128, SZ 117,
Anxiety 107, Healthy 95, OCD 46. Features are 114 band-power (19×6) and 1026
coherence (171 pairs × 6 bands) columns.

Graph: 19 nodes, complete + self-loops (361 edges), node features = 6 band
powers, edge features = 6 per-band coherences.

```bash
python park.py --task multiclass --model gcn       # 4-way deliverable
python park.py --task binary --target Schizophrenia # reference, comparable to Park
```

### Results

Node-feature modes (`--node-features`) control how much connectivity each node
carries: `power` = 6 band powers only; `strength` = + mean coherence per band
(12); `profile` = + the channel's full coherence row (120), the same
connectivity information the classical baselines get.

Accuracy, 10-fold stratified, best node-feature mode per model:

| | binary SZ vs HC (n=212) | 4-way (n=664) |
|---|---|---|
| logistic regression (AB+COH) | **0.731** | 0.428 |
| random forest | 0.721 | **0.449** |
| majority class | — | 0.401 |
| fcnn (profile) | 0.703 | 0.321 |
| gatv2 (profile) | 0.613 | 0.273 |
| gcn (profile) | 0.594 | 0.301 |
| *Park et al. published* | *0.938* | *not attempted in the literature* |

Effect of feeding connectivity into the nodes (binary / 4-way accuracy):

| model | power | strength | profile |
|---|---|---|---|
| fcnn | 0.627 / 0.277 | 0.670 / 0.318 | **0.703 / 0.321** |
| gcn | 0.566 / 0.260 | 0.538 / 0.252 | **0.594 / 0.301** |
| gatv2 | 0.556 / 0.272 | 0.515 / 0.261 | **0.613 / 0.273** |

**The bottleneck was real and it is now mostly closed.** `profile` node features
lift every model, and the FCNN goes 0.627 → 0.703 on binary, within noise of
logistic regression's 0.731. The earlier deficit was an artefact of starving the
graph, not evidence that GNNs are unsuited — which is why it was recorded as a
diagnosis rather than a finding.

**Graph structure does not help on this corpus.** Given identical information,
the graph-blind FCNN beats GCN and GATv2 on both tasks. With 19 nodes and a
complete graph there is little topology to exploit, and message passing over a
fully-connected graph mostly averages the nodes together. Report this as a
negative result about *this graph formulation at this scale*, not about GNNs.

**The binary → multi-class collapse is the headline.** Logistic regression drops
0.731 → 0.428, and the best 4-way model (0.449, random forest) barely clears the
0.401 majority-class rate. Every architecture tried, classical and neural, fails
to separate the disorders from each other while succeeding at
patient-vs-control. Four-way discrimination on these features is close to not
working, and that reproduces across nine model/feature combinations.

**The published number does not reproduce.** Park reports 0.938 for SZ vs HC;
logistic regression on the same features with a clean stratified split reaches
0.731.

### Two bugs found and fixed

1. **Park stores coherence as 0–100, not [0,1].** Fed raw into an unnormalised
   GCN over a 19-node complete graph, activations blew up ~4×10⁸ and accuracy
   went *below chance* (0.439 acc, 0.412 AUC on a binary task). `load_park`
   now rescales and asserts the range.
2. **`normalize=False` is correct only for the baseline.** It reproduces
   EEG-GCNN's 8-node model exactly, but on a 19-node complete graph it must be
   `True`. It is now a parameter, defaulting to `False` so the reproduction is
   untouched — verified by re-running `--eval-ckpt` after every change.

## Phase 3: one GNN per disorder, and raw EEG across hospitals

Full write-up with architectures, all tables and figures:
**[`docs/MiniProject_Results.docx`](docs/MiniProject_Results.docx)**, rebuilt from the result
files by `python make_report.py`.

### Per-disorder GNNs (`per_disorder.py`, Park corpus)

Five binary GNNs (disorder vs healthy) plus an ensemble. OCD excluded (46 subjects).
Protocol: stratified 80/20 subject split fixed once (`results/per_disorder/split.json`);
8 GNN configs compared by 5-fold CV on the train split only; one config chosen for all
five models; trained, Platt-calibrated on train out-of-fold predictions, then evaluated
once on the 180 untouched test subjects.

Selected: **GATv2, mean+max pooling, top-4 coherence graph** (9,952 parameters). The three
best configs all use sparse graphs.

| model | AUC vs healthy [95% CI] | AUC vs other disorders [95% CI] |
|---|---|---|
| schizophrenia | 0.705 [0.53, 0.86] | 0.484 [0.37, 0.61] |
| mood | 0.709 [0.54, 0.86] | 0.506 [0.41, 0.60] |
| addictive | 0.696 [0.53, 0.84] | 0.592 [0.49, 0.69] |
| trauma | 0.733 [0.57, 0.88] | 0.536 [0.41, 0.66] |
| anxiety | 0.797 [0.65, 0.92] | 0.508 [0.37, 0.65] |

Every model detects its disorder against healthy controls (all CIs above 0.5); none
separates its disorder from the other disorders (all CIs include 0.5). The models detect
psychiatric illness, not which illness. Ensemble balanced accuracy 0.235 [0.18, 0.29]
over 6 classes (chance 0.167). Raw models were 5-10x overconfident (calibration slopes
0.10-0.20); calibration took saturated probabilities from 44% to 0%.

**Showing the probabilities.** The five outputs are independent (they do not sum to 1) and
rise together for most patients, so the display leads with one screen - the pre-declared
**combined score, the mean of the five**, flagged at 0.5 - and shows the five bars beneath
it, each labelled with its measured reliability. On the held-out split the combined score
reaches AUC 0.738 [0.59, 0.85] for any patient vs healthy, catching 70% of patients and
wrongly flagging 42% of healthy controls. It did **not** beat the single models (0.63-0.79),
and with 19 controls none of those differences is resolvable.

- Readout page for all 180 held-out subjects: [`docs/eeg_readout.html`](docs/eeg_readout.html),
  built from the result files by `python readout/build.py`
- Same readout on the command line:

```bash
python per_disorder.py predict --ids 503 730 --out results/per_disorder   # trained models are committed
```

### Raw EEG across hospitals (`raw_eeg.py`)

Every external dataset brings its own controls. One pipeline for all sources; the GNN
config is the one tuned on Park, not re-tuned here. ASEEG ships unlabelled channels -
`infer_channel_order.py` recovers the order from the data.

| evaluation | AUC [95% CI] |
|---|---|
| schizophrenia, within ASEEG (51 / 50) | 0.769 [0.67, 0.86] |
| schizophrenia, within Warsaw (14 / 14) | 0.765 [0.56, 0.94] |
| schizophrenia, pooled both hospitals | 0.754 [0.67, 0.83] |
| **train ASEEG, test Warsaw** | **0.526 [0.31, 0.74]** |
| **train Warsaw, test ASEEG** | **0.505 [0.39, 0.62]** |
| depression, within Mumtaz (30 / 28) | 0.945 [0.87, 1.00] |

Within a hospital the GNN detects schizophrenia; trained at one hospital it is at chance
at the other. What is learned is largely site-specific. The depression result has no
second open dataset to check it against, so it should not be read as a transferable
signature.

Phase 3 ran on a laptop CPU (Blackwell unreachable on 2026-09-13).

## Setup (blackwell)

```bash
ssh blackwell
~/envs/eeg/bin/python   # torch 2.11.0+cu128, PyG 2.8.0, CUDA available
cd ~/eeg                # train.py, eeg-gcnn/ (upstream), data/ (FigShare)
```

## Usage

```bash
python train.py --selfcheck                      # no dataset needed

python train.py --eval-ckpt                      # reproduce published Table 2
python train.py --model gcn                      # baseline, trained from scratch
python train.py --model fcnn                     # graph-blind control
python train.py --model gatv2 --edge-mode split  # attention that reads the edges
python train.py --model gcn --topk 3             # sparsity sweep
```

## Design notes

**One pipeline, one flag.** Data loading, splits, seeds, and metrics are shared
across all models. `--model` is the only difference between the September
baseline and the October attention work, so results are comparable and the
"consistent environment" concern costs nothing — `GCNConv` and `GATv2Conv` are
the same import from the same package.

**`--edge-mode split` matters.** Plain `GATConv` has no `edge_weight` argument
and would silently ignore the coherence and distance weights that are the
paper's entire contribution. `GATv2Conv(edge_dim=...)` with `edge_attr` is what
makes attention a *correction on top of* the connectivity estimate rather than a
replacement for it.

**Sparsification is a modelling choice, not a property of the brain.** Electrodes
are not wired to each other; an edge is a statistic (coherence) that is defined
and non-zero for every pair. Zeros exist only where you put them. `--topk` /
`--thresh` put them there so the effect can be measured.

## Planned experiments (Oct–Nov)

| axis | variants |
|---|---|
| conv | `fcnn`, `gcn`, `cheb` (K-hop), `gatv2` |
| edges | `sum` (paper) vs `split` (attention can read both components) |
| density | complete (paper) vs `--topk {2,3,4}` vs `--thresh` sweep |

The 2×2 worth isolating: **attention** vs **input-dependent adjacency**. Tang et
al. (ICLR 2022) get per-sample graphs with no attention at all, so the two are
separable and nobody reports them apart for disorder detection.

## Caveats to keep honest

- 8 nodes is very few for a GNN. If `--model fcnn` matches the GNNs, graph
  structure is not contributing and that is the finding.
- Published AUC is **0.871** (repo README, after an edge-weight type-cast fix),
  not the 0.90 quoted in the abstract. 0.871 is the reproduction target.
- The subject-wise-split leakage critique applies to the MDD literature
  (HUSM/MODMA), **not** to EEG-GCNN, which already splits correctly.
