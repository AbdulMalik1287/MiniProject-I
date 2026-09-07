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
