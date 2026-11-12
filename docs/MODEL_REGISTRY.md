# Model registry

Every model trained in this project, with its settings as they are in the code. Branch `gcn-comparison`.
All evaluation is subject-wise: no window of a test subject is ever trained on. Seed 42 unless stated.

## Shared pipeline (raw-EEG models, `raw_eeg.py`)

| Step | Setting |
|---|---|
| Channels | 19 standard 10-20 electrodes (T7/T8/P7/P8 mapped to T3/T4/T5/T6) |
| Filtering | mains notch (50/60 Hz) -> resample 128 Hz -> 1-45 Hz FIR -> common average reference |
| Windows | 10 s, non-overlapping; reject if any channel > 300 uV peak-to-peak or < 0.2 uV std; first 30 clean windows per subject |
| Bands | delta 1-4, theta 4-8, alpha 8-12, beta 12-25, high-beta 25-30, gamma 30-45 Hz |
| Node features | log relative band power (19 x 6); "profile" mode appends each electrode's coherence row (19 x 6) -> 120-d |
| Edges | magnitude-squared coherence per band (2 s Hann, 50% overlap), top-4 neighbours per node |
| Subject score | mean of window probabilities |
| CV | 5-fold StratifiedGroupKFold / StratifiedKFold over subjects |

## Graph neural networks

| # | Model | File | Architecture | Training | Params |
|---|---|---|---|---|---|
| 1 | **EEG-GCNN** (base paper, reproduced) | `train.py` | 8 bipolar nodes, complete graph + self-loops (64 edges); edge = geodesic distance + spectral coherence; node = 6-band PSD; GCNConv 6->32->20 (improved, unnormalised), BatchNorm, LeakyReLU, add pool, FC 20->10->2 | Released checkpoints ported (transpose of `conv.lin.weight`); evaluated on 478 held-out subjects | 1,156 |
| 2 | **GATv2 per disorder** (Park) | `per_disorder.py` | GATv2Conv(120 -> 8 x 4 heads, edge_dim 6) -> GATv2Conv(32 -> 20) -> BatchNorm -> mean+max pool (40) -> dropout 0.2 -> FC 40->10->2 | Adam lr 1e-3, wd 5e-4, batch 64, 150 epochs, class-weighted CE, Platt calibration; selected = grid index 7 of 8 (gcn/gatv2 x add/meanmax x topk none/4) | 9,952 |
| 3 | **GCN per disorder** (Park) | `per_disorder.py` | Same as 2 with GCNConv 120->32->20, normalised, edge weight = sum of band coherence (grid index 3) | Same as 2 | 5,004 |
| 4 | **GATv2, raw EEG** | `raw_eeg.py --model gnn` | Model 2's config, not re-tuned | Same, 20 epochs (~60k graph visits per fit) | 9,952 |
| 5 | **MSBP-GNN** (Geng et al., Sensors 2026), rebuilt | `msbp.py` | 2 s sub-windows (z-scored per channel); per-electrode depthwise temporal convs, kernels 51/25/13 (8 filters each) -> conv 7, stride 2 -> 64-d + attention pooling; two hemisphere graphs of 11 nodes (Fz/Cz/Pz shared), edges = cosine similarity, ReLU, top-10 per sample; 2 x ChebConv(K=3) with residuals; hemisphere mean pool; attention fusion; dropout 0.3 -> 1 logit | AdamW lr 1e-3, wd 1e-4, batch 256, 12 epochs, BCE with pos_weight | 36,627 |
| 6 | **GATv2-LSTM** (spatio-temporal) | `st_gnn.py` | Model 4's GATv2 embedding per window (40-d) -> LSTM(40 -> 32) over chunks of 6 consecutive windows -> dropout 0.2 -> 1 logit | Adam lr 1e-3, wd 5e-4, batch 32 chunks, 20 epochs, random chunk offset per epoch, BCE with pos_weight | 19,457 |
| 7 | **GATv2 + ComBat** | `site_invariant.py` | Model 4 on ComBat-harmonised features (neuroCombat, parametric; band power and coherence fitted as separate blocks; batch = hospital, no covariates, unlabeled test site included) | As 4 | 9,952 |
| 8 | **GATv2 + Euclidean alignment** | `site_invariant.py` | Model 4 on features recomputed after per-subject whitening by C^-1/2 (mean spatial covariance + ridge 1e-3 x trace/19) | As 4 | 9,952 |
| 9 | **Domain-adversarial GATv2 (DANN)** | `site_invariant.py` | Model 4 + hospital head Linear(40,16)-ReLU-Linear(16, n_sites) behind gradient reversal | As 4; reversal weight lambda = 2/(1+e^(-10p)) - 1, max 1.0; unlabeled held-out site in the hospital loss | 10,642 (2 sites) |
| 10 | **GCN / GraphSAGE / GATv2 ablation** | `graph_ablation.py` | Node = band power only (6-d); graph = none / spatial exp(-d/d_mean) / Pearson / PLV / coherence / hybrid, top-4 + self-loops; 2 conv layers 32 -> 20, BatchNorm, LeakyReLU, mean+max pool, MLP 40->10->3 | Adam lr 1e-3, wd 5e-4, batch 64, 20 epochs, class-weighted CE; 3 repeats x 5 folds | GCN 1,367 / SAGE 2,199 / GATv2 2,407 |
| 11 | **DT-GCN port** (Liu & Meng 2026) | `dtgcn.py` | Per-subject PLV matrix; min-max; binary edges above the threshold percentile; node = [identity(19), degree] (20-d); GCN 20->64->32->16 (BatchNorm, ReLU, dropout 0.5) -> GATConv(16,16, 8 heads, mean, dropout 0.6) -> add pool -> Linear(16,3) | Adam lr 0.01, wd 1e-4, batch 8, 500 epochs; dynamic threshold rule from their code, start 0.5, clip [0.3, 0.7] | 6,547 |
| 12 | **Tigga et al. GATv2-LSTM** (re-run) | `tigga.py` | Their model: GATv2Conv(19 -> 64 x 4 heads) -> LayerNorm -> 2-layer LSTM(256 -> 128, dropout 0.3) -> fusion Linear -> residual -> FC 256->3; their edge list (19 x 19 complete, 20% dropped) | Their settings: Adam lr 0.005, 100 epochs full batch, StandardScaler + noise 0.01; 500 time points per subject | 440,323 |

## Classical and pretrained models

| # | Model | File | Features | Settings |
|---|---|---|---|---|
| 13 | Logistic regression | `baselines.py` | Subject mean: 114 band power + 1,026 coherence values | StandardScaler, L2, C = 0.05 |
| 14 | SVM | `baselines.py` | As 13 | StandardScaler, RBF, C = 1.0, Platt probabilities |
| 15 | Gradient boosting | `baselines.py` | As 13 | HistGradientBoosting, 200 iterations, lr 0.05, depth 3 |
| 16 | Logistic regression, new markers | `rich_features.py`, `baselines.py` | 19 x 15 markers: log absolute band power (6), aperiodic exponent + offset (fit 2-40 Hz, alpha 7-14 Hz excluded), peak alpha frequency, theta/alpha, theta/beta, spectral entropy, Hjorth mobility + complexity, permutation entropy (order 3) | As 13 |
| 17 | Logistic regression, both feature sets | `baselines.py` | 13 + 16 | As 13 |
| 18 | Filter-bank Riemannian | `riemann.py` | 5 bands (1-4, 4-8, 8-13, 13-30, 30-45 Hz); OAS covariance per window; Riemannian mean per subject; tangent space at the training subjects' mean | StandardScaler, logistic regression C = 0.1 |
| 19 | LaBraM, frozen | `labram_probe.py` | `braindecode/labram-pretrained` (5.8M params, MIT); windows resampled to 200 Hz, scaled to 0.1 mV, 19 channels by name; mean token of block 6 (and final layer) | Logistic regression C = 0.05 on subject means; no fine-tuning |
| 20 | Random forest | `three_class.py` | 13 + 16 | 500 trees, balanced class weights |
| 21 | XGBoost | `three_class.py` | 13 + 16 | 300 trees, depth 3, lr 0.05, subsample 0.8, colsample 0.5 |

## Ensembles and training schemes

| # | Model | File | Definition |
|---|---|---|---|
| 22 | Ensemble v1 | `ensemble.py` | Mean within-fold rank of 4 members: GATv2, MSBP-GNN, LR, LR markers |
| 23 | **Ensemble v2 (headline)** | `ensemble.py` | Same, 6 members: v1 + Riemannian + LaBraM; members fixed before results |
| 24 | 3-class ensemble | `three_class.py` | Mean class probabilities of 7 members: GATv2 (3-class head), LR, LR markers, Riemannian, LaBraM, RF, XGBoost |
| 25 | Pooled multi-site training | `pooled.py` | LR / LR markers / Riemannian trained on all same-condition sites of a disorder plus the target's training folds; per-site z-scoring or Riemannian re-centring (label-free) |

## Evaluation protocols used

| Name | Where | Definition |
|---|---|---|
| Within-dataset CV | most tables | 5-fold over subjects; repeated 5x for classical models (seeds 0-4) |
| Shared split | `ensemble.py` | StratifiedKFold(5, seed 42) over subjects, identical for all members |
| Cross-hospital | `raw_eeg.py` | train on one site, test on another; leave-one-hospital-out (paused) |
| DT-GCN protocols | `dtgcn.py` | paper (best test-fold epoch), inner validation (epoch picked on 20% of training subjects), strict (last epoch) |
