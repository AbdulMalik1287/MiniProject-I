# Literature

Every paper and dataset this project used, cited, evaluated or rejected. Each entry
was verified against Crossref or the arXiv API, not typed from memory.

**PDFs are not committed.** This repository is public, and most publishers
(IEEE, Elsevier, Springer) do not permit redistributing their PDFs. Open-access
papers can be downloaded into `literature/pdfs/` (gitignored) with:

```bash
python literature/fetch_pdfs.py          # open-access only; prints links for the rest
```

`references.bib` in this folder holds the same list as BibTeX for the report.

Access column: **OA** = free to read and redistribute (arXiv, PLOS, Frontiers, MDPI,
Scientific Data, PMC); **free** = readable at no cost but not redistributable;
**paywalled** = subscription needed (use the institute library or the DOI link).

---

## 1. Base paper — reproduced

| Reference | Access | What we used it for |
|---|---|---|
| Wagh, N. & Varatharajah, Y. (2020). **EEG-GCNN: Augmenting Electroencephalogram-based Neurological Disease Diagnosis using a Domain-guided Graph Convolutional Neural Network.** *Proceedings of Machine Learning Research* 136:367–378 (ML4H @ NeurIPS 2020). [arXiv:2011.12107](https://arxiv.org/abs/2011.12107) · [code](https://github.com/neerajwagh/eeg-gcnn) · [resources: figshare 13251452](https://doi.org/10.6084/m9.figshare.13251452) | OA | **The base paper.** Its graph design (electrodes as nodes, coherence + electrode distance as edges) is the starting point of our whole pipeline. Reproduced exactly from the released checkpoints: AUC 0.871 (0.001). |

## 2. Main dataset

| Reference | Access | What we used it for |
|---|---|---|
| Park, S. M., Jeong, B., Oh, D. Y., Choi, C.-H., Jung, H. Y., Lee, J.-Y., Lee, D. & Choi, J.-S. (2021). **Identification of Major Psychiatric Disorders From Resting-State Electroencephalography Using a Machine Learning Approach.** *Frontiers in Psychiatry* 12:707581. [doi:10.3389/fpsyt.2021.707581](https://doi.org/10.3389/fpsyt.2021.707581) | OA | The 945-subject psychiatric corpus behind all five per-disorder GNNs: 19 electrodes, band powers and coherence. Also the source of the published binary accuracies we could not reproduce (93.8% for schizophrenia vs our 0.731 with logistic regression). |

## 3. Papers evaluated during selection, and why

| Reference | Access | What we used it for |
|---|---|---|
| Aung, H. W., Li, J. J., An, Y. & Su, S. W. (2025). **A Real-Time Framework for EEG Signal Decoding With Graph Neural Networks and Reinforcement Learning.** *IEEE TNNLS*. [doi:10.1109/TNNLS.2025.3558171](https://doi.org/10.1109/TNNLS.2025.3558171) | paywalled | **Rejected as base paper.** Reinforcement learning is the method, not an add-on, and the task is motor-imagery BCI decoding, not disorder identification. |
| Aung, H. W., Li, J. J., Shi, B., An, Y. & Su, S. W. (2025). **EEG_GLT-Net: Optimising EEG graphs for real-time motor imagery signals classification.** *Biomedical Signal Processing and Control* 101:107458. [doi:10.1016/j.bspc.2024.107458](https://doi.org/10.1016/j.bspc.2024.107458) | paywalled | The graph-construction method the rejected paper builds on; shows what remains if RL is stripped out. |
| Hazra, S. & Ghosh, S. (2025). **Bridging Accuracy and Explainability in EEG-based Graph Attention Network for Depression Detection (ExPANet).** [arXiv:2511.05537](https://arxiv.org/abs/2511.05537) | OA | Indian-authored, closest topical fit; edge-gated attention on PLV graphs. Kept as the target to beat, not the baseline: code unreleased and 97.5% reported accuracy leaves no headroom. Its edge-gating motivated our `edge_dim` design. |
| Wang, Y., Zheng, W., Li, Y. & Yang, H. (2024/2025). **A Hybrid Graph Neural Network for Enhanced EEG-Based Depression Detection.** [arXiv:2410.18103](https://arxiv.org/abs/2410.18103); IJCNN 2025, [doi:10.1109/IJCNN64981.2025.11227546](https://doi.org/10.1109/IJCNN64981.2025.11227546) | OA (arXiv) | Common vs individualised graphs on MODMA and HUSM; considered as a baseline, rejected because no code was released. |
| Abadal, S., Galván, P., Mármol, A., Mammone, N., Ieracitano, C., Lo Giudice, M., Salvini, A. & Morabito, F. C. (2025). **Graph neural networks for electroencephalogram analysis: Alzheimer's disease and epilepsy use cases.** *Neural Networks* 181:106792. [doi:10.1016/j.neunet.2024.106792](https://doi.org/10.1016/j.neunet.2024.106792) | paywalled | The model for how to do multi-class properly: ternary classification kept *within* each coherent population instead of pooling diseases across corpora. |
| Tang, S., Dunnmon, J. A., Saab, K., Zhang, X., Huang, Q., Dubost, F., Rubin, D. L. & Lee-Messer, C. (2022). **Self-Supervised Graph Neural Networks for Improved Electroencephalographic Seizure Analysis.** ICLR 2022. [arXiv:2104.08336](https://arxiv.org/abs/2104.08336) · [code](https://github.com/tsy935/eeg-gnn-ssl) | OA | Source of the distance-vs-correlation graph comparison and of per-sample (dynamic) adjacency; showed that adaptivity and attention are separable ideas. |
| Jain, S. & Srivastava, R. (2025). **LG-TriCapsNet: A lightweight graph capsule framework with nearest neighbor graphs for multi-disease EEG classification.** *Computers in Biology and Medicine* 196:110567. [doi:10.1016/j.compbiomed.2025.110567](https://doi.org/10.1016/j.compbiomed.2025.110567) | paywalled | Cited as the confound to avoid: 98.34% "multi-disease" accuracy over five pooled corpora, where the model can separate datasets rather than diseases. Its announced code repository is empty. |
| Klepl, D., Wu, M. & He, F. (2024). **Graph Neural Network-based EEG Classification: A Survey.** *IEEE TNSRE* 32:493–503. [doi:10.1109/TNSRE.2024.3355750](https://doi.org/10.1109/TNSRE.2024.3355750) · [arXiv:2310.02152](https://arxiv.org/abs/2310.02152) | OA (arXiv) | Survey used for the related-work framing: which graph constructions and conv types are standard in EEG. |
| Ahmed, Z., Wali, A., Shahid, S. et al. (2024). **Psychiatric disorders from EEG signals through deep learning models.** *IBRO Neuroscience Reports* 17:300–310. [doi:10.1016/j.ibneur.2024.09.003](https://doi.org/10.1016/j.ibneur.2024.09.003) | OA | Deep learning on the same Park corpus, binary per disorder, 96–99% accuracy. Cited as the inflated-number pattern our clean evaluation does not reproduce. |

## 4. Datasets we used

| Reference | Access | Role |
|---|---|---|
| Obeid, I. & Picone, J. (2016). **The Temple University Hospital EEG Data Corpus.** *Frontiers in Neuroscience* 10:196. [doi:10.3389/fnins.2016.00196](https://doi.org/10.3389/fnins.2016.00196) | OA (data: registration) | The "diseased" half of the base-paper corpus. |
| Babayan, A., Erbey, M., Kumral, D. et al. (2019). **A mind-brain-body dataset of MRI, EEG, cognition, emotion, and peripheral physiology in young and old adults (MPI LEMON).** *Scientific Data* 6:180308. [doi:10.1038/sdata.2018.308](https://doi.org/10.1038/sdata.2018.308) | OA | The "healthy" half of the base-paper corpus. Together with TUH it creates the site/label confound we report. |
| Bagherzadeh, S., Norouzi, M., Farokhshad, F. et al. (2026). **Atieh Schizophrenia EEG (ASEEG), a novel high-quality dataset.** *Scientific Data*. [doi:10.1038/s41597-026-07968-9](https://doi.org/10.1038/s41597-026-07968-9) · data: [Zenodo 18029536](https://doi.org/10.5281/zenodo.18029536) (CC-BY) | OA | 51 schizophrenia + 50 controls, Iran. Channel order was undocumented and had to be recovered from the data (`infer_channel_order.py`). |
| Olejarczyk, E. & Jernajczyk, W. (2017). **Graph-based analysis of brain connectivity in schizophrenia.** *PLOS ONE* 12(11):e0188629. [doi:10.1371/journal.pone.0188629](https://doi.org/10.1371/journal.pone.0188629) · data: [RepOD 10.18150/repod.0107441](https://doi.org/10.18150/repod.0107441) | OA | 14 schizophrenia + 14 controls, Warsaw. The second hospital in the schizophrenia transfer test. |
| Mumtaz, W., Xia, L., Mohd Yasin, M. A. et al. (2017). **A wavelet-based technique to predict treatment outcome for Major Depressive Disorder.** *PLOS ONE* 12(2):e0171409. [doi:10.1371/journal.pone.0171409](https://doi.org/10.1371/journal.pone.0171409) · data: [figshare 4244171](https://doi.org/10.6084/m9.figshare.4244171) (CC-BY) | OA | 30 MDD + 28 controls, Malaysia (6 of 64 eyes-closed files are dead links). The raw-EEG depression track. |
| Miltiadous, A., Tzimourta, K. D., Afrantou, T. et al. (2023). **A Dataset of Scalp EEG Recordings of Alzheimer's Disease, Frontotemporal Dementia and Healthy Subjects from Routine EEG.** *Data* 8(6):95. [doi:10.3390/data8060095](https://doi.org/10.3390/data8060095) · data: [OpenNeuro ds004504](https://doi.org/10.18112/openneuro.ds004504.v1.0.8) (CC0) | OA | 36 AD + 23 FTD + 29 controls, Thessaloniki. Both dementia models, and the only fair between-disease test (AD vs FTD). |
| Anjum, M. F., Dasgupta, S., Mudumbai, R. et al. (2020). **Linear predictive coding distinguishes spectral EEG features of Parkinson's disease.** *Parkinsonism & Related Disorders* 79:79–85. [doi:10.1016/j.parkreldis.2020.08.001](https://doi.org/10.1016/j.parkreldis.2020.08.001) · data: [OpenNeuro ds004584](https://doi.org/10.18112/openneuro.ds004584.v1.0.0) (CC0) | paywalled (data CC0) | 100 PD + 49 controls, University of Iowa. Records against Pz, which we reconstruct. |
| Cavanagh, J. F., Kumar, P., Mueller, A. A. et al. (2018). **Diminished EEG habituation to novel events effectively classifies Parkinson's patients.** *Clinical Neurophysiology* 129(2):409–418. [doi:10.1016/j.clinph.2017.11.023](https://doi.org/10.1016/j.clinph.2017.11.023) · data: [OpenNeuro ds003490](https://doi.org/10.18112/openneuro.ds003490.v1.1.0) (CC0) | paywalled (data CC0) | 25 PD + 25 controls, University of New Mexico. Only the instructed eyes-open minute is used; patients' OFF-medication session. |
| George, J. S., Strunk, J., Mak-McCully, R. et al. (2013). **Dopaminergic therapy in Parkinson's disease decreases cortical beta band coherence in the resting state.** *NeuroImage: Clinical* 3:261–270. [doi:10.1016/j.nicl.2013.07.013](https://doi.org/10.1016/j.nicl.2013.07.013) · data: [OpenNeuro ds002778](https://doi.org/10.18112/openneuro.ds002778.v1.0.5) (CC0) | OA | 15 PD + 16 controls, UC San Diego (curated by A. Rockhill). **The curators ask to be contacted before a manuscript using this data is submitted.** |

## 5. Datasets evaluated but not used

| Reference | Access | Why not |
|---|---|---|
| van Dijk, H., van Wingen, G., Denys, D. et al. (2022). **The two decades brainclinics research archive for insights in neurophysiology (TDBRAIN).** *Scientific Data* 9:333. [doi:10.1038/s41597-022-01409-z](https://doi.org/10.1038/s41597-022-01409-z) | OA (data: permission) | 1,274 patients with raw EEG, the best route to more data per disorder; needs a permission request, so it is the recommended next step rather than something we could use now. |
| Cai, H., Yuan, Z., Gao, Y. et al. (2022). **A multi-modal open dataset for mental-disorder analysis (MODMA).** *Scientific Data* 9:178. [doi:10.1038/s41597-022-01211-x](https://doi.org/10.1038/s41597-022-01211-x) | OA (data: application) | Depression corpus used by several papers we compare against; access requires an application. |
| Kim, M., Youn, Y. C. & Paik, J. (2023). **Deep learning-based EEG analysis to classify normal, mild cognitive impairment, and dementia (CAUEEG).** *NeuroImage* 272:120054. [doi:10.1016/j.neuroimage.2023.120054](https://doi.org/10.1016/j.neuroimage.2023.120054) | paywalled (data: request) | 1,155 subjects with dementia/MCI labels: the second hospital that would let us test whether the Alzheimer's result transfers. Access request pending. |
| Sareen, E., Singh, L., Gupta, A., Varkey, B. & Achary, K. **EEG dataset of individuals with intellectual and developmental disorder and healthy controls under rest and music stimuli.** Mendeley Data. [doi:10.17632/fshy54ypyh.2](https://doi.org/10.17632/fshy54ypyh.2) (CC-BY) | OA | The only openly available Indian clinical EEG set we found (IIIT-Delhi). Too small to train on (7 + 7 subjects, 14 channels); kept as a possible cross-dataset generalisation test. |

## 6. Methods and models

| Reference | Access | What we used it for |
|---|---|---|
| Kipf, T. N. & Welling, M. (2017). **Semi-Supervised Classification with Graph Convolutional Networks.** ICLR 2017. [arXiv:1609.02907](https://arxiv.org/abs/1609.02907) | OA | GCN — the base paper's convolution and our baseline (`--model gcn`). |
| Veličković, P., Cucurull, G., Casanova, A. et al. (2018). **Graph Attention Networks.** ICLR 2018. [arXiv:1710.10903](https://arxiv.org/abs/1710.10903) | OA | GAT — attention over neighbours; background for the model we chose. |
| Brody, S., Alon, U. & Yahav, E. (2022). **How Attentive are Graph Attention Networks?** ICLR 2022. [arXiv:2105.14491](https://arxiv.org/abs/2105.14491) | OA | **GATv2 — our final model.** Fixes GAT's static attention, and accepts edge features so attention can read all six band coherences. |
| Defferrard, M., Bresson, X. & Vandergheynst, P. (2016). **Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering.** NeurIPS 2016. [arXiv:1606.09375](https://arxiv.org/abs/1606.09375) | OA | ChebConv — the K-hop candidate in our tuning grid. |
| Welch, P. D. (1967). **The use of fast Fourier transform for the estimation of power spectra.** *IEEE Trans. Audio Electroacoust.* 15(2):70–73. [doi:10.1109/TAU.1967.1161901](https://doi.org/10.1109/TAU.1967.1161901) | paywalled | Welch's method: the 2 s Hann windows with 50% overlap behind our band powers and coherence. |
| Platt, J. C. (1999). **Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods.** In *Advances in Large Margin Classifiers*, MIT Press, 61–74. | free (PDF online) | Platt scaling: how we calibrated the over-confident model outputs. |
| Hanley, J. A. & McNeil, B. J. (1982). **The meaning and use of the area under a receiver operating characteristic (ROC) curve.** *Radiology* 143(1):29–36. [doi:10.1148/radiology.143.1.7063747](https://doi.org/10.1148/radiology.143.1.7063747) | paywalled | AUC and its standard error; used to size confidence intervals before we switched to bootstrap. |

## 7. Methodology and critique

| Reference | Access | What we used it for |
|---|---|---|
| Brookshire, G., Kasper, J., Blauch, N. M. et al. (2024). **Data leakage in deep learning studies of translational EEG.** *Frontiers in Neuroscience* 18:1373515. [doi:10.3389/fnins.2024.1373515](https://doi.org/10.3389/fnins.2024.1373515) | OA | The evidence behind our subject-level splitting rule, and our explanation of why published EEG accuracies often do not reproduce. |
| Jain, S. & Wallace, B. C. (2019). **Attention is not Explanation.** NAACL-HLT 2019. [doi:10.18653/v1/N19-1357](https://doi.org/10.18653/v1/N19-1357) | OA | Why we treat attention weights as a hypothesis to test, not as an explanation. |
| Wiegreffe, S. & Pinter, Y. (2019). **Attention is not not Explanation.** EMNLP-IJCNLP 2019. [doi:10.18653/v1/D19-1002](https://doi.org/10.18653/v1/D19-1002) | OA | The counter-argument; both sides are cited in the interpretability discussion. |

## 8. Software

| Reference | Access | What we used it for |
|---|---|---|
| Fey, M. & Lenssen, J. E. (2019). **Fast Graph Representation Learning with PyTorch Geometric.** ICLR 2019 Workshop RLGM. [arXiv:1903.02428](https://arxiv.org/abs/1903.02428) | OA | PyTorch Geometric: `GCNConv`, `GATv2Conv`, `ChebConv`, batching of graphs. |
| Paszke, A., Gross, S., Massa, F. et al. (2019). **PyTorch: An Imperative Style, High-Performance Deep Learning Library.** NeurIPS 2019. [arXiv:1912.01703](https://arxiv.org/abs/1912.01703) | OA | Training framework. |
| Gramfort, A., Luessi, M., Larson, E. et al. (2013). **MEG and EEG data analysis with MNE-Python.** *Frontiers in Neuroscience* 7:267. [doi:10.3389/fnins.2013.00267](https://doi.org/10.3389/fnins.2013.00267) | OA | Reading EDF/BDF/EEGLAB files, filtering, notch, resampling in the raw-EEG pipeline. |
| Pedregosa, F., Varoquaux, G., Gramfort, A. et al. (2011). **Scikit-learn: Machine Learning in Python.** *JMLR* 12:2825–2830. [link](https://jmlr.org/papers/v12/pedregosa11a.html) | OA | Splitting, metrics, logistic-regression baselines, Platt scaling. |
