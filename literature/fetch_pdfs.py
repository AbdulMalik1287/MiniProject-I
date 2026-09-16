#!/usr/bin/env python3
"""Download the open-access papers into literature/pdfs/ (gitignored).

Only papers that are free to redistribute are fetched: arXiv preprints, PLOS,
Frontiers, MDPI and Scientific Data (all open licences), plus PMC copies where
one exists. Paywalled papers (IEEE, Elsevier, Springer) are printed with their
DOI link so they can be opened through the institute library - their PDFs are
deliberately neither downloaded here nor committed to this public repository.

    python literature/fetch_pdfs.py
    python literature/fetch_pdfs.py --list        # show the plan, download nothing
"""
import argparse
import urllib.error
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent / "pdfs"
UA = {"User-Agent": "MiniProject-I-bibliography/1.0 (mailto:bt24csa046@iiitn.ac.in)"}

# (filename, url) - open access, redistributable
OPEN = [
    ("wagh2020_eeg-gcnn.pdf", "https://arxiv.org/pdf/2011.12107"),
    ("park2021_psychiatric-eeg.pdf", "https://www.frontiersin.org/articles/10.3389/fpsyt.2021.707581/pdf"),
    ("hazra2025_expanet.pdf", "https://arxiv.org/pdf/2511.05537"),
    ("wang2024_hybgnn.pdf", "https://arxiv.org/pdf/2410.18103"),
    ("tang2022_eeg-gnn-ssl.pdf", "https://arxiv.org/pdf/2104.08336"),
    ("klepl2023_gnn-eeg-survey.pdf", "https://arxiv.org/pdf/2310.02152"),
    # Elsevier, MDPI and Nature block direct PDF links to non-browser clients, so
    # the open-access copies are taken from Europe PMC instead.
    ("ahmed2024_psychiatric-deep-learning.pdf", "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11466652/fullTextPDF"),
    ("obeid2016_tuh-corpus.pdf", "https://www.frontiersin.org/articles/10.3389/fnins.2016.00196/pdf"),
    ("olejarczyk2017_schizophrenia-connectivity.pdf", "https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0188629&type=printable"),
    ("mumtaz2017_mdd-wavelet.pdf", "https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0171409&type=printable"),
    ("george2013_pd-beta-coherence.pdf", "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC3814961/fullTextPDF"),
    ("brookshire2024_eeg-data-leakage.pdf", "https://www.frontiersin.org/articles/10.3389/fnins.2024.1373515/pdf"),
    ("gramfort2013_mne-python.pdf", "https://www.frontiersin.org/articles/10.3389/fnins.2013.00267/pdf"),
    ("kipf2017_gcn.pdf", "https://arxiv.org/pdf/1609.02907"),
    ("velickovic2018_gat.pdf", "https://arxiv.org/pdf/1710.10903"),
    ("brody2022_gatv2.pdf", "https://arxiv.org/pdf/2105.14491"),
    ("defferrard2016_chebnet.pdf", "https://arxiv.org/pdf/1606.09375"),
    ("fey2019_pytorch-geometric.pdf", "https://arxiv.org/pdf/1903.02428"),
    ("paszke2019_pytorch.pdf", "https://arxiv.org/pdf/1912.01703"),
    ("jain2019_attention-is-not-explanation.pdf", "https://aclanthology.org/N19-1357.pdf"),
    ("wiegreffe2019_attention-is-not-not-explanation.pdf", "https://aclanthology.org/D19-1002.pdf"),
    ("babayan2019_lemon.pdf", "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6371893/fullTextPDF"),
    ("vandijk2022_tdbrain.pdf", "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9198070/fullTextPDF"),
    ("cai2022_modma.pdf", "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9018722/fullTextPDF"),
    ("pedregosa2011_scikit-learn.pdf", "https://www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf"),
]

# Read through the DOI; not redistributable, so not downloaded here.
PAYWALLED = [
    ("Aung 2025, EEG_RL-Net (rejected base paper)", "https://doi.org/10.1109/TNNLS.2025.3558171"),
    ("Aung 2025, EEG_GLT-Net", "https://doi.org/10.1016/j.bspc.2024.107458"),
    ("Abadal 2025, GNNs for EEG: AD and epilepsy", "https://doi.org/10.1016/j.neunet.2024.106792"),
    ("Jain & Srivastava 2025, LG-TriCapsNet", "https://doi.org/10.1016/j.compbiomed.2025.110567"),
    ("Klepl 2024, GNN EEG survey (journal version)", "https://doi.org/10.1109/TNSRE.2024.3355750"),
    ("Anjum 2020, LPC spectral EEG in Parkinson's", "https://doi.org/10.1016/j.parkreldis.2020.08.001"),
    ("Cavanagh 2018, EEG habituation in Parkinson's", "https://doi.org/10.1016/j.clinph.2017.11.023"),
    ("Kim 2023, CAUEEG", "https://doi.org/10.1016/j.neuroimage.2023.120054"),
    ("Welch 1967, FFT power spectra", "https://doi.org/10.1109/TAU.1967.1161901"),
    ("Hanley & McNeil 1982, meaning and use of AUC", "https://doi.org/10.1148/radiology.143.1.7063747"),
    ("Platt 1999, probabilistic outputs (book chapter)", "https://en.wikipedia.org/wiki/Platt_scaling"),
    ("Bagherzadeh 2026, ASEEG (open, PDF link varies)", "https://doi.org/10.1038/s41597-026-07968-9"),
    ("Wang 2025, HybGNN (IJCNN version; arXiv copy is fetched)", "https://doi.org/10.1109/IJCNN64981.2025.11227546"),
    ("Miltiadous 2023, AD/FTD dataset (open access, MDPI blocks scripted downloads)",
     "https://doi.org/10.3390/data8060095"),
]


def fetch(name, url):
    dest = OUT / name
    if dest.exists() and dest.stat().st_size > 20_000:
        return "skip", dest.stat().st_size
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
            data = r.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        return f"FAIL ({e})", 0
    if not data.startswith(b"%PDF"):
        return "FAIL (not a PDF; publisher blocked the direct link)", 0
    dest.write_bytes(data)
    return "ok", len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print the plan without downloading")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"{len(OPEN)} open-access papers -> {OUT}")
    if args.list:
        for name, url in OPEN:
            print(f"  {name:<52} {url}")
    else:
        failed = []
        for name, url in OPEN:
            status, size = fetch(name, url)
            print(f"  {status:<12} {name:<52} {size / 1e6:.1f} MB" if size else f"  {status:<12} {name}")
            if status.startswith("FAIL"):
                failed.append((name, url))
        print(f"\ndownloaded or already present: {len(OPEN) - len(failed)}/{len(OPEN)}")
        for name, url in failed:
            print(f"  open by hand: {name} -> {url}")

    print(f"\n{len(PAYWALLED)} paywalled papers - open through the institute library:")
    for label, url in PAYWALLED:
        print(f"  {label:<52} {url}")


if __name__ == "__main__":
    main()
