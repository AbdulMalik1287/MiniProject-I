#!/usr/bin/env python3
"""Download the neurological EEG datasets used by raw_eeg.py (all CC0, OpenNeuro).

Only the files the pipeline reads are fetched, each size-checked against the
OpenNeuro S3 listing and resumable:

    ds004504  Alzheimer's / frontotemporal dementia / controls, Thessaloniki, eyes closed
              raw recordings only (the preprocessed derivatives are skipped)
    ds004584  Parkinson's / controls, University of Iowa, eyes open
    ds003490  Parkinson's / controls, University of New Mexico: controls' single
              session, and each patient's OFF-medication session (from participants.tsv)
    ds002778  Parkinson's / controls, UC San Diego: controls, and patients' ses-off

    python fetch_neuro.py                      # all four, into data/external/<dataset>/
    python fetch_neuro.py --datasets ds004504
"""
import argparse
import concurrent.futures as cf
import csv
import io
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

S3 = "https://s3.amazonaws.com/openneuro.org"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
ROOT = Path(__file__).resolve().parents[3] / "data" / "external"
if not ROOT.exists():  # running from the main checkout rather than a worktree
    ROOT = Path(__file__).resolve().parent / "data" / "external"


def listing(prefix):
    token, out = None, {}
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if token:
            q["continuation-token"] = token
        root = ET.fromstring(urllib.request.urlopen(f"{S3}?{urllib.parse.urlencode(q)}", timeout=60).read())
        for c in root.findall("s3:Contents", NS):
            out[c.find("s3:Key", NS).text] = int(c.find("s3:Size", NS).text)
        if root.find("s3:IsTruncated", NS).text != "true":
            return out
        token = root.find("s3:NextContinuationToken", NS).text


def participants(ds):
    text = urllib.request.urlopen(f"{S3}/{ds}/participants.tsv", timeout=60).read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text), delimiter="\t"))


def wanted(ds, files):
    """Keys of `files` the pipeline needs for dataset `ds`."""
    raw = {k: s for k, s in files.items() if "/derivatives/" not in k}
    if ds == "ds004504":
        return [k for k in raw if k.endswith("_eeg.set")]
    if ds == "ds004584":
        return [k for k in raw if k.endswith(("_eeg.set", "_eeg.fdt"))]
    if ds == "ds003490":
        keep = []
        for p in participants(ds):
            if p["Group"] == "CTL":
                ses = "ses-01"
            else:  # patients came twice; take whichever session was OFF medication
                ses = "ses-01" if p["sess1_Med"] == "OFF" else "ses-02" if p["sess2_Med"] == "OFF" else None
            if ses is None:
                continue
            base = f"{ds}/{p['participant_id']}/{ses}/eeg/{p['participant_id']}_{ses}_task-Rest"
            keep += [k for k in (f"{base}_eeg.set", f"{base}_eeg.fdt", f"{base}_events.tsv") if k in raw]
        return keep
    if ds == "ds002778":
        return [k for k in raw if k.endswith("_eeg.bdf") and ("/ses-hc/" in k or "/ses-off/" in k)]
    raise ValueError(ds)


def fetch(key, size, attempts=8):
    dest = ROOT / key
    dest.parent.mkdir(parents=True, exist_ok=True)
    for i in range(attempts):
        have = dest.stat().st_size if dest.exists() else 0
        if have == size:
            return key, True
        if have > size:
            dest.unlink()
            have = 0
        req = urllib.request.Request(f"{S3}/{urllib.parse.quote(key)}", headers={"Range": f"bytes={have}-"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "ab" if have else "wb") as f:
                if have and r.status != 206:
                    f.truncate(0)
                while chunk := r.read(1 << 20):
                    f.write(chunk)
        except OSError as e:
            print(f"  retry {i + 1} {key}: {e}", flush=True)
            time.sleep(2 + 2 * i)
    return key, dest.exists() and dest.stat().st_size == size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["ds004504", "ds004584", "ds003490", "ds002778"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    for ds in args.datasets:
        files = listing(f"{ds}/")
        keys = wanted(ds, files)
        for meta in ("participants.tsv", "dataset_description.json", "README"):
            if f"{ds}/{meta}" in files:
                keys.append(f"{ds}/{meta}")
        total = sum(files[k] for k in keys)
        print(f"[{ds}] {len(keys)} files, {total / 1e9:.2f} GB", flush=True)
        failed = []
        with cf.ThreadPoolExecutor(args.workers) as pool:
            for key, ok in pool.map(lambda k: fetch(k, files[k]), keys):
                if not ok:
                    failed.append(key)
        print(f"[{ds}] done, failed: {failed if failed else 'none'}", flush=True)
        if failed:
            sys.exit(1)
    print("NEURO_FETCH_DONE")


if __name__ == "__main__":
    main()
