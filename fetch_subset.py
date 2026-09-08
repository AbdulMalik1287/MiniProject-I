#!/usr/bin/env python3
"""Pull only the needed entries out of the FigShare zip over HTTP range requests.

The full archive is 977 MB, but 1.14 GB of its uncompressed content is
random-forest baseline models we never load. Reading the zip's central directory
and fetching just the entries we want cuts the transfer to roughly 190 MB, which
matters because FigShare serves this at about 100 KB/s.

    python fetch_subset.py --out ~/eeg/data
"""
import argparse
import struct
import sys
import urllib.request
import zlib
from pathlib import Path

URL = "https://ndownloader.figshare.com/files/37681734"

WANTED = [
    "psd_features_data_X",
    "labels_y",
    "master_metadata_index.csv",
    "spec_coh_values.npy",
    "standard_1010.tsv.txt",
]
WANTED_PREFIX = "psd_shallow_eeg_gcnn/"  # the 10 checkpoints, ~30 KB each


def fetch(rng, timeout=120):
    req = urllib.request.Request(URL, headers={"Range": f"bytes={rng}"})
    with urllib.request.urlopen(req, timeout=timeout) as f:
        if f.status != 206:
            raise RuntimeError(f"server ignored Range (status {f.status})")
        return f.read(), f.headers.get("Content-Range")


def central_directory():
    """Returns (total_size, [(name, offset, comp_size, uncomp_size, method), ...])."""
    tail, cr = fetch("-200000")
    total = int(cr.split("/")[1])
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise RuntimeError("no end-of-central-directory record in tail")
    n_entries = struct.unpack("<H", tail[i + 10 : i + 12])[0]
    cd_offset = struct.unpack("<I", tail[i + 16 : i + 20])[0]

    span = total - cd_offset
    cd = tail[-span:] if span <= len(tail) else fetch(f"{cd_offset}-{total - 1}")[0]

    entries, p = [], 0
    for _ in range(n_entries):
        if cd[p : p + 4] != b"PK\x01\x02":
            break
        method = struct.unpack("<H", cd[p + 10 : p + 12])[0]
        csz, usz = struct.unpack("<II", cd[p + 20 : p + 28])
        nl, el, cl = struct.unpack("<HHH", cd[p + 28 : p + 34])
        offset = struct.unpack("<I", cd[p + 42 : p + 46])[0]
        name = cd[p + 46 : p + 46 + nl].decode("utf-8", "replace")
        entries.append((name, offset, csz, usz, method))
        p += 46 + nl + el + cl
    return total, entries


def extract(entry, out_dir):
    """Fetch one entry's bytes and inflate. Local header length is re-read because
    its extra field can differ from the central directory's."""
    name, offset, csz, usz, method = entry
    head, _ = fetch(f"{offset}-{offset + 29}")
    if head[:4] != b"PK\x03\x04":
        raise RuntimeError(f"{name}: bad local header signature")
    nl, el = struct.unpack("<HH", head[26:30])
    start = offset + 30 + nl + el
    raw, _ = fetch(f"{start}-{start + csz - 1}", timeout=1800)
    if len(raw) != csz:
        raise RuntimeError(f"{name}: got {len(raw)} bytes, expected {csz}")

    if method == 0:
        data = raw
    elif method == 8:
        data = zlib.decompress(raw, -zlib.MAX_WBITS)
    else:
        raise RuntimeError(f"{name}: unsupported compression method {method}")
    if len(data) != usz:
        raise RuntimeError(f"{name}: inflated to {len(data)}, expected {usz}")

    dest = Path(out_dir) / Path(name).name  # flatten; train.py expects a flat dir
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return dest, len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/eeg/data")
    ap.add_argument("--probe", action="store_true",
                    help="fetch only the smallest entry, to verify range support")
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    total, entries = central_directory()
    by_name = {e[0]: e for e in entries}
    todo = [by_name[n] for n in WANTED if n in by_name]
    todo += [e for e in entries if e[0].startswith(WANTED_PREFIX)]
    missing = [n for n in WANTED if n not in by_name]
    if missing:
        print(f"MISSING from archive: {missing}", file=sys.stderr)
        return 1

    if args.probe:
        todo = [min(todo, key=lambda e: e[2])]

    plan = sum(e[2] for e in todo)
    print(f"archive {total/1e6:.0f} MB; fetching {len(todo)} entries, {plan/1e6:.0f} MB compressed")
    for e in todo:
        dest, n = extract(e, out)
        print(f"  ok {dest.name:32s} {n/1e6:8.1f} MB")
    print("SUBSET_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
