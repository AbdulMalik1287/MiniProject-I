"""Secondary analysis: two hard tasks re-asked at clinically sharper contrasts.

Cut-offs come from published clinical bands, fixed before running, not tuned on
our results. Each task runs ONCE through the same six-member ensemble as the
main results. These answer easier questions than the main tasks and must be
reported as such, never in place of them.

    unm_moderate_severe   ds003478: BDI-II >= 20 (moderate/severe band, Beck 1996)
                          vs the main task's controls (BDI < 7); the mild 13-19 band is dropped
    telaviv_moderate_pd   ds007526: MDS-UPDRS III >= 33 (moderate motor severity,
                          Martinez-Martin et al. 2015) vs all controls

    python sharpened.py --out ~/eeg/raw2
"""
import argparse
import csv
import json
from pathlib import Path

import torch

import ensemble as EN
import raw_eeg as R


def rows(ds):
    with open(R.ROOT / ds / "participants.tsv", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def subsets():
    unm = {f"ds003478-{p['participant_id']}" for p in rows("ds003478")
           if p["BDI"].isdigit() and (int(p["BDI"]) >= 20 or int(p["BDI"]) < 7)}
    ta = set()
    for p in rows("ds007526"):
        u = p["updrs_part_iii"]
        if p["group"] == "HC" or (u not in ("", "n/a") and float(u) >= 33):
            ta.add(f"ds007526-{p['participant_id']}")
    return {"unm_moderate_severe": ("mood", "ds003478", unm),
            "telaviv_moderate_pd": ("parkinsons", "ds007526", ta)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/eeg/raw2")
    ap.add_argument("--tune", default="results/per_disorder/tune.json")
    ap.add_argument("--save", default="results/sharpened.json")
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    cfg = {**json.loads(Path(args.tune).read_text())["best"]["config"], "epochs": R.RAW_EPOCHS}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = {}
    for name, (task, ds, keep) in subsets().items():
        res, n = EN.run(task, ds, out, cfg, device, subjects=keep)
        report[name] = {"task": task, "dataset": ds, "n": n, **res}
        print(f"{name:<22} n={n:<4}" + "".join(f"  {k} {v['auc']:.3f}" for k, v in res.items()), flush=True)
        Path(args.save).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
