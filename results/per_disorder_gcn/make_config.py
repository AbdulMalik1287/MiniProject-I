"""Pin the best GCN config from the shared tuning grid so GCN and GATv2 can be
compared on the same held-out test set. Run from the repo root:

    python results/per_disorder_gcn/make_config.py
"""
import json
import shutil
from pathlib import Path

SRC = Path("results/per_disorder")
DST = Path("results/per_disorder_gcn")

shutil.copyfile(SRC / "split.json", DST / "split.json")  # identical split, by construction
t = json.loads((SRC / "tune.json").read_text())
best = max((r for r in t["results"] if r["config"]["model"] == "gcn"), key=lambda r: r["mean_auc"])
assert best["grid_index"] == 3, best["grid_index"]
(DST / "tune.json").write_text(json.dumps({
    "results": t["results"], "best": best,
    "note": "GCN counterpart to per_disorder/: same split, same grid, best restricted to "
            "model=gcn (grid 3, mean CV AUC 0.665) so GCN and GATv2 can be compared on the "
            "held-out test set."}, indent=2))
print("best ->", best["config"], round(best["mean_auc"], 3))
