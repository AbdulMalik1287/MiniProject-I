"""GCN vs GATv2 on the same held-out Park test set. Run from the repo root."""
import json
from pathlib import Path

G = json.loads(Path("results/per_disorder_gcn/evaluate.json").read_text())
A = json.loads(Path("results/per_disorder/evaluate.json").read_text())

print(f"{'disorder':<36} {'GCN AUC [95% CI]':<20} {'GATv2 AUC [95% CI]':<20} {'diff':>6}   "
      f"{'vs other: GCN':<14} {'GATv2'}")
for d, a in A["per_disorder"].items():
    g = G["per_disorder"][d]
    print(f"{d:<36} {g['auc']:.3f} [{g['auc_ci'][0]:.2f}, {g['auc_ci'][1]:.2f}]   "
          f"{a['auc']:.3f} [{a['auc_ci'][0]:.2f}, {a['auc_ci'][1]:.2f}]   {g['auc'] - a['auc']:+.3f}   "
          f"{g['auc_vs_other_disorders']:.3f}          {a['auc_vs_other_disorders']:.3f}")
gm, am = G["combined"]["mean"], A["combined"]["mean"]
print(f"\n{'combined (mean of five)':<36} {gm['auc']:.3f} [{gm['auc_ci'][0]:.2f}, {gm['auc_ci'][1]:.2f}]   "
      f"{am['auc']:.3f} [{am['auc_ci'][0]:.2f}, {am['auc_ci'][1]:.2f}]   {gm['auc'] - am['auc']:+.3f}")
print(f"\nraw softmax saturation before calibration: GCN {G['saturated_raw']:.0%}  GATv2 {A['saturated_raw']:.0%}")
