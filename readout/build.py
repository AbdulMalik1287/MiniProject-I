#!/usr/bin/env python3
"""Build docs/eeg_readout.html - the per-disorder model readout page.

Fills readout/template.html with the held-out test predictions and the measured
reliability of each model, read straight from results/per_disorder, so the page
cannot show a number the evaluation did not produce.

    python readout/build.py
"""
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import per_disorder as D  # noqa: E402

RES = ROOT / "results" / "per_disorder"
START_SUBJECT = "730"  # an anxiety patient whose anxiety bar is the lowest of five - the key caveat, shown first
SITE = {"ds004504": "Thessaloniki", "ds004584": "Iowa", "ds003490": "New Mexico", "ds002778": "San Diego"}
GROUP = {"alzheimers": "AD", "ftd": "FTD", "parkinsons": "PD", "control": "controls"}
NEURO = [  # task, heading, context line
    ("alzheimers", "Alzheimer's vs healthy", "Thessaloniki · eyes closed"),
    ("ftd", "Frontotemporal dementia vs healthy", "Thessaloniki · eyes closed"),
    ("alzheimers_vs_ftd", "Alzheimer's vs frontotemporal dementia",
     "both from one hospital, so a fair test of telling two diseases apart"),
    ("parkinsons", "Parkinson's vs healthy",
     "three US hospitals · eyes open · off medication where recorded"),
]


def neuro_groups():
    """Within-hospital, pooled and leave-one-hospital-out results for each neurological task."""
    groups = []
    for task, title, detail in NEURO:
        f = ROOT / "results" / "raw" / f"raw_{task}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        many = len(r["datasets"]) > 2
        rows = []
        for key, v in r.items():
            if not (isinstance(v, dict) and "auc" in v):
                continue
            kind, _, rest = key.partition(":")
            if kind == "pooled" and rest:
                continue  # per-site slices of the pooled model live in the report
            if kind == "transfer" and many:
                continue  # leave-one-hospital-out stands in for six pairwise transfers
            if kind == "within":
                label = f"within {SITE[rest]}"
            elif kind == "pooled":
                label = f"pooled across {len(r['datasets'])} hospitals"
            elif kind == "loso":
                label = f"trained elsewhere, tested at {SITE[rest]}"
            else:
                a, b = rest.split("->")
                label = f"trained {SITE[a]}, tested {SITE[b]}"
            rows.append({"label": label, "auc": v["auc"], "ci": v["ci"],
                         "n": f"{v['n_patients']} {GROUP[r['positive']]} / {v['n_controls']} {GROUP[r['negative']]}"})
        groups.append({"title": title, "detail": detail, "rows": rows})
    return groups


def main():
    ev = json.loads((RES / "evaluate.json").read_text())
    preds = json.loads((RES / "test_predictions.json").read_text())
    if "combined" not in ev:
        raise SystemExit("evaluate.json has no combined score - rerun `per_disorder.py evaluate`")

    models = []
    for d in D.DISORDERS:
        v = ev["per_disorder"][d]
        models.append({"key": D.SHORT[d], "auc": v["auc"], "ci": v["auc_ci"],
                       "other": v["auc_vs_other_disorders"], "other_ci": v["auc_vs_other_ci"],
                       "any": v["auc_any_patient"], "any_ci": v["auc_any_patient_ci"],
                       "n_cases": v["n_pos"]})
    c = ev["combined"]["mean"]
    n_ctl = sum(p["diagnosis"] == D.HEALTHY for p in preds)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        commit = "unknown"

    data = {
        "threshold": 0.5,
        "start": START_SUBJECT if any(p["id"] == START_SUBJECT for p in preds) else preds[0]["id"],
        "generated": date.today().isoformat(), "commit": commit,
        "models": models,
        "combined": {"auc": c["auc"], "auc_ci": c["auc_ci"], "sensitivity": c["sensitivity"],
                     "specificity": c["specificity"], "n_patients": len(preds) - n_ctl, "n_controls": n_ctl},
        "subjects": [{"id": p["id"], "diagnosis": p["diagnosis"], "p": p["p"], "combined": p["combined"]}
                     for p in sorted(preds, key=lambda p: int(p["id"]))],
        "neuro": neuro_groups(),
    }
    assert len(data["subjects"]) == ev["n_test"], "predictions and evaluation disagree on the test split"

    template = (ROOT / "readout" / "template.html").read_text(encoding="utf-8")
    if template.count("/*__DATA__*/") != 1:
        raise SystemExit("template must contain exactly one /*__DATA__*/ placeholder")
    html = template.replace("/*__DATA__*/", json.dumps(data, separators=(",", ":")))
    out = ROOT / "docs" / "eeg_readout.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({len(html) / 1024:.0f} KB, {len(data['subjects'])} subjects)")


if __name__ == "__main__":
    main()
