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
