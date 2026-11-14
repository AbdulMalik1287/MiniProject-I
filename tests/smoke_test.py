"""Smoke test: every module imports and every built-in self-check passes. No datasets needed.

    python tests/smoke_test.py
"""
import importlib
import pkgutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from anywhere

import eeg  # noqa: E402

failed = []


def check(name, fn):
    try:
        fn()
        print(f"ok    {name}")
    except Exception as e:  # report every failure, not just the first
        failed.append(name)
        print(f"FAIL  {name}: {type(e).__name__}: {e}")


for m in pkgutil.walk_packages(eeg.__path__, "eeg."):
    check(f"import {m.name}", lambda n=m.name: importlib.import_module(n))

from eeg.data import park, prep_v2, raw_eeg  # noqa: E402
from eeg.experiments import explain_final, final_system, pooled_deep  # noqa: E402
from eeg.models import train  # noqa: E402

check("models.train selfcheck (EEG-GCNN graph + checkpoint port)", train.selfcheck)
check("data.park selfcheck", park.selfcheck)
check("data.raw_eeg selfcheck (pipeline on synthetic EEG)", raw_eeg.selfcheck)
check("data.prep_v2 selfcheck (wPLI, 1/f power, window rule)", prep_v2.selfcheck)


def _loho():
    d = {"site": np.array(["a", "a", "b", "c", "b"]), "ys": np.array([0, 1, 0, 1, 1]), "u": np.arange(5)}
    for tr, te in pooled_deep.folds(d, loho=True):
        assert len(set(d["site"][te])) == 1 and not set(d["site"][tr]) & set(d["site"][te])
    w = pooled_deep.group_w(["a", "a", "a", "b"], [0, 0, 1, 1])
    assert np.allclose([w[:2].sum(), w[2], w[3]], w[2])  # equal mass per site x label group


def _calibration():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = y + rng.normal(0, 0.8, 200)
    d = {"u": np.arange(200), "ys": y, "site": np.array(["a"] * 200)}
    p = final_system.calibrate(s, y, pooled_deep.folds(d))
    c = final_system.coverage_table(y, p)
    assert c["50%"]["accuracy"] >= c["100%"]["accuracy"] and final_system.ece(y, p) < 0.1


def _cohen():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(500), np.zeros(500)].astype(int)
    assert abs(explain_final.cohen_d(rng.normal(size=(1000, 1)) + y[:, None] * 0.8, y)[0] - 0.8) < 0.15


check("experiments.pooled_deep folds + group weights", _loho)
check("experiments.final_system calibration + coverage", _calibration)
check("experiments.explain_final Cohen's d", _cohen)

print(f"\n{'all checks passed' if not failed else f'{len(failed)} failed: {failed}'}")
sys.exit(1 if failed else 0)
