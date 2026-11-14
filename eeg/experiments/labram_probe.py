"""Frozen LaBraM (Jiang et al., ICLR 2024; braindecode/labram-pretrained, MIT) as a feature extractor.

Each cleaned 10 s window (128 Hz) is resampled to 200 Hz and scaled from uV to
LaBraM's 0.1 mV units; the 19 10-20 channels are passed by name so the
pretrained channel embeddings line up. Embedding = mean over the 190 tokens
(19 channels x 10 one-second patches) of block 6 (middle - best on ds004504 in
the 2026 frozen-representation study) and of the final block. Per subject: mean
over windows. Probe: L2 logistic regression, the repeated subject-wise CV of
eeg/experiments/baselines.py. No fine-tuning - nothing about the test subjects reaches the encoder.

    python -m eeg.experiments.labram_probe --out ~/eeg/raw2
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eeg.data import park as P
from eeg.data import raw_eeg as R
from eeg.experiments.baselines import WITHIN, cv_auc, lr_probe

LAYERS = {"block6": 5, "final": None}


@torch.no_grad()
def embed(win, model, device, batch=32):
    import mne
    mne.set_log_level("ERROR")
    names = [c.upper() for c in P.CHANNELS]
    acts = {}
    h = model.blocks[LAYERS["block6"]].register_forward_hook(lambda m, i, o: acts.__setitem__("b", o))
    out = {k: [] for k in LAYERS}
    for k in range(0, len(win), batch):
        x = mne.filter.resample(win[k:k + batch].astype(np.float64), up=200, down=128) / 100.0
        r = model(torch.tensor(x[..., :2000], dtype=torch.float32, device=device),
                  return_features=True, ch_names=names)
        out["final"].append(r["features"].mean(1).cpu().numpy())
        out["block6"].append(acts["b"][:, 1:].mean(1).cpu().numpy())
    h.remove()
    return {k: np.concatenate(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/eeg/raw2")
    ap.add_argument("--save", default="results/labram.json")
    args = ap.parse_args()
    from eeg.paths import trust_system_certificates
    trust_system_certificates()
    from braindecode.models import Labram
    out = Path(args.out).expanduser()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Labram.from_pretrained("braindecode/labram-pretrained", n_times=2000).eval().to(device)
    emb_cache, report = {}, {}
    for task, ds in WITHIN:
        pos, neg, _ = R.TASKS[task]
        _, _, subject, groups, _, win = R.load_features(out, [ds], raw=True)
        if ds not in emb_cache:
            f = out / "features" / f"{ds}_labram.npz"
            if f.exists():
                emb_cache[ds] = dict(np.load(f))
            else:
                emb_cache[ds] = embed(win, model, device)
                np.savez(f, **emb_cache[ds])
        keep, y = R.task_rows(groups, pos, neg)
        s = subject[keep]
        u = np.unique(s)
        ys = np.array([y[s == v][0] for v in u])
        res = {}
        for layer, e in emb_cache[ds].items():
            e = e[keep]
            x = np.stack([e[s == v].mean(0) for v in u])
            res[layer] = cv_auc(lr_probe, x, ys)
        report[f"{task}:{ds}"] = res
        print(f"{task:<18} {ds:<9} n={len(ys):<4}" + "".join(f"  {k} {a:.3f}±{sd:.3f}" for k, (a, sd) in res.items()),
              flush=True)
    Path(args.save).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
