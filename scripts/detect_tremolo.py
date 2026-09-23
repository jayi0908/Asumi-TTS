#!/usr/bin/env python3
"""Rank synthesised clips by a local F0-oscillation ('tremolo') index.

tremolo_index = mean |second difference of log-F0| over voiced frames,
normalised to frame step.  High values = fast pitch wobble.
Also reports shimmer (amplitude irregularity) via frame RMS second diff.
"""
import argparse
import glob
import pathlib

import numpy as np
import soundfile as sf

SBV2 = pathlib.Path("/Users/jayi0908/Desktop/Something/asumi/Style-Bert-VITS2")


def metrics(path: str) -> tuple[float, float, float]:
    import pyworld as pw
    x, sr = sf.read(path, dtype="float64")
    if x.ndim > 1:
        x = x.mean(1)
    f0, t = pw.dio(x, sr, frame_period=5.0)
    f0 = pw.stonemask(x, f0, t, sr)
    v = f0 > 0
    if v.sum() < 8:
        return float("nan"), float("nan"), float("nan")
    lf = np.log(f0[v])
    d2 = np.abs(np.diff(lf, 2))
    trem = float(np.mean(d2))
    # amplitude (RMS per 5ms frame) second-diff as shimmer proxy
    hop = int(0.005 * sr)
    n = len(x) // hop
    rms = np.array([np.sqrt(np.mean(x[i*hop:(i+1)*hop]**2) + 1e-12) for i in range(n)])
    lr = np.log(rms + 1e-9)
    shim = float(np.mean(np.abs(np.diff(lr, 2))))
    return trem, shim, float(np.mean(f0[v]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=str(SBV2 / "output_eval/e40/*.wav"))
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(args.glob)):
        trem, shim, mf0 = metrics(f)
        rows.append((trem, shim, mf0, pathlib.Path(f).name))
    rows.sort(reverse=True)
    print(f"{'tremolo':>8s} {'shimmer':>8s} {'meanF0':>7s}  file")
    for trem, shim, mf0, name in rows[: args.top]:
        print(f"{trem:8.4f} {shim:8.4f} {mf0:7.1f}  {name}")
    vals = np.array([r[0] for r in rows if not np.isnan(r[0])])
    print(f"\nn={len(vals)} tremolo p50={np.percentile(vals,50):.4f} "
          f"p90={np.percentile(vals,90):.4f} max={vals.max():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
