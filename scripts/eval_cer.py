#!/usr/bin/env python3
"""Round-trip ASR CER for the synthesised eval set (run in the asumi env)."""
import pathlib
import re
import sys

import numpy as np
from faster_whisper import WhisperModel

import argparse
import os

EVAL = pathlib.Path("/Users/jayi0908/Desktop/Something/asumi/Style-Bert-VITS2/output_eval")
_P = argparse.ArgumentParser()
_P.add_argument("--manifest", default="eval_manifest.tsv")
_A = _P.parse_args()
MANIFEST = EVAL / _A.manifest

PUNCT = "、。，．！？!?…・「」『』（）()\"'：:；;,.ー-―—　"


def norm(s: str) -> str:
    s = "".join(c for c in s if c not in PUNCT)
    return s.replace(" ", "").replace("\u3000", "")


def cer(ref: str, hyp: str) -> float:
    r, h = norm(ref), norm(hyp)
    if not r:
        return 1.0
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return float(d[len(r), len(h)]) / len(r)


def main() -> int:
    model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
    rows = [l.split("\t") for l in MANIFEST.read_text().splitlines() if l.strip()]
    results = []
    for gen_name, split, cos, dur, text in rows:
        segs, _ = model.transcribe(str(MANIFEST.parent / gen_name), language="ja", beam_size=5)
        hyp = "".join(s.text for s in segs).strip()
        results.append((split, float(cos), cer(text, hyp), text, hyp))

    for split in ("val", "train", "all"):
        sub = [r for r in results if split == "all" or r[0] == split]
        c = np.array([r[2] for r in sub])
        print(f"{split:5s} n={len(sub):2d}  CER mean={c.mean():.3f} median={np.median(c):.3f} "
              f"exact={int((c == 0).sum())}/{len(sub)}")

    print("\nworst 8 by CER:")
    for split, cos, e, text, hyp in sorted(results, key=lambda r: -r[2])[:8]:
        print(f"  CER={e:.2f} cos={cos:.3f}  ref={text[:32]}")
        print(f"                          hyp={hyp[:32]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
