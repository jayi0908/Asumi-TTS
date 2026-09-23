#!/usr/bin/env python3
"""Command line for the Asumi TTS engine.

Examples
--------
  # single line -> wav
  python -m asumi_tts.cli --text "おはよう、亜澄だよ。" --out hello.wav

  # many lines -> directory (001.wav, 002.wav, ...)
  python -m asumi_tts.cli --text-file lines.txt --out outdir/
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import soundfile as sf

from .engine import AsumiTTSEngine


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Asumi (亜澄) Japanese TTS")
    ap.add_argument("--text", action="append", help="repeatable")
    ap.add_argument("--text-file", help="one utterance per line")
    ap.add_argument("--out", required=True, help="output .wav or directory")
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--style", default="Neutral")
    ap.add_argument("--intonation-scale", type=float, default=1.0)
    ap.add_argument("--length", type=float, default=1.0, help="speaking rate (>1 slower)")
    args = ap.parse_args(argv)

    texts = list(args.text or [])
    if args.text_file:
        texts += [l.strip() for l in Path(args.text_file).read_text().splitlines() if l.strip()]
    if not texts:
        ap.error("provide at least one --text or --text-file")

    tts = AsumiTTSEngine(device=args.device, style=args.style)
    out = Path(args.out)
    multi = len(texts) > 1
    if multi:
        out.mkdir(parents=True, exist_ok=True)
    for i, text in enumerate(texts, 1):
        t0 = time.time()
        sr, wav = tts.speak(text, intonation_scale=args.intonation_scale, length=args.length)
        dst = (out / f"{i:03d}.wav") if multi else out
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dst), wav, sr)
        print(f"[{i}] {dst}  dur={len(wav)/sr:.2f}s  gen={time.time()-t0:.2f}s  {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
