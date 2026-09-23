#!/usr/bin/env python3
"""Build the final training dataset: copy short clips, split long clips.

Reads processed/wav24k/*.wav plus processed/manifests/vad_segments.json.
A clip whose TRIMMED length exceeds MAX_SEC is split at Silero-VAD silence
boundaries into pieces preferably within [MIN_SEC, MAX_SEC].  Pieces shorter
than DROP_SEC are moved to processed/excluded_split/.

Outputs:
  processed/dataset/<stem>.wav | <stem>_pNN.wav
  processed/manifests/dataset.tsv   (name<TAB>source<TAB>start<TAB>end)
  processed/logs/03_split.log
"""
import json
import pathlib
import shutil

import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parent.parent
SEG_JSON = ROOT / "processed" / "manifests" / "vad_segments.json"
WAV_DIR = ROOT / "processed" / "wav24k"
OUT_DIR = ROOT / "processed" / "dataset"
DROP_DIR = ROOT / "processed" / "excluded_split"
MANIFEST = ROOT / "processed" / "manifests" / "dataset.tsv"
LOG = ROOT / "processed" / "logs" / "03_split.log"

SR = 24000
MAX_SEC = 12.0
MIN_SEC = 3.0
HARD_SEC = 16.0
PAD_SEC = 0.08
DROP_SEC = 1.0


def plan_chunks(L: float, segs: list[list[float]]) -> list[tuple[float, float]]:
    if L <= MAX_SEC or not segs:
        return [(0.0, L)]

    chunks: list[list[float]] = [[segs[0][0], segs[0][1]]]
    for s, e in segs[1:]:
        if e - chunks[-1][0] <= MAX_SEC:
            chunks[-1][1] = e
        else:
            chunks.append([s, e])

    merged: list[list[float]] = []
    for c in chunks:
        if merged and (c[1] - c[0]) < MIN_SEC and (c[1] - merged[-1][0]) <= HARD_SEC:
            merged[-1][1] = c[1]
        else:
            merged.append(c)

    if len(merged) > 1 and (merged[0][1] - merged[0][0]) < MIN_SEC:
        if (merged[1][1] - merged[0][0]) <= HARD_SEC:
            merged[1][0] = merged[0][0]
            merged.pop(0)

    return [(max(0.0, s - PAD_SEC), min(L, e + PAD_SEC)) for s, e in merged]


def main() -> int:
    info = json.loads(SEG_JSON.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DROP_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    split_src = pieces = oversize = dropped = 0
    log_lines: list[str] = []

    for stem in sorted(info):
        meta = info[stem]
        src = WAV_DIR / f"{stem}.wav"
        wav, sr = sf.read(str(src), dtype="float32")
        assert sr == SR, f"{stem}: unexpected sr {sr}"
        L = len(wav) / SR
        trim0 = meta["trim"][0]

        if L <= MAX_SEC:
            plan = [(stem + ".wav", 0.0, L)]
        else:
            shift = [[max(0.0, s - trim0), min(L, e - trim0)]
                     for s, e in meta["segments"]]
            plan = [(f"{stem}_p{i:02d}.wav", s, e)
                    for i, (s, e) in enumerate(plan_chunks(L, shift), 1)]
            split_src += 1

        for name, s, e in plan:
            if (e - s) < DROP_SEC:
                sf.write(DROP_DIR / name, wav[int(s * SR):int(e * SR)], SR,
                         subtype="PCM_16")
                dropped += 1
                log_lines.append(f"DROPPED {name} ({(e - s):.2f}s)")
                continue
            sf.write(OUT_DIR / name, wav[int(s * SR):int(e * SR)], SR,
                     subtype="PCM_16")
            rows.append(f"{name}\t{meta['file']}\t{trim0 + s:.3f}\t{trim0 + e:.3f}")
            pieces += 1
            if (e - s) > MAX_SEC + 0.05:
                oversize += 1
                log_lines.append(f"OVERSIZE {name}: {e - s:.2f}s")

    MANIFEST.write_text("\n".join(rows) + "\n")
    LOG.write_text(
        f"sources={len(info)} split_sources={split_src} pieces={pieces} "
        f"oversize={oversize} dropped={dropped}\n" + "\n".join(log_lines) + "\n"
    )
    print(f"sources={len(info)}  final pieces={pieces}  split_sources={split_src}")
    print(f"oversize kept={oversize}  dropped(<{DROP_SEC}s)={dropped}")
    print(f"-> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
