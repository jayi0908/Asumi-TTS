#!/usr/bin/env python3
"""Render the cleaned 44.1 kHz dataset for Style-Bert-VITS2.

Uses processed/manifests/dataset.tsv (name, source, start, end in the ORIGINAL
file timeline) to cut each final clip straight from the source .ogg and write
it to processed/sbv2/asu/raw/<name>.wav (44.1 kHz mono PCM16).  No loudness
processing here: SBV2's resample.py normalises loudness itself.

Also writes processed/sbv2/asu/esd.list:
    raw/<name>.wav|asu|JP|<text>
"""
import concurrent.futures as cf
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATASET = ROOT / "processed" / "manifests" / "dataset.tsv"
TRANSCRIPTS = ROOT / "processed" / "manifests" / "transcripts.tsv"
OUT = ROOT / "processed" / "sbv2" / "asu"
SOURCE_DIR = ROOT / "data" / "source_ogg"
RAW = OUT / "raw"
ESD = OUT / "esd.list"
LOG = ROOT / "processed" / "logs" / "06_render_sbv2.log"

SR = 44100
SPK = "asu"
LANG = "JP"


def cut(source: pathlib.Path, start: float, end: float, dst: pathlib.Path) -> str:
    dur = max(0.05, end - start)
    res = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(source),
         "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
         "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(dst)],
        capture_output=True, text=True,
    )
    return "" if res.returncode == 0 else f"{dst.name}: {res.stderr.strip()[:120]}"


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    texts = {}
    for line in TRANSCRIPTS.read_text().splitlines():
        p = line.split("\t")
        if len(p) >= 2 and p[1].strip():
            texts[p[0]] = p[1].strip()

    jobs, rows = [], []
    for line in DATASET.read_text().splitlines():
        name, source, start, end = line.split("\t")
        if name not in texts:
            continue
        jobs.append((SOURCE_DIR / source, float(start), float(end), RAW / name))
        rows.append(f"raw/{name}|{SPK}|{LANG}|{texts[name]}")

    errors = []
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for err in ex.map(lambda j: cut(*j), jobs):
            if err:
                errors.append(err)

    ESD.write_text("\n".join(rows) + "\n")
    LOG.write_text(f"clips={len(rows)} errors={len(errors)}\n" + "\n".join(errors) + "\n")
    print(f"rendered {len(rows) - len(errors)}/{len(rows)} clips -> {RAW}")
    print(f"esd.list -> {ESD}  errors={len(errors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
