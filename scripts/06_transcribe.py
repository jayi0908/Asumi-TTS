#!/usr/bin/env python3
"""Transcribe the normalised dataset with faster-whisper (Japanese).

Resumable: re-running skips files already present in transcripts.tsv and
appends new results immediately, so progress survives interruption.

Outputs:
  processed/manifests/transcripts.tsv   name<TAB>text<TAB>dur<TAB>logprob<TAB>nsp
  processed/manifests/train_list.txt    "<abs-wav>"|"asu"|"ja"|"<text>"
  processed/manifests/review_asr.txt    low-confidence / suspicious lines
"""
import os
import pathlib
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import soundfile as sf  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "processed" / "dataset_norm"
OUT = ROOT / "processed" / "manifests"
LOG = ROOT / "processed" / "logs" / "05_asr.log"

MODEL = os.environ.get("ASR_MODEL", "large-v3-turbo")
LANG = "ja"
SPK = "asu"
BEAM = int(os.environ.get("ASR_BEAM", "1"))
TSV = OUT / "transcripts.tsv"
TRAIN = OUT / "train_list.txt"
REVIEW = OUT / "review_asr.txt"


def load_done() -> dict[str, str]:
    done: dict[str, str] = {}
    if TSV.exists():
        for line in TSV.read_text().splitlines():
            parts = line.split("\t")
            if parts:
                done[parts[0]] = line
    return done


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    done = load_done()
    files = [f for f in sorted(SRC.glob("*.wav")) if f.name not in done]
    print(f"resume: {len(done)} done, {len(files)} to go", flush=True)

    model = WhisperModel(MODEL, device="cpu", compute_type="int8", cpu_threads=8)
    t0 = time.time()
    todo_dur = sum(sf.info(str(f)).duration for f in files)
    done_dur = 0.0

    with TSV.open("a") as fh:
        for i, f in enumerate(files, 1):
            dur = sf.info(str(f)).duration
            done_dur += dur
            segments, info = model.transcribe(
                str(f), language=LANG, beam_size=BEAM,
                condition_on_previous_text=False, temperature=0.0,
            )
            segs = list(segments)
            text = "".join(s.text for s in segs).strip()
            avg_lp = sum(s.avg_logprob for s in segs) / len(segs) if segs else -99.0
            nsp = max((s.no_speech_prob for s in segs), default=1.0)
            fh.write(f"{f.name}\t{text}\t{dur:.3f}\t{avg_lp:.3f}\t{nsp:.3f}\n")
            fh.flush()
            if i % 25 == 0 or i == len(files):
                el = time.time() - t0
                print(f"  {i}/{len(files)}  audio={done_dur/60:.1f}/{todo_dur/60:.1f}min"
                      f"  elapsed={el/60:.1f}min  rt={done_dur/el:.2f}x", flush=True)

    rows = []
    review = []
    for line in TSV.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        name, text, dur, lp, nsp = parts[0], parts[1], float(parts[2]), float(parts[3]), float(parts[4])
        rows.append((name, text, dur, lp, nsp))
        if not text or lp < -0.8 or nsp > 0.6 or len(text) < 2:
            review.append(f"{name}\t{lp:.2f}\t{nsp:.2f}\t{text}")
    with TRAIN.open("w") as fh:
        for name, text, *_ in rows:
            fh.write(f"{(SRC / name).resolve()}|{SPK}|{LANG}|{text}\n")
    REVIEW.write_text("\n".join(review) + "\n")
    LOG.write_text(f"model={MODEL} beam={BEAM} total={len(rows)} review={len(review)}\n")
    print(f"done: {len(rows)} transcripts, {len(review)} flagged -> {TSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
