#!/usr/bin/env python3
"""VAD-based edge trimming + resample to 24 kHz mono WAV.

For every kept .ogg:
  - run Silero VAD (16 kHz) to locate speech
  - trim leading/trailing silence, keeping PAD_SEC of margin
  - resample to TARGET_SR and write 16-bit PCM WAV

Outputs:
  processed/wav24k/<stem>.wav
  processed/manifests/vad_segments.json
  processed/logs/02_vad_trim.log
"""
import json
import pathlib
import subprocess
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio
from silero_vad import get_speech_timestamps, load_silero_vad

ROOT = pathlib.Path(__file__).resolve().parent.parent
KEPT = ROOT / "processed" / "manifests" / "kept_files.txt"
OUT_DIR = ROOT / "processed" / "wav24k"
SOURCE_DIR = ROOT / "data" / "source_ogg"
SEG_JSON = ROOT / "processed" / "manifests" / "vad_segments.json"
LOG = ROOT / "processed" / "logs" / "02_vad_trim.log"

TARGET_SR = 24000
VAD_SR = 16000
SRC_SR = 48000
PAD_SEC = 0.10


def read_audio(path: pathlib.Path, sr: int) -> torch.Tensor:
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
        "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sr), "-",
    ]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.decode(errors="replace").strip()[:200])
    buf = np.frombuffer(res.stdout, dtype=np.float32)
    return torch.from_numpy(buf.copy())


def main() -> int:
    files = [l for l in KEPT.read_text().splitlines() if l.strip()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = load_silero_vad()
    info: dict[str, dict] = {}
    no_speech: list[str] = []
    failed: list[str] = []
    t0 = time.time()

    for i, name in enumerate(files, 1):
        src = SOURCE_DIR / name
        stem = src.stem
        try:
            sr = SRC_SR
            wav = read_audio(src, sr)
            dur = wav.shape[0] / sr
            wav16 = torchaudio.functional.resample(wav, sr, VAD_SR)
            ts = get_speech_timestamps(
                wav16, model, threshold=0.35, sampling_rate=VAD_SR,
                min_speech_duration_ms=150, min_silence_duration_ms=100,
                speech_pad_ms=30, return_seconds=True,
            )
            if ts:
                start = max(0.0, ts[0]["start"] - PAD_SEC)
                end = min(dur, ts[-1]["end"] + PAD_SEC)
                segs = [[round(s["start"], 3), round(s["end"], 3)] for s in ts]
            else:
                start, end = 0.0, dur
                segs = []
                no_speech.append(name)

            wav_out = wav[int(start * sr):int(end * sr)]
            wav24 = torchaudio.functional.resample(wav_out, sr, TARGET_SR)
            sf.write(OUT_DIR / f"{stem}.wav", wav24.numpy(), TARGET_SR,
                     subtype="PCM_16")

            info[stem] = {
                "file": name,
                "orig_sr": sr,
                "orig_dur": round(dur, 3),
                "trim": [round(start, 3), round(end, 3)],
                "segments": segs,
                "no_speech": not bool(ts),
            }
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{name}: {exc}")

        if i % 200 == 0 or i == len(files):
            print(f"  {i}/{len(files)}  ({time.time() - t0:.0f}s)", flush=True)

    SEG_JSON.write_text(json.dumps(info, ensure_ascii=False, indent=0))
    LOG.write_text(
        f"processed={len(info)} failed={len(failed)} no_speech={len(no_speech)}\n"
        + "FAILED:\n" + "\n".join(failed) + "\n\nNO_SPEECH:\n" + "\n".join(no_speech) + "\n"
    )
    print(f"done: {len(info)} -> {OUT_DIR}")
    print(f"failed={len(failed)} no_speech={len(no_speech)} log={LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
