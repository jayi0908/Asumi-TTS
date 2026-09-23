#!/usr/bin/env python3
"""Two-pass EBU R128 loudness normalisation (linear gain) for the dataset.

Target: I=-18 LUFS, TP=-1.5 dBTP, LRA=11.  Pass 1 measures, pass 2 applies a
constant gain via `linear=true`, so dynamics are preserved (no pumping).

Outputs processed/dataset_norm/<name>.wav
"""
import concurrent.futures as cf
import json
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "processed" / "dataset"
DST = ROOT / "processed" / "dataset_norm"
LOG = ROOT / "processed" / "logs" / "04_loudnorm.log"

SR = 24000
I, TP, LRA = -18.0, -1.5, 11.0
WORKERS = 6
FILTER = f"loudnorm=I={I}:TP={TP}:LRA={LRA}"
JSON_RE = re.compile(r"\{[^{}]*\"input_i\"[^{}]*\}", re.S)


def measure(src: pathlib.Path) -> dict:
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-i", str(src),
         "-af", f"{FILTER}:print_format=json", "-f", "null", "-"],
        capture_output=True, text=True,
    ).stderr
    m = JSON_RE.search(out)
    if not m:
        raise RuntimeError("no loudnorm json")
    return json.loads(m.group(0))


def process(src: pathlib.Path) -> str:
    try:
        meas = measure(src)
        dst = DST / src.name
        af = (
            f"{FILTER}:measured_I={meas['input_i']}:measured_TP={meas['input_tp']}"
            f":measured_LRA={meas['input_lra']}:measured_thresh={meas['input_thresh']}"
            f":offset={meas['target_offset']}:linear=true"
        )
        res = subprocess.run(
            ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(src),
             "-af", af, "-ar", str(SR), "-c:a", "pcm_s16le", str(dst)],
            capture_output=True, text=True,
        )
        if res.returncode != 0:
            return f"FAIL {src.name}: {res.stderr.strip()[:160]}"
        return ""
    except Exception as exc:  # noqa: BLE001
        return f"FAIL {src.name}: {exc}"


def main() -> int:
    DST.mkdir(parents=True, exist_ok=True)
    files = sorted(SRC.glob("*.wav"))
    errors = []
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, err in enumerate(ex.map(process, files), 1):
            if err:
                errors.append(err)
            if i % 300 == 0 or i == len(files):
                print(f"  {i}/{len(files)} errors={len(errors)}", flush=True)

    LOG.write_text(f"files={len(files)} errors={len(errors)}\n" + "\n".join(errors) + "\n")
    print(f"normalised {len(files) - len(errors)}/{len(files)} -> {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
