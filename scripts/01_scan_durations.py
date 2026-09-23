#!/usr/bin/env python3
"""Scan durations of all .ogg files in the dataset root.

Outputs processed/manifests/durations.tsv with columns: duration_sec<TAB>filename
"""
import concurrent.futures as cf
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "processed" / "manifests" / "durations.tsv"
SOURCE_DIR = ROOT / "data" / "source_ogg"


def probe(path: pathlib.Path) -> tuple[float, str]:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0", str(path),
        ],
        capture_output=True, text=True,
    )
    try:
        return float(out.stdout.strip()), path.name
    except ValueError:
        return -1.0, path.name


def main() -> int:
    files = sorted(SOURCE_DIR.glob("*.ogg"))
    if not files:
        print("no .ogg files found", file=sys.stderr)
        return 1

    rows: list[tuple[float, str]] = []
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for dur, name in ex.map(probe, files):
            rows.append((dur, name))

    rows.sort(key=lambda r: r[1])
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w") as fh:
        for dur, name in rows:
            fh.write(f"{dur:.6f}\t{name}\n")

    bad = [n for d, n in rows if d < 0]
    total = sum(d for d, _ in rows if d > 0)
    print(f"scanned {len(rows)} files, total {total / 60:.1f} min, unreadable {len(bad)}")
    for n in bad:
        print(f"  UNREADABLE: {n}")
    print(f"manifest -> {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
