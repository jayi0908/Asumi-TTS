#!/usr/bin/env python3
"""Move clips shorter than MIN_SEC into processed/excluded_short/.

Also emits:
  - processed/manifests/kept_files.txt     (files to keep)
  - processed/manifests/review_short.txt   (MIN_SEC .. REVIEW_MAX, needs spot-check)
"""
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "processed" / "manifests" / "durations.tsv"
EXCLUDE_DIR = ROOT / "processed" / "excluded_short"
SOURCE_DIR = ROOT / "data" / "source_ogg"
KEPT = ROOT / "processed" / "manifests" / "kept_files.txt"
REVIEW = ROOT / "processed" / "manifests" / "review_short.txt"

MIN_SEC = 1.0
REVIEW_MAX = 2.0


def main() -> int:
    rows = []
    for line in MANIFEST.read_text().splitlines():
        dur, name = line.split("\t")
        rows.append((float(dur), name))

    EXCLUDE_DIR.mkdir(parents=True, exist_ok=True)
    moved, kept, review = 0, [], []
    for dur, name in rows:
        src = SOURCE_DIR / name
        if dur < MIN_SEC:
            dst = EXCLUDE_DIR / name
            if src.exists():
                shutil.move(str(src), str(dst))
                moved += 1
        else:
            kept.append(name)
            if dur < REVIEW_MAX:
                review.append(name)

    KEPT.write_text("\n".join(kept) + "\n")
    REVIEW.write_text("\n".join(review) + "\n")
    print(f"excluded (<{MIN_SEC}s): {moved}")
    print(f"kept: {len(kept)}")
    print(f"review ({MIN_SEC}-{REVIEW_MAX}s): {len(review)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
