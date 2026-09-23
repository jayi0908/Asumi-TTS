#!/usr/bin/env python3
"""Synthesise the dataset's own transcripts and score speaker similarity.

Run with the Style-Bert-VITS2 env (has style_bert_vits2 + pyannote):
  cd <Style-Bert-VITS2>
  python /path/to/eval_sbv2_synth.py --model model_assets/asu/asu_e40_s9200.safetensors
"""
import argparse
import pathlib
import random
import sys
import time

import numpy as np
import soundfile as sf

SBV2 = pathlib.Path("/Users/jayi0908/Desktop/Something/asumi/Style-Bert-VITS2")
sys.path.insert(0, str(SBV2))
OUT = SBV2 / "output_eval"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_assets/asu/asu_e40_s9200.safetensors")
    ap.add_argument("--config", default="model_assets/asu/config.json")
    ap.add_argument("--style-vec", default="model_assets/asu/style_vectors.npy")
    ap.add_argument("--n-train", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    import torch
    from pyannote.audio import Inference, Model
    from style_bert_vits2.constants import Languages
    from style_bert_vits2.tts_model import TTSModel

    OUT.mkdir(parents=True, exist_ok=True)
    outdir = OUT / args.tag
    outdir.mkdir(parents=True, exist_ok=True)
    tts = TTSModel(pathlib.Path(args.model), pathlib.Path(args.config),
                   pathlib.Path(args.style_vec), device=args.device)
    tts.load()
    spk = Inference(Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM"),
                    window="whole")

    def emb(path):
        v = np.asarray(spk(str(path))).ravel()
        return v / (np.linalg.norm(v) + 1e-9)

    items = []
    for name in ("val.list", "train.list"):
        lines = [l for l in (SBV2 / "Data/asu" / name).read_text().splitlines() if l.strip()]
        if name == "val.list":
            items += [("val", l) for l in lines]
        else:
            random.seed(args.seed)
            items += [("train", l) for l in random.sample(lines, min(args.n_train, len(lines)))]

    rows = []
    t_all = time.time()
    for i, (split, line) in enumerate(items, 1):
        wav_path, _, _, text = line.split("|")[:4]
        ref = SBV2 / wav_path
        t0 = time.time()
        sr, audio = tts.infer(text, language=Languages.JP, speaker_id=0, style="Neutral")
        el = time.time() - t0
        gen = outdir / f"{split}_{i:03d}.wav"
        sf.write(str(gen), np.asarray(audio), sr)
        cos = float(emb(gen) @ emb(ref))
        dur = len(audio) / sr
        rows.append(f"{gen.name}\t{split}\t{cos:.3f}\t{dur:.2f}\t{text}")
        print(f"[{i}/{len(items)}] {split} cos={cos:.3f} dur={dur:.2f}s gen={el:.2f}s", flush=True)

    (outdir / "eval_manifest.tsv").write_text("\n".join(rows) + "\n")
    print(f"done in {time.time()-t_all:.0f}s -> {outdir/'eval_manifest.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
