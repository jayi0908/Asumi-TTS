#!/usr/bin/env python3
"""Local Japanese TTS CLI for the 'asu' (亜澄) Style-Bert-VITS2 JP-Extra model.

Run with the Style-Bert-VITS2 env, e.g.:
  cd /Users/jayi0908/Desktop/Something/asumi/Style-Bert-VITS2
  HF_ENDPOINT=https://hf-mirror.com conda run -n sbv2 \
    python /Users/jayi0908/Desktop/Something/asumi/asumi_voice/scripts/sbv2_tts.py \
    --text "おはよう。今日もいい天気だね。" --out out.wav

Loads the user dictionary (dict_data/default.csv) so proper nouns such as
亜澄 (あすみ) are read correctly.
"""
import argparse
import pathlib
import sys
import time

import numpy as np
import soundfile as sf

SBV2 = pathlib.Path("/Users/jayi0908/Desktop/Something/asumi/Style-Bert-VITS2")
sys.path.insert(0, str(SBV2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_assets/asu/asu_e40_s9200.safetensors")
    ap.add_argument("--config", default="model_assets/asu/config.json")
    ap.add_argument("--style-vec", default="model_assets/asu/style_vectors.npy")
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--style", default="Neutral")
    # 1.0 = original. Lowering it reduced the wobble but made the voice sound
    # thin/hollow, so it is reverted to the default. Exposed for experiments.
    ap.add_argument("--intonation-scale", type=float, default=1.0)
    ap.add_argument("--dejitter", action="store_true",
                    help="extra WORLD median-F0 smoothing post-process")
    ap.add_argument("--text", action="append", help="repeatable")
    ap.add_argument("--text-file", help="one utterance per line")
    ap.add_argument("--out", default="out.wav", help="output file (or directory for multiple)")
    args = ap.parse_args()

    texts = list(args.text or [])
    if args.text_file:
        texts += [l.strip() for l in pathlib.Path(args.text_file).read_text().splitlines() if l.strip()]
    if not texts:
        ap.error("no --text / --text-file given")

    from style_bert_vits2.constants import Languages
    from style_bert_vits2.nlp.japanese.user_dict import update_dict
    from style_bert_vits2.tts_model import TTSModel

    update_dict()  # apply user dictionary (亜澄 -> アスミ, ...)
    model = TTSModel(pathlib.Path(args.model), pathlib.Path(args.config),
                     pathlib.Path(args.style_vec), device=args.device)
    model.load()

    out = pathlib.Path(args.out)
    multi = len(texts) > 1
    if multi:
        out.mkdir(parents=True, exist_ok=True)
    for i, text in enumerate(texts, 1):
        t0 = time.time()
        sr, audio = model.infer(text, language=Languages.JP, speaker_id=0,
                                style=args.style, intonation_scale=args.intonation_scale)
        wav = np.asarray(audio)
        if args.dejitter:
            wav = _dejitter(wav, sr)
        dur = len(wav) / sr
        dst = (out / f"{i:03d}.wav") if multi else out
        sf.write(str(dst), wav, sr)
        print(f"[{i}] {dst}  dur={dur:.2f}s  gen={time.time()-t0:.2f}s  {text}")
    return 0


def _dejitter(wav: np.ndarray, sr: int, kernel: int = 5) -> np.ndarray:
    """Mild WORLD F0 median smoothing to remove residual pitch wobble."""
    import pyworld as pw
    from scipy.signal import medfilt
    x = wav.astype(np.float64)
    f0, t = pw.dio(x, sr, frame_period=5.0)
    f0 = pw.stonemask(x, f0, t, sr)
    v = f0 > 0
    if v.any():
        f0[v] = medfilt(f0[v], kernel_size=kernel)
    sp = pw.cheaptrick(x, f0, t, sr)
    ap = pw.d4c(x, f0, t, sr)
    return pw.synthesize(f0, sp, ap, sr).astype(np.float32)


if __name__ == "__main__":
    raise SystemExit(main())
