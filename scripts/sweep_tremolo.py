#!/usr/bin/env python3
"""Robust inference-param sweep for tremolo, plus a WORLD de-jitter post-process.

Averages a local F0-oscillation index over several seeds for each config and
writes one sample per config so the user can A/B listen.
"""
import argparse
import pathlib
import sys

import numpy as np
import soundfile as sf

SBV2 = pathlib.Path("/Users/jayi0908/Desktop/Something/asumi/Style-Bert-VITS2")
sys.path.insert(0, str(SBV2))
OUT = SBV2 / "output_stability" / "sweep"


def tremolo(wav: np.ndarray, sr: int) -> float:
    import pyworld as pw
    x = wav.astype(np.float64)
    f0, t = pw.dio(x, sr, frame_period=5.0)
    f0 = pw.stonemask(x, f0, t, sr)
    v = f0 > 0
    if v.sum() < 8:
        return float("nan")
    return float(np.mean(np.abs(np.diff(np.log(f0[v]), 2))))


def dejitter(wav: np.ndarray, sr: int, kernel: int = 5) -> np.ndarray:
    import pyworld as pw
    from scipy.signal import medfilt
    x = wav.astype(np.float64)
    f0, t = pw.dio(x, sr, frame_period=5.0)
    f0 = pw.stonemask(x, f0, t, sr)
    v = f0 > 0
    f0s = f0.copy()
    if v.any():
        f0s[v] = medfilt(f0[v], kernel_size=kernel)
    sp = pw.cheaptrick(x, f0s, t, sr)
    ap = pw.d4c(x, f0s, t, sr)
    y = pw.synthesize(f0s, sp, ap, sr)
    return y.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_assets/asu/asu_e40_s9200.safetensors")
    ap.add_argument("--tag", default="e40")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    import torch
    from style_bert_vits2.constants import Languages
    from style_bert_vits2.nlp.japanese.user_dict import update_dict
    from style_bert_vits2.tts_model import TTSModel

    update_dict()
    model = TTSModel(pathlib.Path(args.model), pathlib.Path("model_assets/asu/config.json"),
                     pathlib.Path("model_assets/asu/style_vectors.npy"), device=args.device)
    model.load()
    outdir = OUT / args.tag
    outdir.mkdir(parents=True, exist_ok=True)

    sentences = {
        "s1": "水着で二人っきりは恥ずかしいよー",
        "s2": "いいですね温泉に引きこもって長風呂したいです",
    }
    configs = [
        ("base",       {}),
        ("sdp0",       dict(sdp_ratio=0.0)),
        ("noise_mid",  dict(noise=0.4, noise_w=0.6)),
        ("noise_low",  dict(noise=0.2, noise_w=0.4)),
        ("noisew_lo",  dict(noise_w=0.4)),
        ("style0",     dict(style_weight=0.0)),
        ("style05",    dict(style_weight=0.5)),
        ("inton08",    dict(intonation_scale=0.8)),
    ]
    seeds = [42, 1, 2]

    print(f"{'config':10s} " + " ".join(f"{k:>8s}" for k in sentences))
    for name, kw in configs:
        vals = []
        for sk, text in sentences.items():
            ts = []
            for i, seed in enumerate(seeds):
                torch.manual_seed(seed)
                sr, audio = model.infer(text, language=Languages.JP, speaker_id=0,
                                        style="Neutral", **kw)
                wav = np.asarray(audio)
                ts.append(tremolo(wav, sr))
                if i == 0:
                    sf.write(str(outdir / f"{sk}_{name}.wav"), wav, sr)
            vals.append(np.nanmean(ts))
        print(f"{name:10s} " + " ".join(f"{v:8.4f}" for v in vals))

    # post-process the base samples
    print("\npost-process (WORLD median F0 smoothing):")
    for sk in sentences:
        w, sr = sf.read(str(outdir / f"{sk}_base.wav"), dtype="float32")
        y = dejitter(w, sr, kernel=5)
        sf.write(str(outdir / f"{sk}_base_dejitter.wav"), y, sr)
        print(f"  {sk}: base={tremolo(w,sr):.4f} -> dejitter={tremolo(y,sr):.4f}")
    print(f"\nsamples -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
