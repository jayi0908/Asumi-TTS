#!/usr/bin/env python3
"""Sweep Style-Bert-VITS2 inference params and measure F0 jitter (tremolo proxy).

Run in the Style-Bert-VITS2 env.  jitter = mean(|dF0|)/mean(F0) over voiced
frames; lower = steadier.  Also reports F0 std (log) as a modulation measure.
"""
import argparse
import pathlib
import sys

import numpy as np
import soundfile as sf

SBV2 = pathlib.Path("/Users/jayi0908/Desktop/Something/asumi/Style-Bert-VITS2")
sys.path.insert(0, str(SBV2))
OUT = SBV2 / "output_stability"


def f0_metrics(wav: np.ndarray, sr: int) -> tuple[float, float, float]:
    import pyworld as pw
    x = wav.astype(np.float64)
    f0, t = pw.dio(x, sr, frame_period=5.0)
    f0 = pw.stonemask(x, f0, t, sr)
    v = f0[f0 > 0]
    if len(v) < 5:
        return float("nan"), float("nan"), float("nan")
    jitter = float(np.mean(np.abs(np.diff(v))) / np.mean(v))
    logf = np.log(v)
    std_log = float(np.std(logf))
    # low-frequency (<2Hz) vs tremolo band (4-9 Hz) energy of the F0 contour
    return jitter, std_log, float(np.mean(v))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_assets/asu/asu_e40_s9200.safetensors")
    ap.add_argument("--tag", default="e40")
    ap.add_argument("--text", default="おはよう、亜澄だよ。今日もいい天気だね。")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    from style_bert_vits2.constants import Languages
    from style_bert_vits2.nlp.japanese.user_dict import update_dict
    from style_bert_vits2.tts_model import TTSModel

    import torch
    update_dict()
    model = TTSModel(pathlib.Path(args.model), pathlib.Path("model_assets/asu/config.json"),
                     pathlib.Path("model_assets/asu/style_vectors.npy"), device=args.device)
    model.load()
    outdir = OUT / args.tag
    outdir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("default",       dict(sdp_ratio=0.2, noise=0.667, noise_w=0.8)),
        ("sdp0",          dict(sdp_ratio=0.0, noise=0.667, noise_w=0.8)),
        ("noise_mid",     dict(sdp_ratio=0.2, noise=0.4, noise_w=0.6)),
        ("noise_low",     dict(sdp_ratio=0.2, noise=0.2, noise_w=0.4)),
        ("sdp0_low",      dict(sdp_ratio=0.0, noise=0.3, noise_w=0.5)),
        ("inton_0.8",     dict(sdp_ratio=0.2, noise=0.4, noise_w=0.6, intonation_scale=0.8)),
        ("length_1.1",    dict(sdp_ratio=0.2, noise=0.4, noise_w=0.6, length=1.1)),
    ]
    print(f"{'config':12s} {'jitter':>8s} {'stdlogF0':>9s} {'meanF0':>7s}")
    for name, kw in configs:
        torch.manual_seed(42)
        sr, audio = model.infer(args.text, language=Languages.JP, speaker_id=0,
                                style="Neutral", **kw)
        wav = np.asarray(audio)
        sf.write(str(outdir / f"{name}.wav"), wav, sr)
        j, s, m = f0_metrics(wav, sr)
        print(f"{name:12s} {j:8.4f} {s:9.4f} {m:7.1f}")

    # seed sensitivity at default params
    print("\nseed sensitivity (default params):")
    js = []
    for seed in (1, 2, 3, 4, 5):
        torch.manual_seed(seed)
        sr, audio = model.infer(args.text, language=Languages.JP, speaker_id=0,
                                style="Neutral")
        j, s, m = f0_metrics(np.asarray(audio), sr)
        js.append(j)
        print(f"  seed={seed} jitter={j:.4f}")
    print(f"  jitter spread: {min(js):.4f}..{max(js):.4f} (mean {np.mean(js):.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
