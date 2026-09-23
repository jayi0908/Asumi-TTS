"""Measure inference memory footprint for the asu JP-Extra model."""
import pathlib
import resource
import time

import numpy as np
import soundfile as sf
import torch

from style_bert_vits2.constants import Languages
from style_bert_vits2.tts_model import TTSModel

OUT = pathlib.Path("/var/folders/5l/ctshz4697l72txt0wmp2gbl00000gn/T/opencode/sbv2_infer.wav")

t0 = time.time()
model = TTSModel(
    pathlib.Path("Data/asu/models/G_0.safetensors"),
    pathlib.Path("Data/asu/config.json"),
    pathlib.Path("model_assets/asu/style_vectors.npy"),
    device="mps",
)
model.load()
print(f"loaded in {time.time() - t0:.1f}s", flush=True)

text = "こんにちは、私は朝美です。今日はいい天気ですね。"
t1 = time.time()
sr, audio = model.infer(text, language=Languages.JP, speaker_id=0, style="Neutral")
dur = len(audio) / sr
sf.write(str(OUT), np.asarray(audio), sr)
print(f"infer: audio={dur:.2f}s in {time.time() - t1:.1f}s  rt={dur / (time.time() - t1):.2f}x", flush=True)
print(
    "mps alloc=%.2fGB driver=%.2fGB  maxrss=%.0fMB"
    % (
        torch.mps.current_allocated_memory() / 1e9,
        torch.mps.driver_allocated_memory() / 1e9,
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024,
    ),
    flush=True,
)
print("holding model for 40s ...", flush=True)
time.sleep(120)
