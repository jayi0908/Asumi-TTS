"""Minimal example: use the TTS engine in-process (recommended if your app is Python).

Run inside the Style-Bert-VITS2 env:
  conda run -n sbv2 python asumi_voice/asumi_tts/examples/python_api.py
"""
import pathlib
import sys

# make `asumi_tts` importable when running this file directly
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from asumi_tts import AsumiTTSEngine  # noqa: E402

tts = AsumiTTSEngine(device="mps")  # loads once (~2 s after warm-up)

# 1) get a waveform in memory
sr, wav = tts.speak("おはよう、亜澄だよ。今日もいい天気だね。")
print("sample_rate:", sr, "duration(s):", round(len(wav) / sr, 2))

# 2) write straight to a file
out = pathlib.Path(__file__).resolve().parents[1] / "outputs" / "example.wav"
tts.speak_to_file("また明日も会えるといいな。", out)
print("wrote", out)
