"""Example HTTP client for the Asumi TTS microservice.

Start the server first:
  conda run -n sbv2 python -m uvicorn asumi_tts.server:app --host 127.0.0.1 --port 8077
Then:
  conda run -n sbv2 python asumi_voice/asumi_tts/examples/http_client.py
"""
import pathlib

import requests

URL = "http://127.0.0.1:8077/tts"

resp = requests.post(URL, json={"text": "こんにちは、亜澄です。よろしくお願いします。"}, timeout=120)
resp.raise_for_status()

out = pathlib.Path(__file__).resolve().parents[1] / "outputs" / "http_example.wav"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_bytes(resp.content)
print("wrote", out, len(resp.content), "bytes")
