#!/usr/bin/env python3
"""HTTP microservice for the Asumi (亜澄) TTS engine.

Start:
  conda run -n sbv2 python -m uvicorn asumi_tts.server:app --host 127.0.0.1 --port 8077

Endpoints:
  GET  /health                 -> {"status": "ok", "device": "mps"}
  POST /tts                    -> body {"text": "...", ...}  returns audio/wav
  POST /tts/base64             -> {"text": ...}              returns {"sample_rate", "audio_base64"}

Example:
  curl -X POST http://127.0.0.1:8077/tts \
       -H 'Content-Type: application/json' \
       -d '{"text":"おはよう、亜澄だよ。"}' --output hello.wav
"""
from __future__ import annotations

import base64
import io
import os
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from .engine import AsumiTTSEngine, default_device

DEVICE = default_device()
STYLE = os.environ.get("ASUMI_STYLE", "Neutral")
_engine: AsumiTTSEngine | None = None


def _env_float(name: str, default: float) -> float:
    """Read a numeric override from the environment, ignoring junk."""
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


# Speech defaults. The Agent sends only `text`, so these are what actually
# decide how lively the output is; overriding them (e.g. ASUM_ SDP_RATIO=0.4)
# tunes expressiveness without touching the app.
DEFAULT_INTONATION_SCALE = _env_float("ASUMI_INTONATION_SCALE", 1.0)
DEFAULT_LENGTH = _env_float("ASUMI_LENGTH", 1.0)
DEFAULT_SDP_RATIO = _env_float("ASUMI_SDP_RATIO", 0.2)
DEFAULT_NOISE = _env_float("ASUMI_NOISE", 0.667)
DEFAULT_NOISE_W = _env_float("ASUMI_NOISE_W", 0.8)


class TTSRequest(BaseModel):
    text: str
    style: str | None = None
    intonation_scale: float = DEFAULT_INTONATION_SCALE
    length: float = DEFAULT_LENGTH
    sdp_ratio: float = DEFAULT_SDP_RATIO
    noise: float = DEFAULT_NOISE
    noise_w: float = DEFAULT_NOISE_W


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    # Warm up here, not in the engine: the server answers many requests, so
    # paying the JP BERT load at start-up keeps the first /tts call fast
    # (~1 s instead of ~12 s). Set ASUMI_WARMUP=0 to trade that back.
    warmup = os.environ.get("ASUMI_WARMUP", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )
    _engine = AsumiTTSEngine(device=DEVICE, warmup=warmup, style=STYLE)
    print(f"[asumi_tts] engine ready on {DEVICE}", flush=True)
    yield
    _engine = None


app = FastAPI(title="Asumi TTS", version="1.0.0", lifespan=lifespan)


def _synth(req: TTSRequest) -> tuple[int, np.ndarray]:
    if _engine is None:
        raise HTTPException(503, "model not loaded")
    if not req.text.strip():
        raise HTTPException(400, "text is empty")
    return _engine.speak(
        req.text,
        style=req.style,
        intonation_scale=req.intonation_scale,
        length=req.length,
        sdp_ratio=req.sdp_ratio,
        noise=req.noise,
        noise_w=req.noise_w,
    )


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok" if _engine else "loading", "device": DEVICE})


@app.post("/tts")
def tts(req: TTSRequest) -> Response:
    sr, wav = _synth(req)
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    return Response(content=buf.getvalue(), media_type="audio/wav")


@app.post("/tts/base64")
def tts_base64(req: TTSRequest) -> JSONResponse:
    sr, wav = _synth(req)
    pcm16 = (np.clip(wav, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    return JSONResponse({
        "sample_rate": sr,
        "format": "pcm_s16le",
        "audio_base64": base64.b64encode(pcm16).decode("ascii"),
    })
