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

from .engine import AsumiTTSEngine

DEVICE = os.environ.get("ASUMI_DEVICE", "mps")
_engine: AsumiTTSEngine | None = None


class TTSRequest(BaseModel):
    text: str
    style: str | None = None
    intonation_scale: float = 1.0
    length: float = 1.0
    sdp_ratio: float = 0.2
    noise: float = 0.667
    noise_w: float = 0.8


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    _engine = AsumiTTSEngine(device=DEVICE)
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
