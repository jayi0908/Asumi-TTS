"""Asumi (亜澄) Japanese TTS engine — thin wrapper around Style-Bert-VITS2.

The heavy lifting lives in the Style-Bert-VITS2 checkout; this module only
loads the fine-tuned 'asu' JP-Extra model once and exposes a simple
`speak(text) -> (sample_rate, float32 waveform)` API.

Must run inside the Style-Bert-VITS2 conda env (`sbv2`).
"""
from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

import numpy as np

# --- paths --------------------------------------------------------------------
# Two different things live at two different places:
#
# * the model is this repository's own product, so it ships inside it;
# * Style-Bert-VITS2 is the framework this wraps, cloned separately because it
#   carries the BERT weights and the dictionary compiler.
#
# Each is overridable, but the defaults make a plain checkout work as long as
# the framework sits next to this repository.
REPO_ROOT = Path(__file__).resolve().parent.parent
SBV2_ROOT = Path(
    os.environ.get("SBV2_ROOT", str(REPO_ROOT.parent / "Style-Bert-VITS2"))
).expanduser()
MODEL_DIR = Path(
    os.environ.get("ASUMI_MODEL_DIR", str(REPO_ROOT / "model_assets" / "asu"))
).expanduser()
DICT_PATH = Path(
    os.environ.get("ASUMI_DICT", str(REPO_ROOT / "dict_data" / "default.csv"))
).expanduser()
DEFAULT_MODEL_FILE = os.environ.get("ASUMI_MODEL", "asu_e40_s9200.safetensors")

# Where the weights live. They are ~240 MB, well past what a git host will take,
# so only the two small model files are in the repository and the weights come
# from the Hub. That repository is private: set HF_TOKEN, or run `hf auth login`
# once, with an account allowed to read it.
HF_REPO_ID = os.environ.get("ASUMI_HF_REPO", "jayi0908/style-bert-vits2-asumi")


def ensure_weights() -> None:
    """Fetch the fine-tuned weights when this checkout does not have them."""
    target = MODEL_DIR / DEFAULT_MODEL_FILE
    if target.is_file():
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise FileNotFoundError(
            f"model weights are missing at {target}, and huggingface_hub is not "
            f"installed to fetch them; install it, or place {DEFAULT_MODEL_FILE} "
            f"in {MODEL_DIR}"
        ) from error

    print(f"[asumi_tts] fetching {DEFAULT_MODEL_FILE} from {HF_REPO_ID} ...")
    hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=DEFAULT_MODEL_FILE,
        local_dir=str(MODEL_DIR),
    )

_SENTINEL = object()


class AsumiTTSEngine:
    """Load once, synthesize many times.

    Example
    -------
    >>> from asumi_tts import AsumiTTSEngine
    >>> tts = AsumiTTSEngine()
    >>> sr, wav = tts.speak("おはよう、亜澄だよ。")
    """

    def __init__(
        self,
        model_file: str = DEFAULT_MODEL_FILE,
        device: str = "mps",
        style: str = "Neutral",
        warmup: bool | None = None,
    ) -> None:
        if str(SBV2_ROOT) not in sys.path:
            sys.path.insert(0, str(SBV2_ROOT))

        from style_bert_vits2.constants import Languages
        from style_bert_vits2.nlp.japanese.user_dict import update_dict
        from style_bert_vits2.tts_model import TTSModel

        self._Languages = Languages
        # Our dictionary, not the framework's: the entries that make proper
        # nouns come out right (亜澄 -> アスミ, 錦亜澄 -> ニシキアスミ) are part of
        # this model, not of the framework.
        update_dict(default_dict_path=DICT_PATH)

        self.device = device
        self.style = style
        ensure_weights()
        self.model_path = MODEL_DIR / model_file
        self.config_path = MODEL_DIR / "config.json"
        self.style_vec_path = MODEL_DIR / "style_vectors.npy"
        for p in (self.model_path, self.config_path, self.style_vec_path):
            if not p.exists():
                raise FileNotFoundError(f"missing model asset: {p}")

        # `TTSModel.infer()` normalizes its output and returns 16-bit ints
        # (convert_to_16_bit_wav), so the values arrive on the ±32768 scale, not
        # the ±1 scale a float waveform means. Rescaling here is what the
        # official server gets for free from writing the int16 array natively;
        # skipping it makes every downstream writer clip almost every sample
        # into a square wave (gross distortion plus broadband hiss). The divisor
        # is read from the model's own config, so the weights stay untouched.
        with open(self.config_path, encoding="utf-8") as fh:
            self.max_wav_value = float(json.load(fh).get("max_wav_value", 32768.0))

        self._tts = TTSModel(
            self.model_path, self.config_path, self.style_vec_path, device=device
        )
        self._tts.load()
        self._lock = threading.Lock()

        # Off by default: a one-shot CLI/API call pays the BERT load either way,
        # so warming first would only add a throwaway synthesis. The HTTP server
        # passes warmup=True, where it moves the cost to start-up instead of the
        # user's first utterance. ASUMI_WARMUP can force it either way.
        if warmup is None:
            warmup = os.environ.get("ASUMI_WARMUP", "0").strip().lower() not in (
                "0", "false", "no", "off",
            )
        if warmup:
            self.warmup()

    def warmup(self) -> float:
        """Run one throwaway synthesis so the first real request is fast.

        The JP BERT (tokenizer plus a ~1.3 GB model) and the MPS kernels are
        loaded lazily on the first `infer()`, which costs ~10 s. Doing it here
        moves that cost to start-up instead of the user's first utterance.
        Returns the number of seconds it took; never raises.
        """
        import time

        started = time.time()
        try:
            with self._lock:
                self._tts.infer(
                    "これはウォームアップのためのテスト文です。",
                    language=self._Languages.JP,
                    speaker_id=0,
                    style=self.style,
                )
        except Exception as error:  # pragma: no cover - never break start-up
            print(f"[asumi_tts] warm-up failed: {error}")
        elapsed = time.time() - started
        print(f"[asumi_tts] warm-up finished in {elapsed:.1f}s")
        return elapsed

    def speak(
        self,
        text: str,
        *,
        style: str | None = None,
        intonation_scale: float = 1.0,
        sdp_ratio: float = 0.2,
        noise: float = 0.667,
        noise_w: float = 0.8,
        length: float = 1.0,
    ) -> tuple[int, np.ndarray]:
        """Synthesize `text` and return (sample_rate, float32 waveform)."""
        text = (text or "").strip()
        if not text:
            raise ValueError("text must not be empty")
        with self._lock:
            sr, audio = self._tts.infer(
                text,
                language=self._Languages.JP,
                speaker_id=0,
                style=style or self.style,
                intonation_scale=intonation_scale,
                sdp_ratio=sdp_ratio,
                noise=noise,
                noise_w=noise_w,
                length=length,
            )
        wav = np.asarray(audio, dtype=np.float32) / self.max_wav_value
        # The peak can still land a hair above full scale; attenuate the whole
        # line rather than let a writer clip it.
        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        if peak > 1.0:
            wav = wav * (0.99 / peak)
        return int(sr), wav

    def speak_to_file(self, text: str, path: str | Path, **kw) -> Path:
        import soundfile as sf

        sr, wav = self.speak(text, **kw)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), wav, sr)
        return path


_ENGINE: object = _SENTINEL


def get_engine(**kwargs) -> AsumiTTSEngine:
    """Process-wide singleton so the model is loaded only once."""
    global _ENGINE
    if _ENGINE is _SENTINEL:
        _ENGINE = AsumiTTSEngine(**kwargs)
    return _ENGINE  # type: ignore[return-value]
