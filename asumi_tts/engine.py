"""Asumi (亜澄) Japanese TTS engine — thin wrapper around Style-Bert-VITS2.

The heavy lifting lives in the Style-Bert-VITS2 checkout; this module only
loads the fine-tuned 'asu' JP-Extra model once and exposes a simple
`speak(text) -> (sample_rate, float32 waveform)` API.

Must run inside the Style-Bert-VITS2 conda env (`sbv2`).
"""
from __future__ import annotations

import json
import os
import subprocess
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

# The ONNX sibling of the weights. It is not on the Hub: DirectML — the only
# GPU route on a Windows box without CUDA — runs ONNX, so the file is exported
# locally from the safetensors above by asumi_tts.export_onnx.
ONNX_MODEL_FILE = os.environ.get(
    "ASUMI_ONNX_MODEL", str(Path(DEFAULT_MODEL_FILE).with_suffix(".onnx"))
)

# The JP BERT the ONNX path runs. Style-Bert-VITS2 ships its tokenizer but not
# the ~650 MB fp16 weights, so they are fetched here just like the model.
ONNX_BERT_REPO = os.environ.get(
    "ASUMI_ONNX_BERT_REPO", "tsukumijima/deberta-v2-large-japanese-char-wwm-onnx"
)

# Which JP ONNX BERT to run: 'fp16' (the framework's, ~650 MB, default) or
# 'fp32' (the same model at full precision, ~1.3 GB, numerically closer to the
# PyTorch path). Handy for A/B-ing whether the fp16 weights are what makes the
# emotion feel thinner than on MPS. Overridable per engine / per CLI run.
ASUMI_ONNX_BERT = "ASUMI_ONNX_BERT"
_ONNX_BERT_FP32_DIR_SUFFIX = "-fp32"


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


def default_device() -> str:
    """Pick the fastest backend this machine can actually run.

    On Windows this is ONNX Runtime on the CPU ('onnx'): PyTorch there is
    CPU-only without CUDA, and DirectML — while it can use the iGPU — rebuilds
    its kernels for every new input length, which costs ~13 s per unseen
    sentence and makes it a poor fit for free-form text (see README). macOS
    keeps PyTorch/MPS. `ASUMI_DEVICE` overrides all of this.
    """
    override = os.environ.get("ASUMI_DEVICE", "").strip()
    if override:
        return override
    if sys.platform == "win32":
        return "onnx"
    if sys.platform == "darwin":
        return "mps"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def onnx_providers_for(device: str):
    """onnxruntime execution providers for one of our device names.

    'onnx' is ONNX Runtime on the CPU; 'dml' is the DirectML GPU path. The CPU
    provider is always last, so a GPU that turns out to be unusable only costs
    speed — never correctness.
    """
    cpu = ("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})
    if device in ("dml", "directml"):
        # device_id 0 is the primary display adapter, i.e. the iGPU on a laptop.
        return [("DmlExecutionProvider", {"device_id": 0}), cpu]
    if device.startswith("cuda"):
        return [
            (
                "CUDAExecutionProvider",
                {
                    "arena_extend_strategy": "kSameAsRequested",
                    "cudnn_conv_algo_search": "DEFAULT",
                },
            ),
            cpu,
        ]
    return [cpu]


def ensure_onnx_bert() -> None:
    """Fetch the JP ONNX BERT weights when the ONNX path needs them."""
    _ensure_sbv2_path()
    from style_bert_vits2.constants import DEFAULT_ONNX_BERT_MODEL_PATHS, Languages

    target = DEFAULT_ONNX_BERT_MODEL_PATHS[Languages.JP]
    if (target / "model_fp16.onnx").is_file():
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise FileNotFoundError(
            f"the ONNX BERT weights are missing at {target / 'model_fp16.onnx'}, "
            f"and huggingface_hub is not installed to fetch them; install it, or "
            f"place model_fp16.onnx there"
        ) from error

    print(f"[asumi_tts] fetching the JP ONNX BERT into {target} ...")
    hf_hub_download(
        repo_id=ONNX_BERT_REPO,
        filename="model_fp16.onnx",
        local_dir=str(target),
    )


def resolve_onnx_bert(value: str | None = None) -> str:
    """Normalize the BERT precision switch to 'fp16' or 'fp32'."""
    raw = value if value is not None else os.environ.get(ASUMI_ONNX_BERT, "fp16")
    return "fp32" if str(raw).strip().lower() in ("fp32", "float32", "1", "true", "yes", "on") else "fp16"


def _ensure_sbv2_path() -> None:
    """Make the framework importable even when these helpers are used alone."""
    if str(SBV2_ROOT) not in sys.path:
        sys.path.insert(0, str(SBV2_ROOT))


def _onnx_bert_dirs() -> tuple[Path, Path]:
    """(framework's fp16 dir, our fp32 dir) for the JP ONNX BERT."""
    _ensure_sbv2_path()
    from style_bert_vits2.constants import DEFAULT_ONNX_BERT_MODEL_PATHS, Languages

    fp16_dir = DEFAULT_ONNX_BERT_MODEL_PATHS[Languages.JP]
    return fp16_dir, fp16_dir.with_name(fp16_dir.name + _ONNX_BERT_FP32_DIR_SUFFIX)


def ensure_onnx_bert_fp32() -> Path:
    """Fetch the fp32 JP ONNX BERT and return the dir to load it from.

    The Hub file is `model.onnx`, but the framework's loader always looks for
    `model_fp16.onnx`, so the fp32 copy lives in its own directory under that
    name — the directory says which precision it really is.
    """
    _fp16_dir, fp32_dir = _onnx_bert_dirs()
    target = fp32_dir / "model_fp16.onnx"
    if target.is_file():
        return fp32_dir

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise FileNotFoundError(
            f"the fp32 ONNX BERT weights are missing at {target}, and "
            f"huggingface_hub is not installed to fetch them; install it, or "
            f"unset {ASUMI_ONNX_BERT}"
        ) from error

    print(f"[asumi_tts] fetching the fp32 JP ONNX BERT into {fp32_dir} ...")
    hf_hub_download(
        repo_id=ONNX_BERT_REPO,
        filename="model.onnx",
        local_dir=str(fp32_dir),
    )
    # hf_hub_download keeps the Hub name; move it to the name the loader wants.
    downloaded = fp32_dir / "model.onnx"
    if downloaded.is_file():
        downloaded.replace(target)
    if not target.is_file():
        raise RuntimeError(f"the fp32 ONNX BERT download finished but {target} is missing")
    return fp32_dir


def convert_to_onnx(force: bool = False) -> Path:
    """Export the fine-tuned weights to ONNX, once.

    The Hub only holds the .safetensors weights, so the ONNX file the DirectML
    path runs is produced locally by `asumi_tts.export_onnx` (which needs
    `onnx`/`onnxsim`, and torch to do the export).
    """
    dst = MODEL_DIR / ONNX_MODEL_FILE
    if dst.is_file() and not force:
        return dst

    ensure_weights()  # the exporter reads the safetensors and config.json next to it
    print(f"[asumi_tts] exporting {ONNX_MODEL_FILE} from {DEFAULT_MODEL_FILE} ...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "asumi_tts.export_onnx",
            "--model",
            str(MODEL_DIR / DEFAULT_MODEL_FILE),
            "--out",
            str(dst),
        ]
        + (["--force"] if force else []),
        cwd=str(REPO_ROOT),
        check=True,
    )
    if not dst.is_file():
        raise RuntimeError(f"the ONNX export finished but {dst} is missing")
    return dst


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
        model_file: str | None = None,
        device: str | None = None,
        style: str = "Neutral",
        warmup: bool | None = None,
        onnx_bert: str | None = None,
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

        device = device or default_device()
        self.device = device
        self.style = style
        self.onnx_bert = resolve_onnx_bert(onnx_bert)

        # DirectML and 'onnx' (CPU) are both served by onnxruntime, everything
        # else by PyTorch. An explicit .onnx file name selects ONNX as well.
        self.use_onnx = device in ("dml", "directml", "onnx") or Path(
            model_file or DEFAULT_MODEL_FILE
        ).suffix == ".onnx"
        if self.use_onnx:
            self._prepare_onnx_bert(device)
            self.model_path = convert_to_onnx()
        else:
            ensure_weights()
            self.model_path = MODEL_DIR / (model_file or DEFAULT_MODEL_FILE)

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
            self.model_path,
            self.config_path,
            self.style_vec_path,
            device=device,
            onnx_providers=onnx_providers_for(device),
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

    def _prepare_onnx_bert(self, device: str) -> None:
        """Load the JP ONNX BERT, fp16 or fp32, into the framework's cache.

        The framework's ONNX feature extractor always asks the default path for
        `model_fp16.onnx`. Seeding the cache here — before the first synthesis —
        is what makes the fp32 switch work without patching the framework.
        """
        from style_bert_vits2.nlp import onnx_bert_models

        if self.onnx_bert == "fp32":
            bert_dir = ensure_onnx_bert_fp32()
            onnx_bert_models.load_model(
                self._Languages.JP,
                pretrained_model_name_or_path=str(bert_dir),
                onnx_providers=onnx_providers_for(device),
            )
        else:
            ensure_onnx_bert()
        print(f"[asumi_tts] ONNX BERT: {self.onnx_bert}", flush=True)

    def warmup(self) -> float:
        """Run one throwaway synthesis so the first real request is fast.

        The JP BERT (tokenizer plus a ~650 MB ONNX model on the DirectML path,
        ~1.3 GB of PyTorch weights otherwise) and the GPU kernels are loaded
        lazily on the first `infer()`, which costs several seconds. Doing it
        here moves that cost to start-up instead of the user's first utterance.
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
