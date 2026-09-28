#!/usr/bin/env python3
"""Export the fine-tuned weights to ONNX.

Style-Bert-VITS2 ships `convert_onnx.py`, but it derives the tracing inputs
with the *PyTorch* JP BERT — a ~1.3 GB download this deployment never needs,
because at inference the ONNX path runs the fp16 ONNX BERT instead. This script
builds the same inputs from that ONNX BERT, so a DirectML box downloads only
what it actually runs.

    python -m asumi_tts.export_onnx [--model ...safetensors] [--out ...onnx]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import cast

from .engine import (
    DEFAULT_MODEL_FILE,
    MODEL_DIR,
    ONNX_MODEL_FILE,
    SBV2_ROOT,
    ensure_onnx_bert,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(MODEL_DIR / DEFAULT_MODEL_FILE))
    ap.add_argument("--out", default=str(MODEL_DIR / ONNX_MODEL_FILE))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    model_path = Path(args.model).expanduser()
    out_path = Path(args.out).expanduser()
    if out_path.is_file() and not args.force:
        print(f"[export_onnx] {out_path} already exists (use --force to redo)")
        return 0

    if str(SBV2_ROOT) not in sys.path:
        sys.path.insert(0, str(SBV2_ROOT))
    ensure_onnx_bert()

    import numpy as np
    import onnx
    import torch
    from onnxsim import model_info, simplify

    from style_bert_vits2.constants import (
        DEFAULT_STYLE,
        DEFAULT_STYLE_WEIGHT,
        Languages,
    )
    from style_bert_vits2.models.infer_onnx import get_text_onnx
    from style_bert_vits2.models.models_jp_extra import (
        SynthesizerTrn as SynthesizerTrnJPExtra,
    )
    from style_bert_vits2.tts_model import TTSModel

    config_path = model_path.parent / "config.json"
    style_vec_path = model_path.parent / "style_vectors.npy"
    for p in (model_path, config_path, style_vec_path):
        if not p.is_file():
            raise FileNotFoundError(f"missing model asset: {p}")

    # The ONNX BERT runs on CPU here: it is one short sentence, and keeping the
    # export independent of DirectML means it also works headless (CI, a box
    # without a GPU) before anything is deployed.
    providers = [("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})]

    # Load the source weights with PyTorch so the graph can be traced from it.
    tts_model = TTSModel(model_path, config_path, style_vec_path, device="cpu")
    tts_model.load()
    if not isinstance(tts_model.net_g, SynthesizerTrnJPExtra):
        raise ValueError("this exporter only handles JP-Extra models")
    net_g = cast(SynthesizerTrnJPExtra, tts_model.net_g)

    # Same shape-preserving inputs the framework's converter uses, except the
    # text features come from the ONNX BERT.
    bert, ja_bert, en_bert, phones, tones, lang_ids = get_text_onnx(
        "今日はいい天気ですね。",
        Languages.JP,
        tts_model.hyper_parameters,
        onnx_providers=providers,
    )
    style_id = tts_model.style2id.get(DEFAULT_STYLE, 0)
    style_vector = tts_model.get_style_vector(style_id, DEFAULT_STYLE_WEIGHT)

    x_tst = torch.from_numpy(phones).unsqueeze(0)
    tones_t = torch.from_numpy(tones).unsqueeze(0)
    lang_ids_t = torch.from_numpy(lang_ids).unsqueeze(0)
    ja_bert_t = torch.from_numpy(np.ascontiguousarray(ja_bert)).unsqueeze(0)
    style_vec_tensor = torch.from_numpy(style_vector).unsqueeze(0)
    x_tst_lengths = torch.LongTensor([phones.shape[0]])
    sid_tensor = torch.LongTensor([0])
    length_scale = torch.tensor(1.0)
    sdp_ratio = torch.tensor(0.0)
    noise_scale = torch.tensor(0.667)
    noise_scale_w = torch.tensor(0.8)

    # JP-Extra ignores the zh/en BERT features, so only ja_bert is an input.
    def forward_jp_extra(
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        sid: torch.Tensor,
        tone: torch.Tensor,
        language: torch.Tensor,
        bert: torch.Tensor,
        style_vec: torch.Tensor,
        length_scale: float = 1.0,
        sdp_ratio: float = 0.0,
        noise_scale: float = 0.667,
        noise_scale_w: float = 0.8,
    ):
        return net_g.infer(
            x,
            x_lengths,
            sid,
            tone,
            language,
            bert,
            style_vec,
            length_scale=length_scale,
            sdp_ratio=sdp_ratio,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
        )

    net_g.forward = forward_jp_extra  # type: ignore[method-assign]

    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "raw.onnx"
        print(f"[export_onnx] tracing {model_path.name} ...")
        # dynamo=False: the default (dynamo) exporter needs onnxscript, and the
        # legacy tracer still emits the dynamic_axes signature we want.
        torch.onnx.export(
            model=net_g,
            args=(
                x_tst,
                x_tst_lengths,
                sid_tensor,
                tones_t,
                lang_ids_t,
                ja_bert_t,
                style_vec_tensor,
                length_scale,
                sdp_ratio,
                noise_scale,
                noise_scale_w,
            ),
            f=str(raw_path),
            verbose=False,
            dynamo=False,
            input_names=[
                "x_tst",
                "x_tst_lengths",
                "sid",
                "tones",
                "language",
                "bert",
                "style_vec",
                "length_scale",
                "sdp_ratio",
                "noise_scale",
                "noise_scale_w",
            ],
            output_names=["output"],
            dynamic_axes={
                "x_tst": {0: "batch_size", 1: "x_tst_max_length"},
                "x_tst_lengths": {0: "batch_size"},
                "sid": {0: "batch_size"},
                "tones": {0: "batch_size", 1: "x_tst_max_length"},
                "language": {0: "batch_size", 1: "x_tst_max_length"},
                "bert": {0: "batch_size", 2: "x_tst_max_length"},
                "style_vec": {0: "batch_size"},
            },
        )

        print("[export_onnx] simplifying ...")
        raw_model = onnx.load(raw_path)
        simplified_model, check = simplify(raw_model)
        if not check:
            raise RuntimeError("onnxsim could not verify the simplified model")
        onnx.save(simplified_model, out_path)

    size_mb = out_path.stat().st_size / 1e6
    print(f"[export_onnx] wrote {out_path} ({size_mb:.1f} MB)")
    model_info.print_simplifying_info(raw_model, simplified_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
