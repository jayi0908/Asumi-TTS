#!/usr/bin/env python3
"""Download the fine-tuned weights from the Hugging Face Hub.

The weights are ~240 MB, past what GitHub accepts, so they live on the Hub and
the two small model files stay here. `asumi_tts` fetches them on first use too;
this script exists to do it up front, e.g. before starting the service.

    HF_TOKEN=... python fetch_model.py

The Hub repository is private, so a token (or a prior `hf auth login`) is
required. Override the target with ASUM_ HF_REPO if you keep your own copy.
"""
from asumi_tts.engine import MODEL_DIR, ensure_weights

if __name__ == "__main__":
    ensure_weights()
    print(f"weights are in {MODEL_DIR}")
