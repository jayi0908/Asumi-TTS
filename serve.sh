#!/usr/bin/env bash
# Serve the voice over HTTP, the way the app runs it.
#
# Usage: ./serve.sh [port]        (default 8077)
#
# The app starts this itself when the setting "lazy start" is on; this script is
# for running it by hand, e.g. to warm the model before first use.
set -euo pipefail

PORT="${1:-8077}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The framework is expected next to this repository; override if it is not.
export SBV2_ROOT="${SBV2_ROOT:-$(dirname "$REPO_ROOT")/Style-Bert-VITS2}"

if [[ ! -d "$SBV2_ROOT/style_bert_vits2" ]]; then
  echo "Style-Bert-VITS2 not found at $SBV2_ROOT" >&2
  echo "clone it there, or export SBV2_ROOT=/path/to/Style-Bert-VITS2" >&2
  exit 1
fi

exec "${ASUMI_PYTHON:-python3}" -m uvicorn asumi_tts.server:app \
  --host 127.0.0.1 --port "$PORT"
