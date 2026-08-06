#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/home/chj/LLMJailbreak/models/Llama-2-7b-chat-hf}"

cd "${REPO_ROOT}"
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python scripts/run_watermark_smoke.py \
    --config config/watermark_watermod_smoke.yaml \
    --model-path "${MODEL_PATH}" \
    --output outputs/watermark/watermod_smoke.jsonl
