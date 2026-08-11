#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${LEAFBENCH_ROOT:-/raid/chj/fingerprint}"
MODEL_ROOT="/raid/chj/fingerprint/models"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${PROJECT_ROOT}"

required_paths=(
  "${MODEL_ROOT}/Qwen2.5-7B"
  "${MODEL_ROOT}/Qwen2.5-7B-Instruct"
  "${MODEL_ROOT}/Llama-2-7b-hf"
  "${MODEL_ROOT}/Llama-2-7b-chat-hf"
  "${MODEL_ROOT}/Mistral-7B-v0.3"
  "${MODEL_ROOT}/Mistral-7B-Instruct-v0.3"
  "${MODEL_ROOT}/TinyLlama-1.1B-Chat-v1.0"
  "${MODEL_ROOT}/all-mpnet-base-v2"
  "${MODEL_ROOT}/nltk_data"
  "${MODEL_ROOT}/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz"
)

missing=0
for path in "${required_paths[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    missing=1
  fi
done
if (( missing )); then
  echo "For ZeroPrint resources, run: bash scripts/setup_zeroprint_resources.sh ${MODEL_ROOT}" >&2
  exit 1
fi

gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
if (( gpu_count < 16 )); then
  echo "Expected at least 16 GPUs, found ${gpu_count}." >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import yaml
import torch

from fingerprint.fingerprint_factory import create_fingerprint_method

paths = sorted(Path("config/v100").glob("*.yaml"))
assert paths, "No V100 YAML files found"
for path in paths:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    assert isinstance(value, dict), f"Invalid YAML root: {path}"
print(f"Parsed {len(paths)} V100 YAML files")

for name in ("trap", "plugae", "zeroprint", "reef"):
    path = Path("config/v100") / f"{name}_validation.yaml"
    with path.open(encoding="utf-8") as stream:
        create_fingerprint_method(yaml.safe_load(stream))
print("Constructed all four fingerprint methods")

assert torch.cuda.is_available(), "PyTorch cannot see CUDA"
assert torch.cuda.device_count() == 16, (
    f"Expected 16 visible CUDA devices, found {torch.cuda.device_count()}"
)
print(f"PyTorch {torch.__version__}, CUDA devices: {torch.cuda.device_count()}")
PY

"${PYTHON_BIN}" -m unittest tests.test_evaluation
echo "V100 validation preflight passed."
