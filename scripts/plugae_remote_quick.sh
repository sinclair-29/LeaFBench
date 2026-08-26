#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${LEAFBENCH_ROOT:-/raid/chj/fingerprint/LeaFBench}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results/plugae_remote_quick/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/plugae_remote_quick/${RUN_ID}}"
BENCHMARK_CONFIG="config/benchmark_plugae_remote_quick.yaml"
FINGERPRINT_CONFIG="config/plugae_remote_quick.yaml"
EVALUATION_CONFIG="config/evaluation_plugae_remote_quick.yaml"
SOURCE_MODEL="Gemma-2-2B"
MODEL_ALIAS="gemma2_2b_quick"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "${RESULTS_ROOT}" "${LOG_ROOT}"

{
  echo "PlugAE quick test: generating fingerprint on GPU ${GPU_ID}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -u evaluation.py generate \
    --benchmark-config "${BENCHMARK_CONFIG}" \
    --fingerprint-config "${FINGERPRINT_CONFIG}" \
    --source-model "${SOURCE_MODEL}" \
    --model-alias "${MODEL_ALIAS}" \
    --results-root "${RESULTS_ROOT}"

  BATCH_DIR="$("${PYTHON_BIN}" - "${RESULTS_ROOT}" "${MODEL_ALIAS}" <<'PY'
import sys
from pathlib import Path

import evaluation

root, alias = sys.argv[1:]
prefix = f"exp_{alias}_plugae_seed_042_"
candidates = []
for path in Path(root).glob(f"{prefix}*"):
    if (path / "fingerprint_config.json").is_file():
        _, variant = evaluation.experiment_variant(path)
        candidates.append((evaluation.variant_to_number(variant), path))
if not candidates:
    raise SystemExit("No PlugAE fingerprint batch was generated.")
print(max(candidates)[1])
PY
)"

  echo "PlugAE quick test: evaluating ${BATCH_DIR}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -u evaluation.py run \
    --benchmark-config "${BENCHMARK_CONFIG}" \
    --fingerprint-config "${FINGERPRINT_CONFIG}" \
    --evaluation-config "${EVALUATION_CONFIG}" \
    --batch-dir "${BATCH_DIR}" \
    --retry-failed \
    --overwrite

  echo
  echo "Completed. Results: ${BATCH_DIR}"
  echo "Inspect: ${BATCH_DIR}/model_modification_robustness.json"
  echo "Inspect: ${BATCH_DIR}/model_specificity.json"
} 2>&1 | tee "${LOG_ROOT}/plugae_quick.log"
