#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${LEAFBENCH_ROOT:-/raid/chj/fingerprint/LeaFBench}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results/v100_validation}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/v100_validation}"
BENCHMARK_CONFIG="config/v100/benchmark_local_validation.yaml"
MODEL_ROOT="/raid/chj/fingerprint/models"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NLTK_DATA="${MODEL_ROOT}/nltk_data"
export GENSIM_DATA_DIR="${MODEL_ROOT}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
bash scripts/v100/preflight.sh
mkdir -p "${RESULTS_ROOT}" "${LOG_ROOT}"

run_method() {
  local method="$1"
  local devices="$2"
  local fingerprint_config="$3"
  local source_model="$4"
  local model_alias="$5"
  local seed="$6"
  local evaluation_config="$7"
  local method_results="${RESULTS_ROOT}/${method}"
  local log_file="${LOG_ROOT}/${method}.log"

  mkdir -p "${method_results}"
  {
    echo "[$(date --iso-8601=seconds)] ${method}: fingerprint generation on GPUs ${devices}"
    CUDA_VISIBLE_DEVICES="${devices}" HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      "${PYTHON_BIN}" evaluation.py generate \
        --benchmark-config "${BENCHMARK_CONFIG}" \
        --fingerprint-config "${fingerprint_config}" \
        --source-model "${source_model}" \
        --model-alias "${model_alias}" \
        --results-root "${method_results}"

    local batch_dir
    batch_dir="$("${PYTHON_BIN}" - "${method_results}" "${model_alias}" "${method}" "${seed}" <<'PY'
import sys
from pathlib import Path

import evaluation

root, alias, method, seed = sys.argv[1:]
prefix = f"exp_{alias}_{method}_seed_{int(seed):03d}_"
candidates = []
for path in Path(root).glob(f"{prefix}*"):
    if not (path / "fingerprint_config.json").is_file():
        continue
    _, variant = evaluation.experiment_variant(path)
    candidates.append((evaluation.variant_to_number(variant), path))
if candidates:
    print(max(candidates)[1])
PY
)"
    if [[ -z "${batch_dir}" ]]; then
      echo "Could not locate generated ${method} fingerprint batch." >&2
      return 1
    fi

    echo "[$(date --iso-8601=seconds)] ${method}: semantic evaluation"
    CUDA_VISIBLE_DEVICES="${devices}" HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      "${PYTHON_BIN}" evaluation.py run \
        --benchmark-config "${BENCHMARK_CONFIG}" \
        --fingerprint-config "${fingerprint_config}" \
        --evaluation-config "${evaluation_config}" \
        --batch-dir "${batch_dir}" \
        --retry-failed
  } 2>&1 | tee "${log_file}"
}

pids=()
names=()

run_method trap "0,1,2,3" config/v100/trap_validation.yaml \
  Qwen2.5-7B-Instruct qwen25_v100 42 \
  config/v100/evaluation_qwen25_instruct_validation.yaml &
pids+=("$!"); names+=(trap)

run_method plugae "4,5,6,7" config/v100/plugae_validation.yaml \
  Qwen2.5-7B qwen25_base_v100 42 \
  config/v100/evaluation_qwen25_base_validation.yaml &
pids+=("$!"); names+=(plugae)

run_method zeroprint "8,9,10,11" config/v100/zeroprint_validation.yaml \
  Qwen2.5-7B-Instruct qwen25_v100 1000 \
  config/v100/evaluation_qwen25_instruct_validation.yaml &
pids+=("$!"); names+=(zeroprint)

run_method reef "12,13,14,15" config/v100/reef_validation.yaml \
  Qwen2.5-7B-Instruct qwen25_v100 42 \
  config/v100/evaluation_qwen25_instruct_validation.yaml &
pids+=("$!"); names+=(reef)

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "${names[$index]} completed."
  else
    echo "${names[$index]} failed; inspect ${LOG_ROOT}/${names[$index]}.log" >&2
    failed=1
  fi
done

validation_failed=0
if ! bash scripts/v100/validate_results.sh "${RESULTS_ROOT}"; then
  validation_failed=1
fi

if (( failed || validation_failed )); then
  exit 1
fi
