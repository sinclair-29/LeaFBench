#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${LEAFBENCH_ROOT:-/raid/chj/fingerprint/LeaFBench}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results/v100_fingerprint_paper}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/v100_fingerprint_paper}"
BENCHMARK_CONFIG="config/v100/paper/benchmark.yaml"
MODEL_ROOT="/raid/chj/fingerprint/models"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NLTK_DATA="${MODEL_ROOT}/nltk_data"
export GENSIM_DATA_DIR="${MODEL_ROOT}"
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${RESULTS_ROOT}" "${LOG_ROOT}"

bash scripts/v100/paper/preflight_paper.sh

run_job() {
  local gpu="$1" method="$2" source="$3" alias="$4" seed="$5" fp_config="$6" eval_config="$7"
  local job_id="gpu${gpu}_${method}_${alias}_seed${seed}"
  local job_root="${RESULTS_ROOT}/${job_id}"
  local log_file="${LOG_ROOT}/${job_id}.log"
  mkdir -p "${job_root}"
  {
    echo "[$(date --iso-8601=seconds)] START ${job_id}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" evaluation.py generate \
      --benchmark-config "${BENCHMARK_CONFIG}" \
      --fingerprint-config "${fp_config}" \
      --source-model "${source}" \
      --model-alias "${alias}" \
      --results-root "${job_root}"

    local batch_dir
    batch_dir="$(find "${job_root}" -mindepth 1 -maxdepth 1 -type d \
      -name "exp_*_${method}_seed_$(printf '%03d' "${seed}")_*" | sort | tail -n 1)"
    if [[ -z "${batch_dir}" ]]; then
      echo "No generated fingerprint batch for ${job_id}" >&2
      return 1
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" evaluation.py run \
      --benchmark-config "${BENCHMARK_CONFIG}" \
      --fingerprint-config "${fp_config}" \
      --evaluation-config "${eval_config}" \
      --batch-dir "${batch_dir}" \
      --retry-failed
    echo "[$(date --iso-8601=seconds)] DONE ${job_id}"
  } > >(tee "${log_file}") 2>&1
}

pids=() names=()
launch() { run_job "$@" & pids+=("$!"); names+=("gpu$1_$2_$4_seed$5"); }

# 16 independent single-GPU jobs. TRAP and PlugAE use three optimization seeds;
# deterministic REEF/ZeroPrint use one batch per source model.
launch 0 trap Qwen2.5-7B-Instruct qwen25_7b_instruct 42 config/v100/paper/trap_seed_42.yaml config/v100/paper/evaluation_qwen25_instruct.yaml
launch 1 trap Qwen2.5-7B-Instruct qwen25_7b_instruct 43 config/v100/paper/trap_seed_43.yaml config/v100/paper/evaluation_qwen25_instruct.yaml
launch 2 trap Qwen2.5-7B-Instruct qwen25_7b_instruct 44 config/v100/paper/trap_seed_44.yaml config/v100/paper/evaluation_qwen25_instruct.yaml
launch 3 trap Llama-2-7B-Chat llama2_7b_chat 42 config/v100/paper/trap_llama_seed_42.yaml config/v100/paper/evaluation_llama2_chat.yaml
launch 4 trap Llama-2-7B-Chat llama2_7b_chat 43 config/v100/paper/trap_llama_seed_43.yaml config/v100/paper/evaluation_llama2_chat.yaml
launch 5 trap Llama-2-7B-Chat llama2_7b_chat 44 config/v100/paper/trap_llama_seed_44.yaml config/v100/paper/evaluation_llama2_chat.yaml
launch 6 plugae Qwen2.5-7B qwen25_7b_base 42 config/v100/paper/plugae_seed_42.yaml config/v100/paper/evaluation_qwen25_base.yaml
launch 7 plugae Qwen2.5-7B qwen25_7b_base 43 config/v100/paper/plugae_seed_43.yaml config/v100/paper/evaluation_qwen25_base.yaml
launch 8 plugae Qwen2.5-7B qwen25_7b_base 44 config/v100/paper/plugae_seed_44.yaml config/v100/paper/evaluation_qwen25_base.yaml
launch 9 plugae Llama-2-7B llama2_7b_base 42 config/v100/paper/plugae_seed_42.yaml config/v100/paper/evaluation_llama2_base.yaml
launch 10 plugae Llama-2-7B llama2_7b_base 43 config/v100/paper/plugae_seed_43.yaml config/v100/paper/evaluation_llama2_base.yaml
launch 11 plugae Llama-2-7B llama2_7b_base 44 config/v100/paper/plugae_seed_44.yaml config/v100/paper/evaluation_llama2_base.yaml
launch 12 zeroprint Qwen2.5-7B-Instruct qwen25_7b_instruct 1000 config/v100/paper/zeroprint_seed_1000.yaml config/v100/paper/evaluation_qwen25_instruct.yaml
launch 13 zeroprint Llama-2-7B-Chat llama2_7b_chat 1000 config/v100/paper/zeroprint_llama_seed_1000.yaml config/v100/paper/evaluation_llama2_chat.yaml
launch 14 reef Qwen2.5-7B-Instruct qwen25_7b_instruct 42 config/v100/paper/reef_seed_42.yaml config/v100/paper/evaluation_qwen25_reef.yaml
launch 15 reef Llama-2-7B-Chat llama2_7b_chat 42 config/v100/paper/reef_seed_42.yaml config/v100/paper/evaluation_llama2_reef.yaml

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "PASS ${names[$index]}"
  else
    echo "FAIL ${names[$index]}" >&2
    failed=1
  fi
done

"${PYTHON_BIN}" scripts/v100/paper/summarize_paper.py "${RESULTS_ROOT}"
exit "${failed}"
