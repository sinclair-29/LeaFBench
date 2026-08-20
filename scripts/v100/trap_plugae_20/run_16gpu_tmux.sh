#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${LEAFBENCH_ROOT:-/raid/chj/fingerprint/LeaFBench}"
CONFIG_ROOT="config/v100/trap_plugae_20"
BENCHMARK_CONFIG="config/v100/paper/benchmark.yaml"
MODEL_ROOT="/raid/chj/fingerprint/models"
SCRIPT_PATH="${PROJECT_ROOT}/scripts/v100/trap_plugae_20/run_16gpu_tmux.sh"

run_worker() {
  local gpu="$1" method="$2" source_model="$3" model_alias="$4"
  local fingerprint_config="$5" evaluation_config="$6" run_id="$7"
  local python_bin="$8"
  local job_id="gpu$(printf '%02d' "${gpu}")_${method}_${model_alias}"
  local results_root="${PROJECT_ROOT}/results/v100_trap_plugae_20/${run_id}"
  local job_root="${results_root}/${job_id}"
  local log_root="${PROJECT_ROOT}/logs/v100_trap_plugae_20/${run_id}"
  local log_file="${log_root}/${job_id}.log"
  local master_log="${log_root}/master.log"
  local status_root="${log_root}/status"
  local status_file="${status_root}/${job_id}.json"
  local phase_file="${status_root}/${job_id}.phase"
  local phase="setup" heartbeat_pid="" worker_pid="$$"

  cd "${PROJECT_ROOT}"
  mkdir -p "${job_root}" "${log_root}" "${status_root}"
  export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  export NLTK_DATA="${MODEL_ROOT}/nltk_data"
  export GENSIM_DATA_DIR="${MODEL_ROOT}"
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export TOKENIZERS_PARALLELISM=false
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1

  exec > >(tee -a "${log_file}") 2>&1
  write_status() {
    local state="$1" exit_code="${2:-null}" now temporary
    now="$(date --iso-8601=seconds)"
    temporary="${status_file}.tmp.${BASHPID}"
    printf '{"job_id":"%s","state":"%s","phase":"%s","pid":%s,"host":"%s","updated_at":"%s","exit_code":%s}\n' \
      "${job_id}" "${state}" "$(<"${phase_file}")" "${worker_pid}" \
      "$(hostname)" "${now}" "${exit_code}" > "${temporary}"
    mv "${temporary}" "${status_file}"
  }
  set_phase() {
    phase="$1"
    printf '%s\n' "${phase}" > "${phase_file}"
    write_status running
  }
  finish_worker() {
    local rc="$?"
    trap - EXIT INT TERM HUP
    if [[ -n "${heartbeat_pid}" ]]; then
      kill "${heartbeat_pid}" 2>/dev/null || true
      wait "${heartbeat_pid}" 2>/dev/null || true
    fi
    if (( rc == 0 )); then
      write_status completed 0
      echo "[$(date --iso-8601=seconds)] DONE ${job_id}" >> "${master_log}"
    else
      write_status failed "${rc}"
      echo "[$(date --iso-8601=seconds)] FAIL ${job_id}: phase=${phase} exit=${rc}" >> "${master_log}"
    fi
    exit "${rc}"
  }
  trap finish_worker EXIT
  trap 'exit 130' INT TERM HUP
  printf '%s\n' "${phase}" > "${phase_file}"
  write_status running
  (
    trap - EXIT INT TERM HUP
    while kill -0 "${worker_pid}" 2>/dev/null; do
      write_status running
      sleep 30
    done
  ) &
  heartbeat_pid="$!"

  echo "[$(date --iso-8601=seconds)] START ${job_id}"
  echo "[$(date --iso-8601=seconds)] START ${job_id}" >> "${master_log}"

  set_phase generate
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u evaluation.py generate \
    --benchmark-config "${BENCHMARK_CONFIG}" \
    --fingerprint-config "${fingerprint_config}" \
    --source-model "${source_model}" \
    --model-alias "${model_alias}" \
    --results-root "${job_root}"

  local batch_dir
  batch_dir="$(find "${job_root}" -mindepth 1 -maxdepth 1 -type d \
    -name "exp_*_${method}_seed_042_*" | sort | tail -n 1)"
  if [[ -z "${batch_dir}" ]]; then
    echo "No generated fingerprint batch found in ${job_root}" >&2
    return 1
  fi

  set_phase evaluate
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u evaluation.py run \
    --benchmark-config "${BENCHMARK_CONFIG}" \
    --fingerprint-config "${fingerprint_config}" \
    --evaluation-config "${evaluation_config}" \
    --batch-dir "${batch_dir}" \
    --retry-failed

  phase="finalize"
  printf '%s\n' "${phase}" > "${phase_file}"
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  run_worker "$@"
  exit $?
fi

cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
SESSION="${SESSION:-fp20_paired}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${PROJECT_ROOT}/results/v100_trap_plugae_20/${RUN_ID}"
LOG_ROOT="${PROJECT_ROOT}/logs/v100_trap_plugae_20/${RUN_ID}"
MASTER_LOG="${LOG_ROOT}/master.log"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  echo "Attach with: tmux attach -t ${SESSION}" >&2
  exit 1
fi

mkdir -p "${RESULTS_ROOT}" "${LOG_ROOT}"
touch "${MASTER_LOG}"

"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import yaml

root = Path("config/v100/trap_plugae_20")
models = (
    "llama7b", "llama2", "mistral01", "qwen25",
    "qwen15", "mistral03", "gemma2_2b", "phi3",
)
for model in models:
    trap = yaml.safe_load((root / f"trap_{model}.yaml").read_text())
    plugae = yaml.safe_load((root / f"plugae_{model}.yaml").read_text())
    evaluation = yaml.safe_load((root / f"eval_{model}.yaml").read_text())
    assert trap["n_goals"] == trap["goal_count"] == 20
    assert trap["gcg_config"]["num_steps"] == 1500
    assert trap["gcg_config"]["search_width"] == 512
    assert trap["gcg_config"]["topk"] == 256
    assert plugae["num_queries"] == 20
    assert plugae["epochs"] == 30
    assert plugae["learning_rate"] == 0.1
    assert len(evaluation["evaluations"]["model_specificity"]["seeds"]) == 10
    sampling = evaluation["evaluations"]["deployment_robustness"]["sampling"]
    assert len(sampling["seeds"]) == 10
    assert sampling["temperature_values"] == [0.7]
    assert sampling["temperature_top_p"] == 0.9
print("Configuration check passed: 8 paired source models, 16 jobs.")
PY

preflight_args=(
  "${PYTHON_BIN}" scripts/v100/preflight_models.py
  --benchmark-config "${BENCHMARK_CONFIG}"
)
for evaluation_config in "${CONFIG_ROOT}"/eval_*.yaml; do
  preflight_args+=(--evaluation-config "${evaluation_config}")
done
if [[ "${LEAFBENCH_PREFLIGHT_CONFIG_ONLY:-0}" == "1" ]]; then
  preflight_args+=(--config-only)
fi
if [[ "${LEAFBENCH_PREFLIGHT_FULL_CPU:-0}" == "1" ]]; then
  preflight_args+=(--full-cpu-load)
fi
"${preflight_args[@]}"

tmux new-session -d -s "${SESSION}" -n monitor
tmux send-keys -t "${SESSION}:monitor" "tail -F $(printf '%q' "${MASTER_LOG}")" C-m

launch() {
  local gpu="$1" method="$2" source_model="$3" model_alias="$4"
  local fingerprint_config="$5" evaluation_config="$6"
  local window="g$(printf '%02d' "${gpu}")_${method}"
  local worker_command
  printf -v worker_command '%q ' \
    bash "${SCRIPT_PATH}" --worker \
    "${gpu}" "${method}" "${source_model}" "${model_alias}" \
    "${fingerprint_config}" "${evaluation_config}" "${RUN_ID}" "${PYTHON_BIN}"
  tmux new-window -d -t "${SESSION}:" -n "${window}" "${worker_command}"
}

# GPUs 0-7: TRAP; GPUs 8-15: PlugAE. Each pair uses the same source model and
# the same evaluation YAML.
launch 0 trap Llama-7B llama_7b "${CONFIG_ROOT}/trap_llama7b.yaml" "${CONFIG_ROOT}/eval_llama7b.yaml"
launch 1 trap Llama-2-7B llama2_7b "${CONFIG_ROOT}/trap_llama2.yaml" "${CONFIG_ROOT}/eval_llama2.yaml"
launch 2 trap Mistral-7B-v0.1 mistral_7b_v01 "${CONFIG_ROOT}/trap_mistral01.yaml" "${CONFIG_ROOT}/eval_mistral01.yaml"
launch 3 trap Qwen2.5-7B qwen25_7b "${CONFIG_ROOT}/trap_qwen25.yaml" "${CONFIG_ROOT}/eval_qwen25.yaml"
launch 4 trap Qwen1.5-7B qwen15_7b "${CONFIG_ROOT}/trap_qwen15.yaml" "${CONFIG_ROOT}/eval_qwen15.yaml"
launch 5 trap Mistral-7B-v0.3 mistral_7b_v03 "${CONFIG_ROOT}/trap_mistral03.yaml" "${CONFIG_ROOT}/eval_mistral03.yaml"
launch 6 trap Gemma-2-2B gemma2_2b "${CONFIG_ROOT}/trap_gemma2_2b.yaml" "${CONFIG_ROOT}/eval_gemma2_2b.yaml"
launch 7 trap Phi-3-Mini-4K-Base phi3_mini_4k_base "${CONFIG_ROOT}/trap_phi3.yaml" "${CONFIG_ROOT}/eval_phi3.yaml"

launch 8 plugae Llama-7B llama_7b "${CONFIG_ROOT}/plugae_llama7b.yaml" "${CONFIG_ROOT}/eval_llama7b.yaml"
launch 9 plugae Llama-2-7B llama2_7b "${CONFIG_ROOT}/plugae_llama2.yaml" "${CONFIG_ROOT}/eval_llama2.yaml"
launch 10 plugae Mistral-7B-v0.1 mistral_7b_v01 "${CONFIG_ROOT}/plugae_mistral01.yaml" "${CONFIG_ROOT}/eval_mistral01.yaml"
launch 11 plugae Qwen2.5-7B qwen25_7b "${CONFIG_ROOT}/plugae_qwen25.yaml" "${CONFIG_ROOT}/eval_qwen25.yaml"
launch 12 plugae Qwen1.5-7B qwen15_7b "${CONFIG_ROOT}/plugae_qwen15.yaml" "${CONFIG_ROOT}/eval_qwen15.yaml"
launch 13 plugae Mistral-7B-v0.3 mistral_7b_v03 "${CONFIG_ROOT}/plugae_mistral03.yaml" "${CONFIG_ROOT}/eval_mistral03.yaml"
launch 14 plugae Gemma-2-2B gemma2_2b "${CONFIG_ROOT}/plugae_gemma2_2b.yaml" "${CONFIG_ROOT}/eval_gemma2_2b.yaml"
launch 15 plugae Phi-3-Mini-4K-Base phi3_mini_4k_base "${CONFIG_ROOT}/plugae_phi3.yaml" "${CONFIG_ROOT}/eval_phi3.yaml"

echo "Started 16 paired jobs in tmux session: ${SESSION}"
echo "Run ID:  ${RUN_ID}"
echo "Results: ${RESULTS_ROOT}"
echo "Logs:    ${LOG_ROOT}"
echo "Attach:  tmux attach -t ${SESSION}"
