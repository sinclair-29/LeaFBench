#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${LEAFBENCH_ROOT:-/raid/chj/fingerprint/LeaFBench}"
CONFIG_ROOT="config/v100/trap_plugae_smoke"
BENCHMARK_CONFIG="${CONFIG_ROOT}/benchmark.yaml"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
GPU_ID="${GPU_ID:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results/v100_trap_plugae_smoke/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/v100_trap_plugae_smoke/${RUN_ID}}"
LOG_FILE="${LOG_ROOT}/run.log"
STATUS_FILE="${LOG_ROOT}/status.json"
REPORT_FILE="${RESULTS_ROOT}/smoke_validation.json"
phase="setup"

cd "${PROJECT_ROOT}"
mkdir -p "${RESULTS_ROOT}" "${LOG_ROOT}"
exec > >(tee -a "${LOG_FILE}") 2>&1

write_status() {
  local state="$1" exit_code="${2:-null}" temporary
  temporary="${STATUS_FILE}.tmp.${BASHPID}"
  printf '{"state":"%s","phase":"%s","updated_at":"%s","exit_code":%s,"results_root":"%s"}\n' \
    "${state}" "${phase}" "$(date --iso-8601=seconds)" "${exit_code}" \
    "${RESULTS_ROOT}" > "${temporary}"
  mv "${temporary}" "${STATUS_FILE}"
}

finish() {
  local rc="$?"
  trap - EXIT INT TERM HUP
  if (( rc == 0 )); then
    write_status completed 0
    echo "PASS: TRAP/PlugAE correctness gate"
  else
    write_status failed "${rc}"
    echo "FAIL: phase=${phase} exit=${rc}"
  fi
  echo "Results: ${RESULTS_ROOT}"
  echo "Validation report: ${REPORT_FILE}"
  exit "${rc}"
}
trap finish EXIT
trap 'exit 130' INT TERM HUP
write_status running

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

phase="runtime_check"
write_status running
"${PYTHON_BIN}" - <<'PY'
from packaging.version import Version
import torch

version = torch.__version__.split("+")[0]
if Version(version) < Version("2.6.0"):
    raise SystemExit(f"PyTorch >= 2.6.0 is required, found {torch.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
print(f"Runtime PASS: torch={torch.__version__}, CUDA={torch.version.cuda}, GPU={torch.cuda.get_device_name(0)}")
PY

phase="protocol_tests"
write_status running
"${PYTHON_BIN}" -m unittest \
  tests.test_trap_plugae_smoke_validation \
  tests.test_plugae_protocol \
  tests.test_trap_protocol \
  tests.test_trap_checkpointing \
  tests.test_evaluation \
  tests.test_preflight_models

phase="preflight"
write_status running
preflight_args=(
  "${PYTHON_BIN}" scripts/v100/preflight_models.py
  --benchmark-config "${BENCHMARK_CONFIG}"
  --evaluation-config "${CONFIG_ROOT}/eval_phi3.yaml"
)
if [[ "${LEAFBENCH_PREFLIGHT_FULL_CPU:-0}" == "1" ]]; then
  preflight_args+=(--full-cpu-load)
fi
"${preflight_args[@]}"

run_method() {
  local method="$1" config="${CONFIG_ROOT}/$2" method_root="${RESULTS_ROOT}/${method}"
  local batch_dir
  mkdir -p "${method_root}"

  phase="${method}_generate"
  write_status running
  if ! "${PYTHON_BIN}" -u evaluation.py generate \
    --benchmark-config "${BENCHMARK_CONFIG}" \
    --fingerprint-config "${config}" \
    --source-model Phi-3-Mini-4K-Base \
    --model-alias phi3_smoke \
    --results-root "${method_root}"; then
    echo "${method}: generation failed"
    return 1
  fi

  batch_dir="$(find "${method_root}" -mindepth 1 -maxdepth 1 -type d -name 'exp_*' | sort | tail -n 1)"
  if [[ -z "${batch_dir}" ]]; then
    echo "${method}: generated batch directory not found" >&2
    return 1
  fi

  phase="${method}_generation_gate"
  write_status running
  if ! "${PYTHON_BIN}" scripts/v100/trap_plugae_smoke/validate_results.py \
    --batch "${batch_dir}" \
    --method "${method}" \
    --config-root "${CONFIG_ROOT}" \
    --report "${RESULTS_ROOT}/${method}_generation_validation.json"; then
    echo "${method}: source-fingerprint gate failed; formal evaluation was not started"
    return 1
  fi

  phase="${method}_evaluate"
  write_status running
  if ! "${PYTHON_BIN}" -u evaluation.py run \
    --benchmark-config "${BENCHMARK_CONFIG}" \
    --fingerprint-config "${config}" \
    --evaluation-config "${CONFIG_ROOT}/eval_phi3.yaml" \
    --batch-dir "${batch_dir}" \
    --retry-failed; then
    echo "${method}: evaluation reported failure; final validator will classify it"
    return 1
  fi
}

experiment_rc=0
run_method trap trap_phi3.yaml || experiment_rc=1
run_method plugae plugae_phi3.yaml || experiment_rc=1

phase="validate"
write_status running
validation_rc=0
"${PYTHON_BIN}" scripts/v100/trap_plugae_smoke/validate_results.py \
  --results-root "${RESULTS_ROOT}" \
  --config-root "${CONFIG_ROOT}" \
  --report "${REPORT_FILE}" || validation_rc=$?

if (( experiment_rc != 0 || validation_rc != 0 )); then
  exit 1
fi
