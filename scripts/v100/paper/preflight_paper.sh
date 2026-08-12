#!/usr/bin/env bash

set -Eeuo pipefail
PROJECT_ROOT="${LEAFBENCH_ROOT:-/raid/chj/fingerprint/LeaFBench}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ROOT="/raid/chj/fingerprint/models"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NLTK_DATA="${MODEL_ROOT}/nltk_data"
export GENSIM_DATA_DIR="${MODEL_ROOT}"

required=(
  Qwen2.5-7B Qwen2.5-7B-Instruct Qwen2.5-14B Qwen2.5-14B-Instruct
  Qwen1.5-7B Qwen1.5-7B-Chat Qwen3-14B-Instruct
  Llama-2-7b-hf Llama-2-7b-chat-hf CodeLlama-7b-hf
  Llama-3.1-8B Llama-3.1-8B-Instruct
  Mistral-7B-v0.3 Mistral-7B-Instruct-v0.3
  gemma-2-2b gemma-2-9b-it Phi-3-mini-4k-base Phi-3-mini-4k-instruct
  opt-1.3b all-mpnet-base-v2 nltk_data
)
missing=0
for name in "${required[@]}"; do
  [[ -e "${MODEL_ROOT}/${name}" ]] || { echo "Missing ${MODEL_ROOT}/${name}" >&2; missing=1; }
done
[[ -e "${MODEL_ROOT}/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz" ]] || {
  echo "Missing ${MODEL_ROOT}/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz" >&2
  missing=1
}
(( missing == 0 )) || exit 1

gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
[[ "${gpu_count}" -eq 16 ]] || { echo "Expected exactly 16 GPUs, found ${gpu_count}" >&2; exit 1; }

"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import yaml
import evaluation

root = Path("config/v100/paper")
paths = sorted(root.glob("*.yaml"))
assert paths, "No paper YAML files found"
for path in paths:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    assert isinstance(value, dict), path

benchmark = evaluation.load_yaml(root / "benchmark.yaml")
available = set()
for family in benchmark["models"]:
    for key in ("pretrained_model", "instruct_model"):
        if family.get(key):
            available.add(family[key]["model_name"])
for path in root.glob("evaluation_*.yaml"):
    config = evaluation.load_yaml(path)
    groups = evaluation.model_groups(config)
    named = {x["model_name"] for rows in groups.values() for x in rows}
    deployment = config["evaluations"].get("deployment_robustness", {})
    if deployment.get("model_name"):
        named.add(deployment["model_name"])
    assert named <= available, f"{path}: missing {sorted(named - available)}"
print(f"Parsed {len(paths)} paper configs; registered {len(available)} model aliases")
PY

"${PYTHON_BIN}" -m unittest tests.test_evaluation
echo "Paper experiment preflight passed."
