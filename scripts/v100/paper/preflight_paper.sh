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
  llama-7b-hf Llama-2-7b-hf Llama-2-7b-chat-hf CodeLlama-7b-hf
  guanaco-7B-HF vicuna-7b-v1.3 Mistral-7B-v0.1 Mistral-7B-Instruct-v0.1
  Llama-3.1-8B Llama-3.1-8B-Instruct
  Mistral-7B-v0.3 Mistral-7B-Instruct-v0.3
  gemma-2-2b gemma-2-9b-it Phi-3-mini-4k-base Phi-3-mini-4k-instruct
  opt-1.3b all-mpnet-base-v2 nltk_data
)
if [[ "${LEAFBENCH_PREFLIGHT_CONFIG_ONLY:-0}" != "1" ]]; then
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
fi

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
    for model in family.get("predeployed_models", []):
        available.add(model["model_name"])
for path in root.glob("evaluation_*.yaml"):
    config = evaluation.load_yaml(path)
    groups = evaluation.model_groups(config)
    named = {x["model_name"] for rows in groups.values() for x in rows}
    deployment = config["evaluations"].get("deployment_robustness", {})
    if deployment.get("model_name"):
        named.add(deployment["model_name"])
    assert named <= available, f"{path}: missing {sorted(named - available)}"

trap_configs = [evaluation.load_yaml(path) for path in root.glob("trap*.yaml")]
assert trap_configs, "Missing TRAP paper configs"
for config in trap_configs:
    gcg = config["gcg_config"]
    assert config["n_goals"] == 100
    assert config["seed"] == 42
    assert config["string_type"] == "number" and config["string_length"] == 4
    assert config["test_n_times"] == 10
    assert gcg["num_steps"] == 1500 and gcg["search_width"] == 512
    assert gcg["batch_size"] == 512 and gcg["topk"] == 256
    assert gcg["seed"] == 42
    assert gcg["early_stop"] is False
    assert gcg["filter_words_path"] == "data/filter_words_number.csv"
assert sorted(set((x["goal_offset"], x["goal_count"]) for x in trap_configs)) == [(0, 34), (34, 33), (67, 33)]
assert all(x["prompt_seed"] == 41 for x in trap_configs)

zeroprint_configs = [evaluation.load_yaml(path) for path in root.glob("zeroprint*.yaml")]
assert zeroprint_configs, "Missing ZeroPrint paper configs"
for config in zeroprint_configs:
    assert config["n_samples"] == 2
    assert config["resample"] is False
    assert config["query_csv_path"] == "data/zeroprint_humaneval_seed_1000.csv"
    assert config["query_datasets"] == ["openai_humaneval"]
    assert config["word_substitution_config"]["n_augmented_samples"] == 4
    assert config["word_substitution_config"]["k_words_to_replace"] == 3
    assert config["word_substitution_config"]["top_m_neighbors"] == 10
    assert config["num_repeats"] == 20 and config["ridge_alpha"] == 0.001
    assert config["output_truncate_length"] == 0
    assert config["model_generation"] == {
        "do_sample": True, "temperature": 0.7, "top_p": 0.9,
        "top_k": 50, "max_new_tokens": 512,
    }

import csv
with Path("data/zeroprint_humaneval_seed_1000.csv").open(newline="", encoding="utf-8") as stream:
    humaneval = list(csv.DictReader(stream))
assert [row["source_task"] for row in humaneval] == ["HumanEval/72", "HumanEval/163"]
assert all(row["query"].startswith("Complete the following code: ") for row in humaneval)
for path in root.glob("evaluation_zeroprint_*.yaml"):
    config = evaluation.load_yaml(path)
    assert config["evaluations"]["model_specificity"]["seeds"] == [1000]
    assert config["evaluations"]["deployment_robustness"]["enabled"] is False

reef = evaluation.load_yaml(root / "reef_seed_42.yaml")
assert reef["num_samples"] == 200 and reef["layers"] == 18
assert evaluation.load_yaml(root / "reef_llama2.yaml")["num_samples"] == 200

for path in root.glob("plugae*.yaml"):
    config = evaluation.load_yaml(path)
    assert config["copyright_token"] == "mkahg"
    assert config["num_queries"] == 50
    assert config["learning_rate"] == 0.1 and config["epochs"] == 30
print(f"Parsed {len(paths)} paper configs; registered {len(available)} model aliases")
PY

"${PYTHON_BIN}" -m unittest \
  tests.test_evaluation tests.test_trap_protocol tests.test_reef_protocol
echo "Paper experiment preflight passed."
