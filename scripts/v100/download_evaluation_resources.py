#!/usr/bin/env python3
"""Download the checkpoints and calibration data required by Exp-2 to Exp-5."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


CURRENT_CONFIG_MODELS = (
    ("HumanLLMs/Human-Like-Qwen2.5-7B-Instruct", "Human-Like-Qwen2.5-7B-Instruct"),
    ("lightblue/qwen2.5-7b-instruct-simpo", "qwen2.5-7b-instruct-simpo"),
    ("prithivMLmods/Math-IIO-7B-Instruct", "Math-IIO-7B-Instruct"),
    ("nguyentd/FinancialAdvice-Qwen2.5-7B", "FinancialAdvice-Qwen2.5-7B"),
    ("Qwen/Qwen2.5-Coder-7B-Instruct", "Qwen2.5-Coder-7B-Instruct"),
    ("Qwen/Qwen2-7B-Instruct", "Qwen2-7B-Instruct"),
    ("Qwen/Qwen2.5-3B-Instruct", "Qwen2.5-3B-Instruct"),
    ("01-ai/Yi-6B-Chat", "Yi-6B-Chat"),
    ("Joyqiuyue/Llama-2-7b-chat-hf-dpo", "Llama-2-7b-chat-hf-dpo"),
    ("EdwardYu/llama-2-7b-MedQuAD", "llama-2-7b-MedQuAD"),
    ("aengusl/llama2-7b-sft-lora", "llama2-7b-sft-lora"),
    ("surrey-nlp/AG-Llama-2-7b", "AG-Llama-2-7b"),
    ("meta-llama/Llama-2-13b-chat-hf", "Llama-2-13b-chat-hf"),
    ("openai-community/gpt2", "gpt2"),
)

# Exp-4 names these exact checkpoints. The current repository configurations
# substitute already-local Gemma-2-2B-IT and Llama-3.1-8B-Instruct instead.
STRICT_EXP4_MODELS = (
    ("google/gemma-7b-it", "gemma-7b-it"),
    ("meta-llama/Meta-Llama-3-8B-Instruct", "Meta-Llama-3-8B-Instruct"),
)

IGNORE_PATTERNS = (
    "*.gguf",
    "*.onnx",
    "*.tflite",
    "*.h5",
    "*.msgpack",
    "*.ot",
    "*.pth",
    "original/*",
    "onnx/*",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("/raid/chj/fingerprint/models"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/raid/chj/fingerprint/datasets"),
    )
    parser.add_argument(
        "--strict-exp4",
        action="store_true",
        help="Also download exact Gemma-7B and Llama-3 checkpoints named in exp4.md.",
    )
    return parser.parse_args()


def weight_patterns(repo_id: str, token: str | None) -> list[str]:
    files = HfApi(token=token).list_repo_files(repo_id=repo_id, token=token)
    patterns = [
        "*.json",
        "*.safetensors",
        "*.model",
        "*.tiktoken",
        "*.txt",
        "*.py",
        "*.jinja",
    ]
    if not any(name.endswith(".safetensors") for name in files):
        patterns.append("*.bin")
    return patterns


def main() -> int:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    models = list(CURRENT_CONFIG_MODELS)
    if args.strict_exp4:
        models.extend(STRICT_EXP4_MODELS)

    args.model_root.mkdir(parents=True, exist_ok=True)
    args.dataset_root.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(args.model_root).free / 1024**3
    recommended_gib = 200 if args.strict_exp4 else 165
    if free_gib < recommended_gib:
        raise RuntimeError(
            f"Only {free_gib:.1f} GiB free under {args.model_root}; "
            f"at least {recommended_gib} GiB is recommended."
        )

    for index, (repo_id, directory_name) in enumerate(models, start=1):
        destination = args.model_root / directory_name
        print(f"[{index}/{len(models)}] {repo_id} -> {destination}", flush=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=destination,
            token=token,
            allow_patterns=weight_patterns(repo_id, token),
            ignore_patterns=IGNORE_PATTERNS,
        )

    dataset_destination = args.dataset_root / "cais_mmlu"
    print(f"Downloading cais/mmlu -> {dataset_destination}", flush=True)
    snapshot_download(
        repo_id="cais/mmlu",
        repo_type="dataset",
        local_dir=dataset_destination,
        token=token,
    )
    print("Evaluation resources downloaded successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
