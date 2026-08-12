#!/usr/bin/env python3
"""Download the checkpoints and calibration data required by Exp-2 to Exp-5."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


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

# Verified against the ModelScope model API. A value of None means that the
# exact experiment checkpoint is not mirrored there; do not silently replace
# it with a similarly named model because that changes the experiment.
MODELSCOPE_MODELS = {
    "HumanLLMs/Human-Like-Qwen2.5-7B-Instruct": "HumanLLMs/Human-Like-Qwen2.5-7B-Instruct",
    "lightblue/qwen2.5-7b-instruct-simpo": None,
    "prithivMLmods/Math-IIO-7B-Instruct": "prithivMLmods/Math-IIO-7B-Instruct",
    "nguyentd/FinancialAdvice-Qwen2.5-7B": None,
    "Qwen/Qwen2.5-Coder-7B-Instruct": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Qwen/Qwen2-7B-Instruct": "Qwen/Qwen2-7B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct",
    "01-ai/Yi-6B-Chat": "01ai/Yi-6B-Chat",
    "Joyqiuyue/Llama-2-7b-chat-hf-dpo": None,
    "EdwardYu/llama-2-7b-MedQuAD": None,
    "aengusl/llama2-7b-sft-lora": None,
    "surrey-nlp/AG-Llama-2-7b": None,
    "meta-llama/Llama-2-13b-chat-hf": "modelscope/Llama-2-13b-chat-ms",
    "openai-community/gpt2": "AI-ModelScope/gpt2",
    "google/gemma-7b-it": "LLM-Research/gemma-7b-it",
    "meta-llama/Meta-Llama-3-8B-Instruct": "LLM-Research/Meta-Llama-3-8B-Instruct",
}

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
    parser.add_argument(
        "--backend",
        choices=("modelscope", "huggingface"),
        default="modelscope",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="With ModelScope, download available mirrors and report exact missing repos.",
    )
    return parser.parse_args()


def weight_patterns(repo_id: str, token: str | None) -> list[str]:
    from huggingface_hub import HfApi

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


def download_huggingface_model(
    repo_id: str, destination: Path, token: str | None
) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        local_dir=destination,
        token=token,
        allow_patterns=weight_patterns(repo_id, token),
        ignore_patterns=IGNORE_PATTERNS,
    )


def download_modelscope(kind: str, repo_id: str, destination: Path) -> None:
    command = [
        "modelscope",
        "download",
        f"--{kind}",
        repo_id,
        "--local_dir",
        str(destination),
    ]
    subprocess.run(command, check=True)


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

    missing_modelscope = [
        repo_id
        for repo_id, _ in models
        if args.backend == "modelscope" and MODELSCOPE_MODELS.get(repo_id) is None
    ]
    if missing_modelscope and not args.allow_partial:
        missing = "\n".join(f"  - {repo_id}" for repo_id in missing_modelscope)
        raise RuntimeError(
            "The exact checkpoints below are not available on ModelScope:\n"
            f"{missing}\n"
            "Re-run with --allow-partial to download all verified mirrors, then "
            "transfer the missing checkpoints separately."
        )

    for index, (repo_id, directory_name) in enumerate(models, start=1):
        destination = args.model_root / directory_name
        if args.backend == "modelscope":
            source_id = MODELSCOPE_MODELS.get(repo_id)
            if source_id is None:
                print(f"[{index}/{len(models)}] MISSING ON MODELSCOPE: {repo_id}")
                continue
            print(
                f"[{index}/{len(models)}] ModelScope {source_id} -> {destination}",
                flush=True,
            )
            download_modelscope("model", source_id, destination)
        else:
            print(
                f"[{index}/{len(models)}] Hugging Face {repo_id} -> {destination}",
                flush=True,
            )
            download_huggingface_model(repo_id, destination, token)

    dataset_destination = args.dataset_root / "cais_mmlu"
    print(f"Downloading cais/mmlu -> {dataset_destination}", flush=True)
    if args.backend == "modelscope":
        download_modelscope("dataset", "cais/mmlu", dataset_destination)
    else:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id="cais/mmlu",
            repo_type="dataset",
            local_dir=dataset_destination,
            token=token,
        )

    if missing_modelscope:
        print("\nDownloaded every exact resource available from ModelScope.")
        print("Still missing:")
        for repo_id in missing_modelscope:
            print(f"  - {repo_id}")
        return 0
    print("Evaluation resources downloaded successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
