#!/usr/bin/env python3
"""Offline model/tokenizer/checkpoint preflight for V100 experiment launchers."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def benchmark_models(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for family in config.get("models", []):
        for key in ("pretrained_model", "instruct_model"):
            entry = family.get(key)
            if entry:
                normalized = dict(entry)
                normalized.setdefault(
                    "type", "instruct" if key == "instruct_model" else "pretrained"
                )
                result[entry["model_name"]] = normalized
        for entry in family.get("predeployed_models", []):
            result[entry["model_name"]] = dict(entry)
    return result


def referenced_models(paths: list[Path]) -> set[str]:
    names: set[str] = set()
    for path in paths:
        config = load_yaml(path)
        source = config.get("source_model") or {}
        if source.get("model_name"):
            names.add(source["model_name"])
        for entries in (config.get("model_groups") or {}).values():
            for entry in entries or []:
                names.add(entry if isinstance(entry, str) else entry["model_name"])
        for spec in (config.get("evaluations") or {}).values():
            if isinstance(spec, dict) and spec.get("model_name"):
                names.add(spec["model_name"])
    return names


def weight_files(model_path: Path) -> list[Path]:
    indexes = sorted(model_path.glob("*.index.json"))
    if indexes:
        files: set[Path] = set()
        for index_path in indexes:
            value = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = value.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"Empty or invalid weight_map: {index_path}")
            files.update(model_path / name for name in weight_map.values())
        missing = sorted(path for path in files if not path.is_file())
        if missing:
            raise FileNotFoundError(
                "Missing indexed weight shard(s): "
                + ", ".join(str(path) for path in missing)
            )
        return sorted(files)
    files = sorted(model_path.glob("*.safetensors"))
    files.extend(sorted(model_path.glob("pytorch_model*.bin")))
    if not files:
        raise FileNotFoundError(f"No model weights or weight index in {model_path}")
    return files


def inspect_weight_file(path: Path) -> None:
    if path.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as stream:
            keys = list(stream.keys())
            if not keys:
                raise ValueError(f"Safetensors shard has no tensors: {path}")
            stream.get_slice(keys[0]).get_shape()
    else:
        import torch

        try:
            value = torch.load(path, map_location="meta", weights_only=True)
        except TypeError:  # PyTorch releases before weights_only was added
            value = torch.load(path, map_location="meta")
        if not isinstance(value, dict) or not value:
            raise ValueError(f"PyTorch weight shard is empty or invalid: {path}")


def check_model(name: str, entry: dict[str, Any], full_cpu_load: bool) -> None:
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    path = Path(entry["model_path"])
    if not path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {path}")
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    probe = tokenizer("preflight", add_special_tokens=True)
    if not probe.get("input_ids"):
        raise ValueError(f"Tokenizer produced no input ids: {name}")
    if entry.get("type") in {"instruct", "instruction_tuning"}:
        template = getattr(tokenizer, "chat_template", None)
        if template:
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "preflight"}],
                tokenize=False,
                add_generation_prompt=True,
            )
    for shard in weight_files(path):
        inspect_weight_file(shard)
    with init_empty_weights():
        AutoModelForCausalLM.from_config(config)
    if full_cpu_load:
        import torch

        loaded_model = AutoModelForCausalLM.from_pretrained(
            path,
            local_files_only=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map={"": "cpu"},
        )
        if loaded_model.config.model_type != config.model_type:
            raise ValueError(
                f"Loaded model type changed: {loaded_model.config.model_type} "
                f"!= {config.model_type}"
            )
        del loaded_model
        gc.collect()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-config", type=Path, required=True)
    parser.add_argument(
        "--evaluation-config", type=Path, action="append", default=[]
    )
    parser.add_argument("--config-only", action="store_true")
    parser.add_argument("--full-cpu-load", action="store_true")
    args = parser.parse_args()

    benchmark = load_yaml(args.benchmark_config)
    registry = benchmark_models(benchmark)
    names = referenced_models(args.evaluation_config)
    if not names:
        names = set(registry)
    missing = sorted(names - set(registry))
    errors: list[str] = []
    if missing:
        errors.append("Models absent from benchmark registry: " + ", ".join(missing))
    if not args.config_only:
        for name in sorted(names & set(registry)):
            try:
                check_model(name, registry[name], args.full_cpu_load)
                print(f"PASS {name}")
            except Exception as error:  # collect every broken resource in one pass
                errors.append(f"{name}: {type(error).__name__}: {error}")
                print(f"FAIL {name}: {type(error).__name__}: {error}")
    if errors:
        print("Preflight failed:\n- " + "\n- ".join(errors))
        return 1
    print(f"Preflight passed for {len(names)} referenced model(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
