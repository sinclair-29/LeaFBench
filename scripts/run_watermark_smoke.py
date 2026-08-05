#!/usr/bin/env python3
"""Run five matched unwatermarked/watermarked generations from an offline corpus."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

# When this file is executed directly, Python adds ``scripts/`` rather than the
# repository root to sys.path.  Add the root so LeaFBench's top-level packages
# are importable without requiring callers to set PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml
from transformers import set_seed

from benchmark.benchmark import Benchmark
from deploying_techniques.watermark.corpus import load_watermark_corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Optional local-path override for both model entries in the smoke YAML.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSONL result path.")
    return parser.parse_args()


def load_config(path: Path, model_path: Path | None) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if model_path is not None:
        resolved = str(model_path.expanduser().resolve())
        for model in config["models"]:
            for key in ("pretrained_model", "instruct_model"):
                if model.get(key) is not None:
                    model[key]["model_path"] = resolved
    for model in config["models"]:
        for key in ("pretrained_model", "instruct_model"):
            if model.get(key) is None:
                continue
            local_path = Path(model[key]["model_path"]).expanduser()
            if not local_path.exists():
                raise FileNotFoundError(
                    f"Local model path does not exist: {local_path}. "
                    "Edit the YAML or pass --model-path."
                )
    return config


def find_watermarked_model(benchmark: Benchmark):
    matches = [model for model in benchmark.models.values() if model.type == "watermark"]
    if len(matches) != 1:
        raise ValueError(f"Smoke config must create exactly one watermarked model; found {len(matches)}")
    return matches[0]


def validate_detection(result: dict[str, Any]) -> None:
    for key in ("z_score", "p_value", "green_fraction"):
        if key not in result or not math.isfinite(float(result[key])):
            raise RuntimeError(f"Detector returned invalid {key}: {result.get(key)!r}")
    if int(result.get("num_tokens_scored", 0)) <= 0:
        raise RuntimeError("Detector scored no generated tokens")


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.model_path)
    watermark_configs = config.get("deploying_techniques", {}).get("watermark", [])
    if len(watermark_configs) != 1:
        raise ValueError("Smoke config must contain one deploying_techniques.watermark entry")
    method = watermark_configs[0]["method"]
    prompt_path = Path(config["smoke"]["prompt_path"])
    if not prompt_path.is_absolute():
        prompt_path = (Path.cwd() / prompt_path).resolve()
    records = load_watermark_corpus(prompt_path, expected_method=method)
    prompts = [record["prompt"] for record in records]
    seed = int(config["smoke"].get("seed", 1234))

    benchmark = Benchmark(config, accelerator=None, fingerprint_type="black-box")
    watermarked_model = find_watermarked_model(benchmark)

    # Reset to the same RNG state so the only intended difference is the
    # watermark logits transformation.
    set_seed(seed)
    baseline_outputs = watermarked_model.generate_unwatermarked(prompts)
    set_seed(seed)
    watermarked_outputs = watermarked_model.generate(prompts)
    baseline_detections = watermarked_model.detect(baseline_outputs)
    watermarked_detections = watermarked_model.detect(watermarked_outputs)

    output_stream = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_stream = args.output.open("w", encoding="utf-8")
    try:
        for record, baseline, watermarked, baseline_detection, watermarked_detection in zip(
            records,
            baseline_outputs,
            watermarked_outputs,
            baseline_detections,
            watermarked_detections,
        ):
            if not baseline.strip() or not watermarked.strip():
                raise RuntimeError(f"Empty generation for {record['id']}")
            validate_detection(baseline_detection)
            validate_detection(watermarked_detection)
            result = {
                "id": record["id"],
                "source_index": record["source_index"],
                "method": method,
                "prompt": record["prompt"],
                "natural_text": record["natural_text"],
                "unwatermarked_text": baseline,
                "watermarked_text": watermarked,
                "unwatermarked_detection": baseline_detection,
                "watermarked_detection": watermarked_detection,
                # Backward-compatible alias for consumers of the first smoke format.
                "detection": watermarked_detection,
            }
            print(
                f"{record['id']}: "
                f"unwm_tokens={baseline_detection['num_tokens_scored']} "
                f"unwm_z={baseline_detection['z_score']:.4f} "
                f"unwm_detected={baseline_detection['is_watermarked']} | "
                f"wm_tokens={watermarked_detection['num_tokens_scored']} "
                f"wm_z={watermarked_detection['z_score']:.4f} "
                f"wm_detected={watermarked_detection['is_watermarked']}"
            )
            if output_stream is not None:
                output_stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    finally:
        if output_stream is not None:
            output_stream.close()

    baseline_detected = sum(
        bool(result["is_watermarked"]) for result in baseline_detections
    )
    watermarked_detected = sum(
        bool(result["is_watermarked"]) for result in watermarked_detections
    )
    print(
        f"summary: method={method} samples={len(records)} "
        f"unwatermarked_detected={baseline_detected}/{len(records)} "
        f"watermarked_detected={watermarked_detected}/{len(records)}"
    )


if __name__ == "__main__":
    main()
