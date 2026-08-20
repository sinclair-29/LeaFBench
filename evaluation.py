"""Artifact-first evaluation for LeaFBench fingerprinting methods.

The command deliberately separates fingerprint generation from evaluation:

    python evaluation.py generate ...
    python evaluation.py run ...

The first command writes one immutable, numbered fingerprint batch.  The
second command only loads that batch and writes the four semantic evaluation
files selected by one source-model evaluation configuration.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np
import yaml


SCHEMA_VERSION = 2
EVALUATION_NAMES = (
    "model_modification_robustness",
    "deployment_robustness",
    "model_specificity",
    "prompt_stealthiness",
)
ARTIFACT_PATTERN = re.compile(r"^\d{3,}\.json$")
EXPERIMENT_PATTERN = re.compile(
    r"^(?P<base>exp_[a-z0-9_]+_seed_\d+)_(?P<variant>[a-z]+)$"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: Union[Path, str]) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return data


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path: Union[Path, str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    if not normalized:
        raise ValueError(f"Cannot derive a filesystem alias from {value!r}")
    return normalized


def atomic_write_json(path: Union[Path, str], value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_write_text(path: Union[Path, str], value: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path: Union[Path, str]) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def seed_fingerprint_generation(seed: int) -> None:
    """Seed every RNG used by fingerprint preparation and optimization."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        from transformers import set_seed
    except ModuleNotFoundError:
        # Configuration/unit-test environments may intentionally omit the GPU
        # runtime. Real fingerprint generation imports both dependencies later
        # and therefore still fails fast if the experiment environment is bad.
        return
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def variant_to_number(value: str) -> int:
    number = 0
    for character in value:
        if character < "a" or character > "z":
            raise ValueError(f"Invalid experiment variant: {value}")
        number = number * 26 + ord(character) - ord("a") + 1
    return number


def number_to_variant(number: int) -> str:
    if number < 1:
        raise ValueError("Experiment variants start at one.")
    characters = []
    while number:
        number, remainder = divmod(number - 1, 26)
        characters.append(chr(ord("a") + remainder))
    return "".join(reversed(characters))


def artifact_paths(batch_dir: Union[Path, str]) -> List[Path]:
    batch_dir = Path(batch_dir)
    if not batch_dir.is_dir():
        return []
    paths = [path for path in batch_dir.iterdir() if ARTIFACT_PATTERN.match(path.name)]
    return sorted(paths, key=lambda path: int(path.stem))


def load_records(
    batch_dir: Union[Path, str], *, allow_incomplete: bool = False
) -> List[Dict[str, Any]]:
    batch_dir = Path(batch_dir)
    config_path = batch_dir / "fingerprint_config.json"
    if config_path.is_file() and not allow_incomplete:
        config = read_json(config_path)
        if int(config.get("schema_version", 1)) >= 2:
            status = config.get("status")
            expected_count = config.get("expected_artifact_count")
            if status not in {"completed", "completed_with_warnings"}:
                raise ValueError(
                    f"Fingerprint batch is incomplete ({status!r}): {batch_dir}"
                )
            if expected_count is not None and config.get("artifact_count") != expected_count:
                raise ValueError(
                    "Completed fingerprint manifest has inconsistent artifact counts: "
                    f"{batch_dir}"
                )
    paths = artifact_paths(batch_dir)
    if not paths:
        raise FileNotFoundError(f"No numbered fingerprint artifacts found in {batch_dir}")
    expected = list(range(1, len(paths) + 1))
    actual = [int(path.stem) for path in paths]
    if actual != expected:
        raise ValueError(
            f"Fingerprint artifact numbering must be contiguous from 001; found {actual}"
        )
    return [read_json(path) for path in paths]


def experiment_base_name(model_alias: str, method: str, seed: int) -> str:
    return f"exp_{slug(model_alias)}_{slug(method)}_seed_{seed:03d}"


def experiment_variant(path: Path) -> tuple[str, str]:
    match = EXPERIMENT_PATTERN.match(path.name)
    if not match:
        raise ValueError(
            "Experiment folders must use "
            "exp_<model>_<method>_seed_<seed>_<variant>: "
            f"{path.name}"
        )
    return match.group("base"), match.group("variant")


def next_variant_directory(results_root: Path, base_name: str) -> Path:
    variants = []
    for candidate in results_root.glob(f"{base_name}_*"):
        if not candidate.is_dir():
            continue
        try:
            candidate_base, variant = experiment_variant(candidate)
        except ValueError:
            continue
        if candidate_base == base_name:
            variants.append(variant_to_number(variant))
    return results_root / f"{base_name}_{number_to_variant(max(variants, default=0) + 1)}"


def generation_identity(
    fingerprint_config: Mapping[str, Any],
    source_model: Any,
    benchmark_config_path: Path,
) -> Dict[str, Any]:
    method = fingerprint_config.get("fingerprint_method")
    if not method:
        raise ValueError("fingerprint_method is required.")
    if "seed" not in fingerprint_config:
        raise ValueError("Fingerprint generation config requires an explicit seed.")
    return {
        "fingerprint_method": method,
        "optimization_seed": int(fingerprint_config["seed"]),
        "source_model": {
            "model_name": source_model.model_name,
            "model_path": source_model.model_path,
            "model_family": source_model.model_family,
            "model_type": source_model.type,
        },
        "fingerprint_parameters": {
            key: value
            for key, value in fingerprint_config.items()
            if key not in {"cached_fingerprints_path", "re_fingerprinting"}
        },
        "benchmark_config": {
            "path": str(benchmark_config_path),
            "sha256": file_hash(benchmark_config_path),
        },
    }


def resolve_generation_directory(
    results_root: Path,
    base_name: str,
    identity_hash: str,
) -> Path:
    matches = []
    for candidate in results_root.glob(f"{base_name}_*"):
        config_path = candidate / "fingerprint_config.json"
        if not config_path.is_file():
            continue
        try:
            stored = read_json(config_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if stored.get("config_hash") == identity_hash:
            _, variant = experiment_variant(candidate)
            matches.append((variant_to_number(variant), candidate))
    if matches:
        return max(matches)[1]
    return next_variant_directory(results_root, base_name)


def write_fingerprint_batch(
    batch_dir: Path,
    records: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any],
) -> None:
    if not records:
        raise ValueError("A generated fingerprint batch cannot be empty.")
    batch_dir.mkdir(parents=True, exist_ok=True)
    existing = artifact_paths(batch_dir)
    if existing:
        if len(existing) == len(records):
            return
        raise ValueError(
            f"Refusing to mix {len(records)} artifacts with an existing "
            f"{len(existing)}-artifact batch: {batch_dir}"
        )
    config = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": batch_dir.name,
        "config_hash": stable_hash(identity),
        "created_at": utc_now(),
        "status": "completed",
        "expected_artifact_count": len(records),
        "artifact_count": len(records),
        "warnings": [],
        "updated_at": utc_now(),
        **copy.deepcopy(identity),
    }
    atomic_write_json(batch_dir / "fingerprint_config.json", config)
    width = max(3, len(str(len(records))))
    for index, record in enumerate(records, start=1):
        atomic_write_json(batch_dir / f"{index:0{width}d}.json", record)


def initialize_fingerprint_manifest(
    batch_dir: Path,
    identity: Mapping[str, Any],
    expected_artifact_count: Optional[int],
) -> Dict[str, Any]:
    """Create or validate a resumable schema-v2 generation manifest."""
    batch_dir.mkdir(parents=True, exist_ok=True)
    path = batch_dir / "fingerprint_config.json"
    identity_hash = stable_hash(identity)
    if path.is_file():
        manifest = read_json(path)
        if manifest.get("config_hash") != identity_hash:
            raise ValueError(f"Fingerprint generation identity changed: {batch_dir}")
        if int(manifest.get("schema_version", 1)) < 2:
            return manifest
        stored_expected = manifest.get("expected_artifact_count")
        if (
            stored_expected is not None
            and expected_artifact_count is not None
            and stored_expected != expected_artifact_count
        ):
            raise ValueError(
                "Fingerprint expected artifact count changed while resuming: "
                f"{stored_expected} != {expected_artifact_count}"
            )
        return manifest

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": batch_dir.name,
        "config_hash": identity_hash,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "status": "generating",
        "expected_artifact_count": expected_artifact_count,
        "artifact_count": 0,
        "warnings": [],
        **copy.deepcopy(identity),
    }
    atomic_write_json(path, manifest)
    return manifest


def update_fingerprint_manifest(
    batch_dir: Path,
    *,
    status: Optional[str] = None,
    artifact_count: Optional[int] = None,
    warnings: Optional[Sequence[str]] = None,
    error: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    path = batch_dir / "fingerprint_config.json"
    manifest = read_json(path)
    if int(manifest.get("schema_version", 1)) < 2:
        return manifest
    if status is not None:
        manifest["status"] = status
    if artifact_count is not None:
        manifest["artifact_count"] = int(artifact_count)
    if warnings is not None:
        manifest["warnings"] = list(dict.fromkeys(str(item) for item in warnings))
    if error is not None:
        manifest["error"] = dict(error)
    elif status in {"generating", "completed", "completed_with_warnings"}:
        manifest.pop("error", None)
    manifest["updated_at"] = utc_now()
    atomic_write_json(path, manifest)
    return manifest


def clone_evaluation_variant(batch_dir: Path) -> Path:
    base_name, _ = experiment_variant(batch_dir)
    target = next_variant_directory(batch_dir.parent, base_name)
    target.mkdir(parents=True, exist_ok=False)
    config = read_json(batch_dir / "fingerprint_config.json")
    previous_experiment_id = config["experiment_id"]
    config["experiment_id"] = target.name
    config["cloned_from"] = previous_experiment_id
    config["cloned_at"] = utc_now()
    atomic_write_json(target / "fingerprint_config.json", config)
    for source_path in artifact_paths(batch_dir):
        record = read_json(source_path)
        item_index = int(record.get("item_index", int(source_path.stem)))
        record["fingerprint_id"] = f"{target.name}:{item_index:03d}"
        atomic_write_json(target / source_path.name, record)
    return target


def evaluation_hashes_in(batch_dir: Path) -> set[str]:
    hashes = set()
    for name in EVALUATION_NAMES:
        path = batch_dir / f"{name}.json"
        if not path.is_file():
            continue
        value = read_json(path).get("evaluation_config_hash")
        if value:
            hashes.add(value)
    return hashes


def resolve_evaluation_directory(batch_dir: Path, evaluation_hash: str) -> Path:
    stored_hashes = evaluation_hashes_in(batch_dir)
    if not stored_hashes or stored_hashes == {evaluation_hash}:
        return batch_dir
    base_name, _ = experiment_variant(batch_dir)
    source_hash = read_json(batch_dir / "fingerprint_config.json").get("config_hash")
    for candidate in sorted(batch_dir.parent.glob(f"{base_name}_*")):
        config_path = candidate / "fingerprint_config.json"
        if not config_path.is_file():
            continue
        if read_json(config_path).get("config_hash") != source_hash:
            continue
        if evaluation_hashes_in(candidate) == {evaluation_hash}:
            return candidate
    return clone_evaluation_variant(batch_dir)


def normalize_model_entry(entry: Any, default_role: Optional[str] = None) -> Dict[str, Any]:
    if isinstance(entry, str):
        return {"model_name": entry, "role": default_role}
    if not isinstance(entry, dict) or not entry.get("model_name"):
        raise ValueError(f"Model entries require model_name: {entry!r}")
    normalized = dict(entry)
    normalized.setdefault("role", default_role)
    return normalized


def model_groups(config: Mapping[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    groups = config.get("model_groups")
    if not isinstance(groups, dict):
        raise ValueError("Evaluation config requires model_groups.")
    result = {
        "original": [
            normalize_model_entry(entry, "original")
            for entry in groups.get("original", [])
        ],
        "derivatives": [
            normalize_model_entry(entry, "derivative")
            for entry in groups.get("derivatives", [])
        ],
        "negatives": [
            normalize_model_entry(entry, "negative")
            for entry in groups.get("negatives", [])
        ],
    }
    if len(result["original"]) != 1:
        raise ValueError("model_groups.original must contain exactly one model.")
    source_model_name = (config.get("source_model") or {}).get("model_name")
    if result["original"][0]["model_name"] != source_model_name:
        raise ValueError(
            "model_groups.original must be the source model named by source_model."
        )
    for entry in result["derivatives"]:
        if not entry.get("modification_type"):
            raise ValueError(
                f"Derivative model {entry['model_name']} requires modification_type."
            )
    for entry in result["negatives"]:
        if not entry.get("negative_type"):
            raise ValueError(
                f"Negative model {entry['model_name']} requires negative_type."
            )
        threshold_split = entry.get("threshold_split")
        if threshold_split not in {None, "calibration", "held_out"}:
            raise ValueError(
                f"Negative model {entry['model_name']} has invalid threshold_split "
                f"{threshold_split!r}; expected calibration or held_out."
            )
    names = [
        entry["model_name"]
        for group_entries in result.values()
        for entry in group_entries
    ]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            "Models cannot appear in multiple model groups: " + ", ".join(duplicates)
        )
    return result


def require_models(benchmark: Any, entries: Iterable[Mapping[str, Any]]) -> None:
    available = set(benchmark.get_all_models())
    missing = sorted({entry["model_name"] for entry in entries} - available)
    if missing:
        raise ValueError(
            "Evaluation models are absent from the benchmark config: " + ", ".join(missing)
        )


@contextlib.contextmanager
def temporary_model_parameters(
    model: Any,
    generation: Optional[Mapping[str, Any]] = None,
    system_prompt: Any = ...,
):
    previous = copy.deepcopy(model.params or {})
    updated = copy.deepcopy(previous)
    for key, value in (generation or {}).items():
        if key not in {"seed", "input_mode"}:
            updated[key] = value
    if system_prompt is not ...:
        if system_prompt is None:
            updated.pop("system_prompt", None)
        else:
            updated["system_prompt"] = system_prompt
    model.params = updated
    try:
        yield
    finally:
        model.params = previous


def verify_model(
    method: Any,
    source_model: Any,
    testing_model: Any,
    generation: Optional[Mapping[str, Any]],
    system_prompt: Any = ...,
) -> Dict[str, Any]:
    with temporary_model_parameters(testing_model, generation, system_prompt):
        result = method.verify_fingerprint(source_model, testing_model, generation)
    value = result.to_dict()
    value.update(
        {
            "evaluation_model": testing_model.model_name,
            "model_type": testing_model.type,
            "generation": dict(generation or {}),
        }
    )
    return value


def mean_numeric(records: Sequence[Mapping[str, Any]], field: str = "metrics") -> Dict[str, float]:
    values: Dict[str, List[float]] = defaultdict(list)
    for record in records:
        for name, value in record.get(field, {}).items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values[name].append(float(value))
    return {name: float(np.mean(items)) for name, items in values.items() if items}


def safe_ratio(value: float, baseline: float) -> Optional[float]:
    return float(value / baseline) if baseline else None


def effective_decoding_seeds(
    seeds: Sequence[Any], generation: Mapping[str, Any]
) -> tuple[List[int], List[int]]:
    requested = [int(seed) for seed in seeds]
    if not requested:
        raise ValueError("At least one decoding seed is required.")
    effective = requested if generation.get("do_sample", True) else [0]
    return requested, effective


def youden_threshold(scores: Sequence[float], labels: Sequence[int]) -> float:
    if len(scores) != len(labels) or not scores:
        raise ValueError("Youden threshold requires equally sized non-empty scores and labels.")
    if set(labels) != {0, 1}:
        raise ValueError("Youden threshold requires both positive and negative models.")
    candidates = sorted({float(score) for score in scores}, reverse=True)
    if not all(math.isfinite(candidate) for candidate in candidates):
        raise ValueError("Youden threshold scores must all be finite.")
    best_threshold = candidates[0]
    best_j = -float("inf")
    labels_array = np.asarray(labels)
    scores_array = np.asarray(scores, dtype=float)
    for threshold in candidates:
        predictions = scores_array >= threshold
        tpr = float(predictions[labels_array == 1].mean())
        fpr = float(predictions[labels_array == 0].mean())
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_threshold = threshold
    return float(best_threshold)


def run_model_modification(
    method: Any,
    source_model: Any,
    benchmark: Any,
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    spec: Mapping[str, Any],
    checkpoint=None,
    existing: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    generation = dict(spec.get("generation", {}))
    generation.setdefault("input_mode", "model_rendered")
    existing_rows = {
        row.get("evaluation_model"): row
        for row in (existing or {}).get("models", [])
        if isinstance(row, dict) and row.get("status") == "completed"
    }
    rows = []
    entries = list(groups["original"]) + list(groups["derivatives"])
    if (
        generation.get("input_mode") == "source_rendered"
        and getattr(method, "requires_model_rendered_cross_model", False)
        and any(entry["model_name"] != source_model.model_name for entry in entries)
    ):
        raise ValueError(
            "Cross-model modification evaluation requires input_mode: model_rendered."
        )
    for entry in entries:
        cached = existing_rows.get(entry["model_name"])
        if cached is not None:
            rows.append(cached)
            continue
        try:
            model = benchmark.get_all_models()[entry["model_name"]]
            result = verify_model(method, source_model, model, generation)
            result.update(
                {
                    "role": entry["role"],
                    "modification_type": entry.get("modification_type"),
                    "status": "completed",
                }
            )
        except Exception as error:
            result = {
                "evaluation_model": entry["model_name"],
                "role": entry["role"],
                "modification_type": entry.get("modification_type"),
                "status": "failed",
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            }
        rows.append(result)
        if checkpoint is not None:
            checkpoint({"models": rows, "summary": {}})

    completed = [row for row in rows if row.get("status") == "completed"]
    source_rows = [row for row in completed if row["role"] == "original"]
    baseline = source_rows[0]["score"] if source_rows else None
    if baseline is not None:
        for row in completed:
            row["metrics"]["retention_rate"] = safe_ratio(row["score"], baseline)
    derivative_rows = [row for row in completed if row["role"] == "derivative"]
    derivative_retentions = [
        row["metrics"]["retention_rate"]
        for row in derivative_rows
        if row["metrics"].get("retention_rate") is not None
    ]
    return {
        "models": rows,
        "summary": {
            "original_score": baseline,
            "mean_derivative_score": (
                float(np.mean([row["score"] for row in derivative_rows]))
                if derivative_rows
                else None
            ),
            "mean_derivative_retention_rate": (
                float(np.mean(derivative_retentions))
                if derivative_retentions
                else None
            ),
            "complete_panel": len(completed) == len(entries),
        },
        "failure_count": sum(row.get("status") == "failed" for row in rows),
    }


def deployment_conditions(spec: Mapping[str, Any], method: Any) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    conditions = []
    not_applicable = []
    system_specs = spec.get("system_prompts", [])
    if system_specs:
        if method.supports_evaluation("deployment_robustness", "system_prompts"):
            for item in system_specs:
                if not isinstance(item, dict) or not item.get("id"):
                    raise ValueError("Each system prompt condition requires an id.")
                generation = dict(item.get("generation", spec.get("generation", {})))
                generation.setdefault("seed", 0)
                generation["input_mode"] = (
                    "source_rendered" if item.get("prompt") is None else "model_rendered"
                )
                conditions.append(
                    {
                        "component": "system_prompts",
                        "condition_id": item["id"],
                        "system_prompt": item.get("prompt"),
                        "generation": generation,
                        "requested_seeds": [int(generation["seed"])],
                        "effective_seeds": [int(generation["seed"])],
                    }
                )
        else:
            not_applicable.append(
                {"component": "system_prompts", "reason": "unsupported_by_method"}
            )

    sampling = spec.get("sampling")
    if sampling:
        if not method.supports_evaluation("deployment_robustness", "sampling"):
            not_applicable.append(
                {"component": "sampling", "reason": "unsupported_by_method"}
            )
        else:
            seeds = sampling.get("seeds")
            if not isinstance(seeds, list) or not seeds:
                raise ValueError("deployment_robustness.sampling.seeds must be non-empty.")
            common = dict(sampling.get("generation", {}))
            temperatures = sampling.get("temperature_values", [])
            fixed_top_p = sampling.get("temperature_top_p")
            if temperatures and fixed_top_p is None:
                raise ValueError("temperature_top_p is required with temperature_values.")
            for temperature in temperatures:
                do_sample = float(temperature) > 0
                condition_seeds = seeds if do_sample else [0]
                for seed in condition_seeds:
                    conditions.append(
                        {
                            "component": "temperature",
                            "condition_id": f"temperature_{float(temperature):g}",
                            "system_prompt": ...,
                            "generation": {
                                **common,
                                "temperature": float(temperature),
                                "top_p": float(fixed_top_p),
                                "do_sample": do_sample,
                                "seed": int(seed),
                                "input_mode": "source_rendered",
                            },
                            "requested_seeds": [int(value) for value in seeds],
                            "effective_seeds": [int(value) for value in condition_seeds],
                        }
                    )
            top_p_values = sampling.get("top_p_values", [])
            fixed_temperature = sampling.get("top_p_temperature")
            if top_p_values and fixed_temperature is None:
                raise ValueError("top_p_temperature is required with top_p_values.")
            for top_p in top_p_values:
                for seed in seeds:
                    conditions.append(
                        {
                            "component": "top_p",
                            "condition_id": f"top_p_{float(top_p):g}",
                            "system_prompt": ...,
                            "generation": {
                                **common,
                                "temperature": float(fixed_temperature),
                                "top_p": float(top_p),
                                "do_sample": True,
                                "seed": int(seed),
                                "input_mode": "source_rendered",
                            },
                            "requested_seeds": [int(value) for value in seeds],
                            "effective_seeds": [int(value) for value in seeds],
                        }
                    )
    if not conditions and not not_applicable:
        raise ValueError("deployment_robustness defines no conditions.")
    return conditions, not_applicable


def run_deployment(
    method: Any,
    source_model: Any,
    benchmark: Any,
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    spec: Mapping[str, Any],
) -> Dict[str, Any]:
    from benchmark.model_interface import ModelInterface

    model_name = spec.get("model_name", groups["original"][0]["model_name"])
    if model_name not in benchmark.get_all_models():
        raise ValueError(
            f"Deployment evaluation model is absent from the benchmark config: {model_name}"
        )
    model = benchmark.get_all_models()[model_name]
    has_non_default_system_prompt = any(
        isinstance(item, dict) and item.get("prompt") is not None
        for item in spec.get("system_prompts", [])
    )
    if (
        has_non_default_system_prompt
        and type(model).render_prompts is ModelInterface.render_prompts
    ):
        raise ValueError(
            f"System prompts are ineffective for raw model {model_name}; select an "
            "instruction/deployment model or remove system_prompts."
        )
    conditions, not_applicable = deployment_conditions(spec, method)
    results = []
    for condition in conditions:
        result = verify_model(
            method,
            source_model,
            model,
            condition["generation"],
            condition["system_prompt"],
        )
        result["component"] = condition["component"]
        result["condition_id"] = condition["condition_id"]
        result["requested_seeds"] = condition["requested_seeds"]
        result["effective_seeds"] = condition["effective_seeds"]
        if condition["system_prompt"] is not ...:
            result["system_prompt"] = condition["system_prompt"]
        results.append(result)

    grouped = defaultdict(list)
    for result in results:
        grouped[(result["component"], result["condition_id"])].append(result)
    summary = []
    for (component, condition_id), rows in sorted(grouped.items()):
        summary.append(
            {
                "component": component,
                "condition_id": condition_id,
                "num_runs": len(rows),
                "requested_seeds": rows[0]["requested_seeds"],
                "effective_seeds": rows[0]["effective_seeds"],
                **mean_numeric(rows),
            }
        )
    return {"conditions": results, "not_applicable": not_applicable, "summary": summary}


def run_specificity(
    method: Any,
    source_model: Any,
    benchmark: Any,
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    spec: Mapping[str, Any],
    checkpoint=None,
    existing: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    positives = list(groups["original"]) + list(groups["derivatives"])
    negatives = list(groups["negatives"])
    if not positives or not negatives:
        raise ValueError("model_specificity requires positive and negative models.")
    configured_splits = {entry.get("threshold_split") for entry in negatives}
    explicit_split = any(value is not None for value in configured_splits)
    if explicit_split:
        if None in configured_splits or not {
            "calibration", "held_out"
        }.issubset(configured_splits):
            raise ValueError(
                "threshold_split requires every negative to be assigned and at "
                "least one calibration and one held_out negative."
            )
    seeds = spec.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("model_specificity.seeds must be a non-empty list.")
    base_generation = dict(spec.get("generation", {}))
    base_generation.setdefault("input_mode", "model_rendered")
    if (
        base_generation.get("input_mode") == "source_rendered"
        and getattr(method, "requires_model_rendered_cross_model", False)
        and any(
            entry["model_name"] != source_model.model_name
            for entry in positives + negatives
        )
    ):
        raise ValueError(
            "Cross-model specificity evaluation requires input_mode: model_rendered."
        )
    requested_seeds, effective_seeds = effective_decoding_seeds(
        seeds, base_generation
    )
    existing_rows = {
        row.get("evaluation_model"): row
        for row in (existing or {}).get("models", [])
        if isinstance(row, dict) and row.get("status") == "completed"
    }
    rows = []
    labeled_groups = (
        (1, list(groups["original"])),
        (None, list(groups["derivatives"])),
        (0, negatives),
    )
    for label, entries in labeled_groups:
        for entry in entries:
            cached = existing_rows.get(entry["model_name"])
            if cached is not None:
                rows.append(cached)
                continue
            try:
                model = benchmark.get_all_models()[entry["model_name"]]
                seed_rows = []
                for seed in effective_seeds:
                    generation = {**base_generation, "seed": int(seed)}
                    seed_rows.append(
                        verify_model(method, source_model, model, generation)
                    )
                row = {
                    "evaluation_model": entry["model_name"],
                    "label": label,
                    "role": entry["role"],
                    "negative_type": entry.get("negative_type"),
                    "threshold_split": entry.get("threshold_split"),
                    "score": float(np.mean([item["score"] for item in seed_rows])),
                    "metrics": mean_numeric(seed_rows),
                    "runs": seed_rows,
                    "status": "completed",
                }
            except Exception as error:
                row = {
                    "evaluation_model": entry["model_name"],
                    "label": label,
                    "role": entry["role"],
                    "negative_type": entry.get("negative_type"),
                    "threshold_split": entry.get("threshold_split"),
                    "status": "failed",
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    },
                }
            rows.append(row)
            if checkpoint is not None:
                checkpoint(
                    {
                        "models": rows,
                        "summary": {},
                        "requested_seeds": requested_seeds,
                        "effective_seeds": effective_seeds,
                    }
                )

    completed = [row for row in rows if row.get("status") == "completed"]
    source_rows = [row for row in completed if row["role"] == "original"]
    negative_rows = [row for row in completed if row["role"] == "negative"]
    derivative_rows = [row for row in completed if row["role"] == "derivative"]
    source_score = source_rows[0]["score"] if source_rows else None
    warnings = []
    if len(negatives) < 6:
        warnings.append(
            f"small_panel: only {len(negatives)} negative models are configured"
        )

    pairwise_auc = None
    source_margin = None
    rank = {
        "status": "unavailable",
        "lower": None,
        "upper": None,
        "tie_count": None,
    }
    if source_score is not None and negative_rows:
        negative_scores = [row["score"] for row in negative_rows]
        pairwise_auc = float(
            np.mean(
                [
                    (source_score > negative)
                    + 0.5 * (source_score == negative)
                    for negative in negative_scores
                ]
            )
        )
        source_margin = float(source_score - max(negative_scores))
        panel_scores = [source_score, *negative_scores]
        greater = sum(score > source_score for score in panel_scores)
        tied = sum(score == source_score for score in panel_scores)
        if len(set(panel_scores)) == 1:
            rank = {
                "status": "degenerate_all_equal",
                "lower": None,
                "upper": None,
                "tie_count": tied,
            }
        else:
            rank = {
                "status": "ranked_with_ties" if tied > 1 else "ranked",
                "lower": 1 + greater,
                "upper": greater + tied,
                "tie_count": tied,
            }

    calibration_rows = (
        [row for row in negative_rows if row.get("threshold_split") == "calibration"]
        if explicit_split
        else negative_rows
    )
    held_out_rows = (
        [row for row in negative_rows if row.get("threshold_split") == "held_out"]
        if explicit_split
        else []
    )
    threshold = None
    threshold_status = "unavailable"
    in_sample_fpr = None
    held_out_fpr = None
    calibration_scores = (
        [source_score, *[row["score"] for row in calibration_rows]]
        if source_score is not None and calibration_rows
        else []
    )
    if calibration_scores:
        if len(set(calibration_scores)) == 1:
            threshold_status = "degenerate_all_equal"
        else:
            threshold = youden_threshold(
                calibration_scores,
                [1] + [0] * len(calibration_rows),
            )
            threshold_status = (
                "held_out" if explicit_split else "descriptive_in_sample"
            )
            for row in completed:
                row["predicted_positive"] = int(row["score"] >= threshold)
            in_sample_fpr = float(
                np.mean([row["predicted_positive"] for row in calibration_rows])
            )
            if held_out_rows:
                held_out_fpr = float(
                    np.mean([row["predicted_positive"] for row in held_out_rows])
                )
    if threshold is None:
        for row in completed:
            row["predicted_positive"] = None

    type_summary = []
    for negative_type in sorted({row["negative_type"] for row in negative_rows}):
        type_rows = [row for row in negative_rows if row["negative_type"] == negative_type]
        metrics = mean_numeric(type_rows)
        event_success = []
        event_invalid = []
        for row in type_rows:
            for run in row["runs"]:
                event_success.extend(trial.get("success", 0) for trial in run.get("trials", []))
                event_invalid.extend(trial.get("invalid", 0) for trial in run.get("trials", []))
        type_summary.append(
            {
                "negative_type": negative_type,
                "model_fpr": (
                    float(np.mean([row["predicted_positive"] for row in type_rows]))
                    if threshold is not None
                    else None
                ),
                "event_fpr": float(np.mean(event_success)) if event_success else None,
                "invalid_rate": float(np.mean(event_invalid)) if event_invalid else None,
                **metrics,
            }
        )
    return {
        "threshold": {
            "strategy": "youden_j",
            "value": threshold,
            "status": threshold_status,
            "calibration_source": (
                "source_and_calibration_negatives"
                if explicit_split
                else "source_and_same_panel_negatives"
            ),
            "in_sample_descriptive_fpr": in_sample_fpr,
            "held_out_fpr": held_out_fpr,
        },
        "models": rows,
        "summary": {
            "roc_auc": pairwise_auc,
            "roc_auc_definition": "source_vs_negative_models_only",
            "source_score_rank": rank["lower"],
            "source_rank": rank,
            "source_score_margin_over_best_negative": source_margin,
            "in_sample_descriptive_fpr": in_sample_fpr,
            "held_out_fpr": held_out_fpr,
            "overall_model_fpr": held_out_fpr,
            "negative_model_count": len(negative_rows),
            "configured_negative_model_count": len(negatives),
            "auc_resolution": (
                0.5 / len(negative_rows) if negative_rows else None
            ),
            "derivative_robustness": [
                {
                    "evaluation_model": row["evaluation_model"],
                    "score": row["score"],
                    "absolute_drop": (
                        float(source_score - row["score"])
                        if source_score is not None
                        else None
                    ),
                    "retention_rate": (
                        safe_ratio(row["score"], source_score)
                        if source_score is not None
                        else None
                    ),
                }
                for row in derivative_rows
            ],
            "complete_panel": len(completed) == len(positives) + len(negatives),
            "by_negative_type": type_summary,
        },
        "requested_seeds": requested_seeds,
        "effective_seeds": effective_seeds,
        "warnings": warnings,
        "failure_count": sum(row.get("status") == "failed" for row in rows),
    }


def load_calibration_texts(spec: Mapping[str, Any]) -> List[str]:
    column = spec.get("column")
    if not column:
        raise ValueError("Calibration configuration requires column.")
    if spec.get("path"):
        path = Path(spec["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Calibration data does not exist: {path}")
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows or column not in rows[0]:
            raise ValueError(
                f"Calibration CSV must contain column {column!r}: {path}"
            )
        texts = [row[column] for row in rows if row.get(column)]
    elif spec.get("dataset"):
        from datasets import load_dataset

        dataset = load_dataset(
            spec["dataset"],
            spec.get("subset"),
            split=spec.get("split", "test"),
        )
        if column not in dataset.column_names:
            raise ValueError(
                f"Calibration dataset must contain column {column!r}; "
                f"found {dataset.column_names}."
            )
        texts = [value for value in dataset[column] if value]
    else:
        raise ValueError("Calibration requires either path or dataset.")
    sample_size = int(spec.get("sample_size", len(texts)))
    if sample_size > len(texts):
        raise ValueError(
            f"Calibration requested {sample_size} texts but only {len(texts)} exist."
        )
    rng = np.random.default_rng(int(spec.get("seed", 0)))
    indices = rng.choice(len(texts), size=sample_size, replace=False)
    return [texts[index] for index in indices]


def calculate_log_ppl(
    texts: Sequence[str],
    model: Any,
    tokenizer: Any,
    max_length: int,
) -> List[float]:
    import torch

    values = []
    device = next(model.parameters()).device
    for text in texts:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoded["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            raise ValueError("Prompt stealthiness cannot score an input shorter than two tokens.")
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
        values.append(float(output.loss.detach().cpu()))
    return values


def run_stealthiness(
    method: Any,
    records: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
) -> Dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    texts = method.stealth_texts(records)
    if not texts:
        raise ValueError("This fingerprint batch exposes no textual prompts to score.")
    reference_model = spec.get("reference_model")
    if not reference_model:
        raise ValueError("prompt_stealthiness.reference_model is required.")
    calibration_spec = spec.get("calibration")
    if not isinstance(calibration_spec, dict):
        raise ValueError("prompt_stealthiness.calibration is required.")
    quantile = float(calibration_spec.get("quantile", 0.999))
    if not 0 < quantile < 1:
        raise ValueError("Calibration quantile must be between zero and one.")
    max_length = int(spec.get("max_length", 1024))
    tokenizer = AutoTokenizer.from_pretrained(reference_model)
    model = AutoModelForCausalLM.from_pretrained(reference_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    calibration_texts = load_calibration_texts(calibration_spec)
    calibration_values = calculate_log_ppl(
        calibration_texts, model, tokenizer, max_length
    )
    threshold = float(np.quantile(calibration_values, quantile))
    prompt_values = calculate_log_ppl(
        [item["text"] for item in texts], model, tokenizer, max_length
    )
    trials = []
    for item, value in zip(texts, prompt_values):
        trials.append(
            {
                **item,
                "log_ppl": value,
                "passes_filter": int(value <= threshold),
            }
        )
    summary = []
    for kind in sorted({item["kind"] for item in trials}):
        kind_rows = [item for item in trials if item["kind"] == kind]
        summary.append(
            {
                "input_kind": kind,
                "mean_log_ppl": float(np.mean([item["log_ppl"] for item in kind_rows])),
                "filter_pass_rate": float(
                    np.mean([item["passes_filter"] for item in kind_rows])
                ),
                "num_prompts": len(kind_rows),
            }
        )
    return {
        "calibration": {
            "reference_model": reference_model,
            "sample_size": len(calibration_values),
            "quantile": quantile,
            "threshold_log_ppl": threshold,
            "mean_calibration_log_ppl": float(np.mean(calibration_values)),
        },
        "trials": trials,
        "summary": summary,
    }


def result_status(batch_dir: Path, name: str) -> str:
    path = batch_dir / f"{name}.json"
    if not path.is_file():
        return "pending"
    try:
        return str(read_json(path).get("status", "unknown"))
    except (OSError, ValueError, json.JSONDecodeError):
        return "invalid"


def update_experiments_index(results_root: Path) -> None:
    rows = []
    for directory in sorted(results_root.glob("exp_*")):
        config_path = directory / "fingerprint_config.json"
        if not directory.is_dir() or not config_path.is_file():
            continue
        config = read_json(config_path)
        source = config.get("source_model", {})
        parameters = config.get("fingerprint_parameters", {})
        method_parameters = {
            key: value
            for key, value in parameters.items()
            if key
            not in {
                "fingerprint_method",
                "fingerprint_type",
                "seed",
                "cached_fingerprints_path",
                "re_fingerprinting",
            }
        }
        rows.append(
            [
                directory.name,
                source.get("model_name", "?"),
                config.get("fingerprint_method", "?"),
                str(config.get("optimization_seed", "?")),
                str(config.get("artifact_count", "?")),
                f"`{canonical_json(method_parameters)}`",
                f"`{str(config.get('config_hash', ''))[:8]}`",
                *[result_status(directory, name) for name in EVALUATION_NAMES],
            ]
        )
    header = [
        "experiment",
        "source model",
        "method",
        "fingerprint seed",
        "artifacts",
        "generation parameters",
        "config hash",
        *EVALUATION_NAMES,
    ]
    lines = [
        "# Fingerprint experiments",
        "",
        "This file is generated from each batch's `fingerprint_config.json` and evaluation results. Do not edit it manually.",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        row[0] = f"[{row[0]}](./{row[0]}/)"
        safe = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(safe) + " |")
    lines.append("")
    atomic_write_text(results_root / "EXPERIMENTS.md", "\n".join(lines))


def make_result_envelope(
    name: str,
    batch_dir: Path,
    method_name: str,
    source_model_name: str,
    evaluation_config: Mapping[str, Any],
    evaluation_hash: str,
    status: str,
    payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_name": name,
        "status": status,
        "experiment_id": batch_dir.name,
        "fingerprint_method": method_name,
        "source_model": source_model_name,
        "evaluation_config_hash": evaluation_hash,
        "evaluation_config": copy.deepcopy(evaluation_config),
        "updated_at": utc_now(),
        **(dict(payload or {})),
    }


def should_run_result(
    path: Path,
    evaluation_hash: str,
    retry_failed: bool,
    overwrite: bool,
) -> bool:
    if not path.is_file():
        return True
    existing = read_json(path)
    if existing.get("evaluation_config_hash") != evaluation_hash:
        raise ValueError(
            f"Evaluation config mismatch remained after variant resolution: {path}"
        )
    status = existing.get("status")
    if overwrite:
        return True
    if status in {"failed", "partial_failed", "running"}:
        return retry_failed
    return False


def run_evaluations(args: argparse.Namespace) -> int:
    from accelerate import Accelerator
    from benchmark.benchmark import Benchmark
    from fingerprint.fingerprint_factory import create_fingerprint_method

    benchmark_config_path = Path(args.benchmark_config).resolve()
    fingerprint_config_path = Path(args.fingerprint_config).resolve()
    evaluation_config_path = Path(args.evaluation_config).resolve()
    benchmark_config = load_yaml(benchmark_config_path)
    fingerprint_config = load_yaml(fingerprint_config_path)
    if args.optimization_seed is not None:
        fingerprint_config["seed"] = int(args.optimization_seed)
    evaluation_config = load_yaml(evaluation_config_path)
    evaluation_hash = stable_hash(evaluation_config)
    batch_dir = Path(args.batch_dir).resolve()
    if not (batch_dir / "fingerprint_config.json").is_file():
        raise FileNotFoundError(f"Not a fingerprint batch: {batch_dir}")
    batch_dir = resolve_evaluation_directory(batch_dir, evaluation_hash)
    print(f"Evaluation batch: {batch_dir}")

    batch_config = read_json(batch_dir / "fingerprint_config.json")
    if int(batch_config.get("schema_version", 1)) >= 2:
        if batch_config.get("status") not in {"completed", "completed_with_warnings"}:
            raise ValueError(
                "Evaluation requires a complete fingerprint batch; generation status is "
                f"{batch_config.get('status')!r}."
            )
    if batch_config.get("config_hash") != stable_hash(
        {
            key: batch_config[key]
            for key in (
                "fingerprint_method",
                "optimization_seed",
                "source_model",
                "fingerprint_parameters",
                "benchmark_config",
            )
        }
    ):
        raise ValueError("fingerprint_config.json identity hash is invalid.")
    if batch_config["fingerprint_method"] != fingerprint_config.get("fingerprint_method"):
        raise ValueError("Fingerprint method config does not match the saved batch.")
    current_parameters = {
        key: value
        for key, value in fingerprint_config.items()
        if key not in {"cached_fingerprints_path", "re_fingerprinting"}
    }
    if batch_config["fingerprint_parameters"] != current_parameters:
        raise ValueError(
            "Fingerprint parameters do not match the saved batch; use the exact "
            "generation configuration recorded in fingerprint_config.json."
        )
    if batch_config["benchmark_config"]["sha256"] != file_hash(benchmark_config_path):
        raise ValueError(
            "Benchmark model registry differs from the one used to generate this batch."
        )

    groups = model_groups(evaluation_config)
    evaluations = evaluation_config.get("evaluations")
    if not isinstance(evaluations, dict):
        raise ValueError("Evaluation config requires an evaluations mapping.")
    unknown_evaluations = set(evaluations) - set(EVALUATION_NAMES)
    if unknown_evaluations:
        raise ValueError(
            "Unknown evaluation name(s): " + ", ".join(sorted(unknown_evaluations))
        )
    for name, spec in evaluations.items():
        if not isinstance(spec, dict):
            raise ValueError(f"evaluations.{name} must be a mapping.")
    configured_source = evaluation_config.get("source_model", {})
    source_model_name = configured_source.get("model_name")
    if source_model_name != batch_config["source_model"]["model_name"]:
        raise ValueError(
            "Evaluation source model does not match fingerprint batch: "
            f"{source_model_name!r} != {batch_config['source_model']['model_name']!r}"
        )

    accelerator = Accelerator()
    method = create_fingerprint_method(fingerprint_config, accelerator=accelerator)
    benchmark = Benchmark(
        benchmark_config,
        accelerator=accelerator,
        fingerprint_type=fingerprint_config.get("fingerprint_type", "black-box"),
        fingerprint_method=fingerprint_config.get("fingerprint_method"),
    )
    required_entries = list(groups["original"])
    if (evaluations.get("model_modification_robustness") or {}).get("enabled"):
        required_entries.extend(groups["derivatives"])
    if (evaluations.get("model_specificity") or {}).get("enabled"):
        required_entries.extend(groups["derivatives"])
        required_entries.extend(groups["negatives"])
    deployment_spec = evaluations.get("deployment_robustness") or {}
    if deployment_spec.get("enabled") and deployment_spec.get("model_name"):
        required_entries.append(
            {"model_name": deployment_spec["model_name"], "role": "deployment"}
        )
    require_models(benchmark, required_entries)
    if source_model_name not in benchmark.get_all_models():
        raise ValueError(f"Source model not found in benchmark config: {source_model_name}")
    records = load_records(batch_dir)
    method.prepare_evaluation(
        records,
        train_models=benchmark.get_training_models(),
    )
    source_model = benchmark.get_all_models()[source_model_name]
    source_model.set_fingerprint(method.fingerprint_from_records(records))
    source_model.fingerprint_records = records

    runners = {
        "model_modification_robustness": lambda spec, checkpoint=None, existing=None: run_model_modification(
            method, source_model, benchmark, groups, spec, checkpoint, existing
        ),
        "deployment_robustness": lambda spec: run_deployment(
            method, source_model, benchmark, groups, spec
        ),
        "model_specificity": lambda spec, checkpoint=None, existing=None: run_specificity(
            method, source_model, benchmark, groups, spec, checkpoint, existing
        ),
        "prompt_stealthiness": lambda spec: run_stealthiness(method, records, spec),
    }

    failures = 0
    for name in EVALUATION_NAMES:
        spec = evaluations.get(name, {})
        if not isinstance(spec, dict):
            raise ValueError(f"evaluations.{name} must be a mapping.")
        if not spec.get("enabled", False):
            continue
        path = batch_dir / f"{name}.json"
        if not should_run_result(
            path, evaluation_hash, args.retry_failed, args.overwrite
        ):
            print(f"Skipping {name}: existing result policy")
            continue
        if not method.supports_evaluation(name):
            result = make_result_envelope(
                name,
                batch_dir,
                batch_config["fingerprint_method"],
                source_model_name,
                evaluation_config,
                evaluation_hash,
                "not_applicable",
                {"reason": "unsupported_by_method"},
            )
            atomic_write_json(path, result)
            continue
        try:
            existing = read_json(path) if path.is_file() else None

            def checkpoint(partial_payload):
                partial_result = make_result_envelope(
                    name,
                    batch_dir,
                    batch_config["fingerprint_method"],
                    source_model_name,
                    evaluation_config,
                    evaluation_hash,
                    "running",
                    partial_payload,
                )
                atomic_write_json(path, partial_result)

            if name in {"model_modification_robustness", "model_specificity"}:
                payload = runners[name](spec, checkpoint, existing)
            else:
                payload = runners[name](spec)
            failure_count = int(payload.get("failure_count", 0))
            status = "partial_failed" if failure_count else "completed"
            if failure_count:
                failures += 1
            result = make_result_envelope(
                name,
                batch_dir,
                batch_config["fingerprint_method"],
                source_model_name,
                evaluation_config,
                evaluation_hash,
                status,
                payload,
            )
        except Exception as error:
            failures += 1
            partial = read_json(path) if path.is_file() else {}
            preserved = {
                key: partial[key]
                for key in (
                    "models", "summary", "requested_seeds", "effective_seeds"
                )
                if key in partial
            }
            preserved["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
            result = make_result_envelope(
                name,
                batch_dir,
                batch_config["fingerprint_method"],
                source_model_name,
                evaluation_config,
                evaluation_hash,
                "partial_failed" if preserved.get("models") else "failed",
                preserved,
            )
        atomic_write_json(path, result)
        print(f"{name}: {result['status']}")

    update_experiments_index(batch_dir.parent)
    return 1 if failures else 0


def generate_batch(args: argparse.Namespace) -> int:
    from accelerate import Accelerator
    from benchmark.benchmark import Benchmark
    from fingerprint.fingerprint_factory import create_fingerprint_method

    benchmark_config_path = Path(args.benchmark_config).resolve()
    fingerprint_config_path = Path(args.fingerprint_config).resolve()
    benchmark_config = load_yaml(benchmark_config_path)
    fingerprint_config = load_yaml(fingerprint_config_path)
    if args.optimization_seed is not None:
        fingerprint_config["seed"] = int(args.optimization_seed)
    # The optimization seed is part of the immutable experiment identity, so it
    # must govern every RNG used before and during fingerprint construction.
    # This covers Python-based TRAP target generation and ZeroPrint query/
    # substitution sampling in addition to NumPy and Torch operations.
    generation_seed = int(fingerprint_config["seed"])
    seed_fingerprint_generation(generation_seed)
    source_model_name = args.source_model
    model_alias = args.model_alias or source_model_name

    accelerator = Accelerator()
    method = create_fingerprint_method(fingerprint_config, accelerator=accelerator)
    benchmark = Benchmark(
        benchmark_config,
        accelerator=accelerator,
        fingerprint_type=fingerprint_config.get("fingerprint_type", "black-box"),
        fingerprint_method=fingerprint_config.get("fingerprint_method"),
    )
    if source_model_name not in benchmark.get_all_models():
        raise ValueError(f"Source model not found in benchmark config: {source_model_name}")
    source_model = benchmark.get_all_models()[source_model_name]
    if source_model.type not in set(method.candidate_model_types):
        raise ValueError(
            f"{source_model_name} has type {source_model.type!r}, but "
            f"{type(method).__name__} requires {method.candidate_model_types}."
        )

    identity = generation_identity(
        fingerprint_config, source_model, benchmark_config_path
    )
    identity_hash = stable_hash(identity)
    results_root = Path(args.results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    base_name = experiment_base_name(
        model_alias,
        fingerprint_config["fingerprint_method"],
        int(fingerprint_config["seed"]),
    )
    batch_dir = resolve_generation_directory(results_root, base_name, identity_hash)
    method.prepare(train_models=benchmark.get_training_models())
    expected_count = method.expected_artifact_count()
    manifest = initialize_fingerprint_manifest(batch_dir, identity, expected_count)
    existing_records = (
        load_records(batch_dir, allow_incomplete=True)
        if artifact_paths(batch_dir)
        else []
    )
    method.validate_partial_records(existing_records)
    if int(manifest.get("schema_version", 1)) < 2:
        if existing_records:
            print(f"Fingerprint batch already exists: {batch_dir}")
            update_experiments_index(results_root)
            return 0
        raise ValueError("A legacy fingerprint manifest cannot be resumed without artifacts.")
    if manifest.get("status") in {"completed", "completed_with_warnings"}:
        if expected_count is None or len(existing_records) == expected_count:
            print(f"Fingerprint batch already exists: {batch_dir}")
            update_experiments_index(results_root)
            return 0

    warnings = list(manifest.get("warnings", []))
    for record in existing_records:
        warnings.extend(
            str(item)
            for item in (record.get("metadata") or {}).get("quality_warnings", [])
        )
    if expected_count is not None and len(existing_records) == expected_count:
        final_status = "completed_with_warnings" if warnings else "completed"
        update_fingerprint_manifest(
            batch_dir,
            status=final_status,
            artifact_count=len(existing_records),
            warnings=warnings,
        )
        update_experiments_index(results_root)
        print(f"Fingerprint batch finalized after recovery: {batch_dir}")
        return 0
    update_fingerprint_manifest(
        batch_dir,
        status="generating",
        artifact_count=len(existing_records),
        warnings=warnings,
    )
    try:
        next_index = len(existing_records) + 1
        width = max(3, len(str(expected_count or next_index)))
        for record in method.iter_fingerprint_records(
            source_model, batch_dir.name, start_index=next_index
        ):
            item_index = int(record.get("item_index", 0))
            if item_index != next_index:
                raise ValueError(
                    "Fingerprint records must be yielded contiguously; expected "
                    f"{next_index}, got {item_index}."
                )
            path = batch_dir / f"{item_index:0{width}d}.json"
            atomic_write_json(path, record)
            record_warnings = (record.get("metadata") or {}).get(
                "quality_warnings", []
            )
            warnings.extend(str(item) for item in record_warnings)
            update_fingerprint_manifest(
                batch_dir,
                status="generating",
                artifact_count=item_index,
                warnings=warnings,
            )
            next_index += 1

        actual_count = len(artifact_paths(batch_dir))
        if expected_count is not None and actual_count != expected_count:
            raise ValueError(
                f"Fingerprint generation produced {actual_count} artifacts; "
                f"expected {expected_count}."
            )
        final_status = "completed_with_warnings" if warnings else "completed"
        update_fingerprint_manifest(
            batch_dir,
            status=final_status,
            artifact_count=actual_count,
            warnings=warnings,
        )
    except Exception as error:
        update_fingerprint_manifest(
            batch_dir,
            status="generation_failed",
            artifact_count=len(artifact_paths(batch_dir)),
            warnings=warnings,
            error={
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    update_experiments_index(results_root)
    print(f"Fingerprint batch generated: {batch_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate immutable fingerprints, then evaluate them separately."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    method_inputs = argparse.ArgumentParser(add_help=False)
    method_inputs.add_argument("--benchmark-config", required=True)
    method_inputs.add_argument("--fingerprint-config", required=True)

    generate = subparsers.add_parser("generate", parents=[method_inputs])
    generate.add_argument("--source-model", required=True)
    generate.add_argument("--model-alias")
    generate.add_argument("--optimization-seed", type=int)
    generate.add_argument("--results-root", default="results")
    generate.set_defaults(handler=generate_batch)

    run = subparsers.add_parser("run", parents=[method_inputs])
    run.add_argument("--evaluation-config", required=True)
    run.add_argument("--batch-dir", required=True)
    run.add_argument("--optimization-seed", type=int)
    run.add_argument("--retry-failed", action="store_true")
    run.add_argument("--overwrite", action="store_true")
    run.set_defaults(handler=run_evaluations)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
