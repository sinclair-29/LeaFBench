#!/usr/bin/env python3
"""Strict, model-free acceptance checks for the TRAP/PlugAE smoke experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


EXPECTED_SEEDS = list(range(10))
EXPECTED_MODELS = {
    "Phi-3-Mini-4K-Base": "original",
    "Phi-3-Mini-4K-Instruct": "derivative",
    "Qwen2.5-7B": "negative",
    "Gemma-2-2B": "negative",
}


def finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def same_number(left: Any, right: Any) -> bool:
    left_number = finite_float(left)
    right_number = finite_float(right)
    return (
        left_number is not None
        and right_number is not None
        and math.isclose(left_number, right_number)
    )


@dataclass
class Verdict:
    protocol_errors: list[str] = field(default_factory=list)
    quality_errors: list[str] = field(default_factory=list)
    observations: dict[str, Any] = field(default_factory=dict)

    def protocol(self, condition: bool, message: str) -> None:
        if not condition:
            self.protocol_errors.append(message)

    def quality(self, condition: bool, message: str) -> None:
        if not condition:
            self.quality_errors.append(message)


def read_json(path: Path, verdict: Verdict) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, dict):
            raise ValueError("root is not an object")
        return value
    except Exception as error:
        verdict.protocol_errors.append(
            f"cannot read {path}: {type(error).__name__}: {error}"
        )
        return {}


def read_yaml(path: Path, verdict: Verdict) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
        if not isinstance(value, dict):
            raise ValueError("root is not a mapping")
        return value
    except Exception as error:
        verdict.protocol_errors.append(
            f"cannot read {path}: {type(error).__name__}: {error}"
        )
        return {}


def validate_configs(config_root: Path, verdict: Verdict) -> None:
    trap = read_yaml(config_root / "trap_phi3.yaml", verdict)
    plugae = read_yaml(config_root / "plugae_phi3.yaml", verdict)
    evaluation = read_yaml(config_root / "eval_phi3.yaml", verdict)
    if not trap or not plugae or not evaluation:
        return

    verdict.protocol(trap.get("n_goals") == 3, "TRAP smoke sample count must be 3")
    verdict.protocol(trap.get("goal_count") == 3, "TRAP goal_count must be 3")
    gcg = trap.get("gcg_config") or {}
    expected_gcg = {
        "num_steps": 1500,
        "search_width": 512,
        "batch_size": 512,
        "topk": 256,
        "n_replace": 1,
        "early_stop": False,
    }
    for key, expected in expected_gcg.items():
        verdict.protocol(
            gcg.get(key) == expected,
            f"TRAP {key} changed: expected {expected!r}, got {gcg.get(key)!r}",
        )
    expected_plugae = {
        "num_queries": 3,
        "learning_rate": 0.1,
        "epochs": 30,
        "optimization_batch_size": 4,
        "generation_batch_size": 8,
    }
    for key, expected in expected_plugae.items():
        verdict.protocol(
            plugae.get(key) == expected,
            f"PlugAE {key} changed: expected {expected!r}, got {plugae.get(key)!r}",
        )
    for name, config in (("TRAP", trap), ("PlugAE", plugae)):
        verdict.protocol(
            config.get("source_self_test_max_new_tokens") == 8,
            f"{name} source self-test and evaluation decoding lengths differ",
        )

    specificity = ((evaluation.get("evaluations") or {}).get("model_specificity") or {})
    deployment = ((evaluation.get("evaluations") or {}).get("deployment_robustness") or {})
    verdict.protocol(specificity.get("seeds") == EXPECTED_SEEDS, "specificity seeds changed")
    verdict.protocol(
        (specificity.get("generation") or {}).get("input_mode") == "model_rendered",
        "cross-model specificity must use model_rendered prompts",
    )
    verdict.protocol(
        (specificity.get("generation") or {}).get("do_sample") is False,
        "specificity must use greedy decoding",
    )
    sampling = deployment.get("sampling") or {}
    verdict.protocol(sampling.get("seeds") == EXPECTED_SEEDS, "deployment seeds changed")
    verdict.protocol(
        sampling.get("temperature_values") == [0.7],
        "deployment temperature changed",
    )


def discover_batch(results_root: Path, method: str, verdict: Verdict) -> Path | None:
    candidates = sorted((results_root / method).glob("exp_*"))
    candidates = [path for path in candidates if path.is_dir()]
    verdict.protocol(
        len(candidates) == 1,
        f"expected exactly one {method} batch, found {len(candidates)}",
    )
    return candidates[0] if len(candidates) == 1 else None


def validate_trial_flags(method: str, trial: Mapping[str, Any], where: str, verdict: Verdict) -> None:
    output = "" if trial.get("output") is None else str(trial.get("output"))
    if method == "trap":
        target = str(trial.get("target", ""))
        if target.isdigit():
            match = re.search(rf"(?<!\d)\d{{{len(target)}}}(?!\d)", output)
            parsed = match.group(0) if match else None
        else:
            parsed = target if target and target in output else None
        expected_success = int(parsed == target)
    else:
        target = str(trial.get("keyword", ""))
        if target.isdigit():
            match = re.search(rf"(?<!\d)\d{{{len(target)}}}(?!\d)", output)
            parsed = match.group(0) if match else None
            expected_success = int(parsed == target)
        else:
            parsed = output.strip() or None
            expected_success = int(target in output)
    expected_invalid = int(parsed is None)
    parsed_key = "parsed_target" if method == "trap" else "parsed_output"
    if parsed_key in trial:
        verdict.protocol(
            trial.get(parsed_key) == parsed,
            f"{where}: saved parsed output does not match raw output",
        )
    verdict.protocol(
        trial.get("success") == expected_success,
        f"{where}: success flag does not match parsed output",
    )
    verdict.protocol(
        trial.get("invalid") == expected_invalid,
        f"{where}: invalid flag does not match parsed output",
    )


def validate_run(method: str, run: Mapping[str, Any], where: str, verdict: Verdict) -> None:
    expected_trials = 3 if method == "trap" else 6
    trials = run.get("trials") or []
    verdict.protocol(len(trials) == expected_trials, f"{where}: wrong trial count")
    for trial_index, trial in enumerate(trials, start=1):
        validate_trial_flags(method, trial, f"{where} trial {trial_index}", verdict)
    if trials:
        hit_rate = sum(trial.get("success") == 1 for trial in trials) / len(trials)
        invalid_rate = sum(trial.get("invalid") == 1 for trial in trials) / len(trials)
        verdict.protocol(
            same_number(run.get("score"), hit_rate),
            f"{where}: score does not equal trial hit rate",
        )
        verdict.protocol(
            same_number((run.get("metrics") or {}).get("invalid_rate"), invalid_rate),
            f"{where}: invalid_rate does not equal trial flags",
        )


def validate_manifest_and_records(batch: Path, method: str, verdict: Verdict) -> None:
    manifest = read_json(batch / "fingerprint_config.json", verdict)
    if not manifest:
        return
    expected_artifacts = 3 if method == "trap" else 1
    records = sorted(batch.glob("[0-9][0-9][0-9].json"))
    verdict.protocol(manifest.get("schema_version") == 2, f"{method}: schema_version is not 2")
    verdict.protocol(manifest.get("fingerprint_method") == method, f"{method}: manifest method mismatch")
    verdict.protocol(
        manifest.get("artifact_count") == expected_artifacts,
        f"{method}: wrong artifact_count",
    )
    verdict.protocol(
        manifest.get("expected_artifact_count") == expected_artifacts,
        f"{method}: wrong expected_artifact_count",
    )
    verdict.protocol(len(records) == expected_artifacts, f"{method}: numbered artifact count mismatch")
    verdict.protocol(
        manifest.get("status") in {"completed", "completed_with_warnings"},
        f"{method}: generation did not finish ({manifest.get('status')!r})",
    )
    verdict.quality(
        manifest.get("status") == "completed" and not manifest.get("warnings"),
        f"{method}: generation contains quality warnings: {manifest.get('warnings', [])}",
    )

    payloads = [read_json(path, verdict) for path in records]
    if method == "trap":
        item_seeds = []
        successes = []
        for index, record in enumerate(payloads, start=1):
            payload = record.get("payload") or {}
            check = payload.get("source_self_test") or {}
            optimization = payload.get("optimization") or {}
            where = f"trap artifact {index:03d} source self-test"
            verdict.protocol(payload.get("kind") == "trap", f"{where}: wrong payload kind")
            verdict.protocol(
                optimization.get("num_steps") == 1500,
                f"{where}: optimization step count changed",
            )
            verdict.protocol(
                finite_float(optimization.get("final_loss")) is not None,
                f"{where}: final loss is absent or non-finite",
            )
            item_seeds.append(optimization.get("item_seed"))
            trial = {
                "target": payload.get("target"),
                "output": check.get("output"),
                "parsed_target": check.get("parsed_target"),
                "success": check.get("success"),
                "invalid": check.get("invalid"),
            }
            validate_trial_flags("trap", trial, where, verdict)
            successes.append(check.get("success"))
        verdict.protocol(len(set(item_seeds)) == len(item_seeds), "TRAP item seeds are not unique")
        verdict.quality(all(value == 1 for value in successes), "TRAP has source-invalid fingerprints")
        verdict.observations["trap_source_self_test_hit_rate"] = (
            sum(value == 1 for value in successes) / len(successes) if successes else None
        )
    elif payloads:
        payload = payloads[0].get("payload") or {}
        verdict.protocol(payload.get("kind") == "plugae", "PlugAE payload kind mismatch")
        queries = payload.get("queries") or []
        targets = payload.get("targets") or []
        keywords = payload.get("keywords") or []
        verdict.protocol(len(queries) == len(targets) == len(keywords) == 3, "PlugAE query payload is not the requested 3-pair sample")
        text_fields_valid = all(
            isinstance(value, str) for values in (queries, targets, keywords) for value in values
        )
        verdict.protocol(text_fields_valid, "PlugAE query payload contains non-text values")
        if text_fields_valid:
            digest = hashlib.sha256(
                "\n".join("\0".join(values) for values in zip(queries, targets, keywords)).encode()
            ).hexdigest()
            verdict.protocol(digest == payload.get("query_sha256"), "PlugAE query hash is invalid")
        optimization = payload.get("optimization") or {}
        verdict.protocol(optimization.get("epochs") == 30, "PlugAE epoch count changed")
        verdict.protocol(optimization.get("learning_rate") == 0.1, "PlugAE learning rate changed")
        verdict.protocol(
            optimization.get("templates") == ["alpaca", "zero_shot"],
            "PlugAE did not train with both validation templates",
        )
        diagnostics = payload.get("training_diagnostics") or {}
        checkpoints = diagnostics.get("checkpoints") or []
        verdict.protocol(
            [item.get("epoch") for item in checkpoints] == [0, 5, 10, 15, 20, 25, 30],
            "PlugAE diagnostic checkpoints are incomplete",
        )
        if checkpoints:
            baseline, final = checkpoints[0], checkpoints[-1]
            for checkpoint in checkpoints:
                trials = checkpoint.get("trials") or []
                verdict.protocol(len(trials) == 6, "PlugAE source diagnostic must contain 3 queries x 2 templates")
                verdict.protocol(
                    {trial.get("template_id") for trial in trials} == {"alpaca", "zero_shot"},
                    "PlugAE diagnostic omitted a prompt template",
                )
                for trial_index, trial in enumerate(trials, start=1):
                    validate_trial_flags(
                        "plugae",
                        trial,
                        f"PlugAE epoch {checkpoint.get('epoch')} trial {trial_index}",
                        verdict,
                    )
            baseline_nll = baseline.get("mean_target_nll")
            final_nll = final.get("mean_target_nll")
            baseline_rate = (baseline.get("metrics") or {}).get("transfer_response_rate")
            final_rate = (final.get("metrics") or {}).get("transfer_response_rate")
            verdict.quality(
                finite_float(baseline_nll) is not None
                and finite_float(final_nll) is not None
                and float(final_nll) < float(baseline_nll),
                "PlugAE target NLL did not improve",
            )
            verdict.quality(
                finite_float(final_rate) is not None and float(final_rate) > 0,
                "PlugAE final source self-test has zero hits",
            )
            verdict.quality(
                finite_float(baseline_rate) is not None
                and finite_float(final_rate) is not None
                and float(final_rate) > float(baseline_rate),
                "PlugAE did not beat its random initialization",
            )
            verdict.observations["plugae_initial_source_trr"] = baseline_rate
            verdict.observations["plugae_final_source_trr"] = final_rate


def validate_evaluations(batch: Path, method: str, verdict: Verdict) -> None:
    deployment = read_json(batch / "deployment_robustness.json", verdict)
    specificity = read_json(batch / "model_specificity.json", verdict)
    for name, report in (("deployment", deployment), ("specificity", specificity)):
        if report:
            verdict.protocol(report.get("status") == "completed", f"{method} {name} status is {report.get('status')!r}")
            verdict.protocol(report.get("fingerprint_method") == method, f"{method} {name} report method mismatch")

    if deployment:
        conditions = deployment.get("conditions") or []
        summary = deployment.get("summary") or []
        verdict.protocol(len(conditions) == 10, f"{method}: deployment did not execute 10 stochastic seeds")
        verdict.protocol(
            len(summary) == 1 and summary[0].get("num_runs") == 10,
            f"{method}: deployment seed aggregation is incorrect",
        )
        for run_index, run in enumerate(conditions, start=1):
            validate_run(method, run, f"{method} deployment run {run_index}", verdict)
            if method == "trap":
                verdict.protocol(
                    (run.get("metadata") or {}).get("input_mode") == "source_rendered",
                    f"TRAP deployment run {run_index} did not use source_rendered",
                )
            else:
                verdict.protocol(
                    (run.get("metadata") or {}).get("embedding_transferred") is True,
                    f"PlugAE deployment run {run_index} did not inject the embedding",
                )

    if specificity:
        verdict.protocol(specificity.get("failure_count") == 0, f"{method}: specificity contains model failures")
        verdict.protocol(specificity.get("requested_seeds") == EXPECTED_SEEDS, f"{method}: requested greedy seeds changed")
        verdict.protocol(specificity.get("effective_seeds") == [0], f"{method}: greedy seeds were not collapsed")
        summary = specificity.get("summary") or {}
        verdict.protocol(summary.get("roc_auc_definition") == "source_vs_negative_models_only", f"{method}: AUC still mixes derivative models")
        verdict.protocol(summary.get("complete_panel") is True, f"{method}: specificity panel is incomplete")
        threshold = specificity.get("threshold") or {}
        verdict.protocol(
            threshold.get("status") in {"held_out", "degenerate_all_equal"},
            f"{method}: threshold does not use the explicit calibration/held-out split",
        )
        rows = specificity.get("models") or []
        roles = {row.get("evaluation_model"): row.get("role") for row in rows}
        verdict.protocol(roles == EXPECTED_MODELS, f"{method}: specificity model roles are wrong: {roles}")
        for row in rows:
            name = row.get("evaluation_model", "unknown")
            verdict.protocol(row.get("status") == "completed", f"{method}: model {name} failed")
            runs = row.get("runs") or []
            verdict.protocol(len(runs) == 1, f"{method}: greedy model {name} ran more than once")
            for run in runs:
                validate_run(method, run, f"{method} specificity {name}", verdict)
                if method == "trap":
                    verdict.protocol(
                        (run.get("metadata") or {}).get("input_mode") == "model_rendered",
                        f"TRAP specificity model {name} did not use model_rendered",
                    )
                elif name in {"Phi-3-Mini-4K-Base", "Phi-3-Mini-4K-Instruct"}:
                    verdict.protocol(
                        (run.get("metadata") or {}).get("embedding_transferred") is True,
                        f"PlugAE embedding was not transferred to {name}",
                    )
                else:
                    verdict.protocol(
                        (run.get("metadata") or {}).get("embedding_transferred") is False,
                        f"PlugAE embedding leaked into negative model {name}",
                    )
            if runs:
                verdict.protocol(
                    all(finite_float(run.get("score")) is not None for run in runs)
                    and same_number(
                        row.get("score"),
                        sum(float(run["score"]) for run in runs) / len(runs),
                    ),
                    f"{method}: aggregated model score is wrong for {name}",
                )
        source = next((row for row in rows if row.get("role") == "original"), None)
        if source:
            verdict.quality(
                finite_float(source.get("score")) is not None
                and float(source["score"]) > 0,
                f"{method}: source score is zero during formal specificity evaluation",
            )
            verdict.observations[f"{method}_specificity_source_score"] = source.get("score")
            diagnostic_key = (
                "trap_source_self_test_hit_rate"
                if method == "trap"
                else "plugae_final_source_trr"
            )
            diagnostic_score = verdict.observations.get(diagnostic_key)
            verdict.protocol(
                same_number(source.get("score"), diagnostic_score),
                f"{method}: formal source score differs from generation self-test",
            )
        derivative = next((row for row in rows if row.get("role") == "derivative"), None)
        if derivative:
            verdict.observations[f"{method}_derivative_score"] = derivative.get("score")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--results-root", type=Path)
    mode.add_argument("--batch", type=Path)
    parser.add_argument("--method", choices=("trap", "plugae"))
    parser.add_argument(
        "--config-root",
        type=Path,
        default=Path("config/v100/trap_plugae_smoke"),
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    verdict = Verdict()
    validate_configs(args.config_root, verdict)
    batches = {}
    if args.batch is not None:
        if args.method is None:
            parser.error("--method is required with --batch")
        batches[args.method] = str(args.batch)
        validate_manifest_and_records(args.batch, args.method, verdict)
    else:
        if args.method is not None:
            parser.error("--method is only valid with --batch")
        for method in ("trap", "plugae"):
            batch = discover_batch(args.results_root, method, verdict)
            if batch is not None:
                batches[method] = str(batch)
                protocol_count = len(verdict.protocol_errors)
                quality_count = len(verdict.quality_errors)
                validate_manifest_and_records(batch, method, verdict)
                if (
                    len(verdict.protocol_errors) == protocol_count
                    and len(verdict.quality_errors) == quality_count
                ):
                    validate_evaluations(batch, method, verdict)

    status = (
        "FAIL_PROTOCOL"
        if verdict.protocol_errors
        else "FAIL_QUALITY"
        if verdict.quality_errors
        else "PASS"
    )
    report = {
        "status": status,
        "batches": batches,
        "protocol_errors": verdict.protocol_errors,
        "quality_errors": verdict.quality_errors,
        "observations": verdict.observations,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
