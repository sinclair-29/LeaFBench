#!/usr/bin/env python3
"""Summarize one TRAP/PlugAE run without loading any model or GPU.

The script is intentionally read-only with respect to experiment batches.  All
generated reports are written to a separate analysis directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


REPORTS = ("deployment_robustness", "model_specificity")
MASTER_EVENT = re.compile(r"\]\s+(START|DONE|FAIL)\s+([^:\s]+)(?::\s*(.*))?")


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def safe_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, None
    try:
        return read_json(path), None
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return None, f"{type(error).__name__}: {error}"


def finite_number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(float(value)) else None


def scalar_metrics(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}{key}"
        if isinstance(item, Mapping):
            result.update(scalar_metrics(item, f"{name}."))
        elif item is None or isinstance(item, (str, bool)):
            result[name] = item
        elif finite_number(item) is not None:
            result[name] = item
    return result


def write_csv(path: Path, rows: list[dict[str, Any]], preferred: Iterable[str]) -> None:
    fields = list(preferred)
    extras = sorted({key for row in rows for key in row} - set(fields))
    fields.extend(extras)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_master(logs_root: Path | None) -> tuple[dict[str, list[dict[str, str]]], list[str]]:
    events: dict[str, list[dict[str, str]]] = {}
    errors: list[str] = []
    if logs_root is None:
        return events, errors
    master = logs_root / "master.log"
    if not master.is_file():
        errors.append(f"master log missing: {master}")
        return events, errors
    try:
        lines = master.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as error:
        errors.append(f"cannot read master log: {error}")
        return events, errors
    for line in lines:
        match = MASTER_EVENT.search(line)
        if not match:
            continue
        event, job_id, message = match.groups()
        events.setdefault(job_id, []).append(
            {"event": event, "message": message or "", "line": line}
        )
    return events, errors


def log_tail(logs_root: Path | None, job_id: str, limit: int = 30) -> list[str]:
    if logs_root is None:
        return []
    path = logs_root / f"{job_id}.log"
    if not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    except OSError:
        return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-limit:]


def batch_variant(path: Path) -> int:
    match = re.search(r"_([a-z]+)$", path.name)
    if not match:
        return 0
    value = 0
    for character in match.group(1):
        value = value * 26 + ord(character) - ord("a") + 1
    return value


def choose_batch(paths: list[Path]) -> Path:
    def rank(path: Path) -> tuple[int, int, int]:
        statuses = []
        for report in REPORTS:
            value, _ = safe_json(path / f"{report}.json")
            statuses.append(value.get("status") if value else None)
        return (
            sum(status == "completed" for status in statuses),
            sum(status is not None for status in statuses),
            batch_variant(path),
        )

    return max(paths, key=rank)


def expected_and_actual(config: Mapping[str, Any], batch: Path) -> tuple[int | None, int, int | None]:
    method = config.get("fingerprint_method")
    parameters = config.get("fingerprint_parameters") or {}
    artifact_count = config.get("artifact_count")
    expected: int | None
    query_count: int | None = None
    if method == "trap":
        expected = parameters.get("goal_count", parameters.get("n_goals"))
    elif method == "plugae":
        # PlugAE stores one universal embedding artifact trained on N query-target pairs.
        expected = 1
        query_count = parameters.get("num_queries")
        records = sorted(batch.glob("[0-9][0-9][0-9].json"))
        if records:
            record, _ = safe_json(records[0])
            queries = ((record or {}).get("payload") or {}).get("queries")
            if isinstance(queries, list):
                query_count = len(queries)
    else:
        expected = artifact_count if isinstance(artifact_count, int) else None
    actual = len(list(batch.glob("[0-9][0-9][0-9].json")))
    return expected, actual, query_count


def error_message(value: Mapping[str, Any] | None, parse_error: str | None) -> str:
    if parse_error:
        return parse_error
    if not value:
        return ""
    error = value.get("error")
    if isinstance(error, Mapping):
        kind = error.get("type", "Error")
        message = error.get("message", "")
        return f"{kind}: {message}".strip()
    return ""


def status_for(row: Mapping[str, Any], events: list[dict[str, str]]) -> str:
    report_statuses = [row[f"{name}_status"] for name in REPORTS]
    if all(status == "completed" for status in report_statuses):
        return "completed"
    if any(status in {"failed", "invalid"} for status in report_statuses):
        return "evaluation_failed"
    if row.get("batch"):
        if row.get("actual_artifacts") != row.get("expected_artifacts"):
            return "artifact_incomplete"
        return "generated_evaluation_incomplete"
    if any(event["event"] == "FAIL" for event in events):
        return "generation_failed"
    if any(event["event"] == "START" for event in events):
        return "started_without_saved_batch"
    return "not_started_or_unknown"


def analyze(results_root: Path, logs_root: Path | None) -> dict[str, Any]:
    master_events, discovery_warnings = parse_master(logs_root)
    batch_by_job: dict[str, list[Path]] = {}
    for config_path in sorted(results_root.glob("**/fingerprint_config.json")):
        batch = config_path.parent
        relative = batch.relative_to(results_root)
        job_id = relative.parts[0] if len(relative.parts) > 1 else "unscoped"
        batch_by_job.setdefault(job_id, []).append(batch)

    job_ids = set(batch_by_job) | set(master_events)
    if logs_root is not None and logs_root.is_dir():
        job_ids.update(path.stem for path in logs_root.glob("gpu*.log"))

    jobs: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    deployment_rows: list[dict[str, Any]] = []
    failed_tails: dict[str, list[str]] = {}

    for job_id in sorted(job_ids):
        events = master_events.get(job_id, [])
        paths = batch_by_job.get(job_id, [])
        row: dict[str, Any] = {
            "job_id": job_id,
            "method": "",
            "source_model": "",
            "batch": "",
            "expected_artifacts": None,
            "actual_artifacts": 0,
            "query_target_pairs": None,
            "deployment_robustness_status": "missing",
            "model_specificity_status": "missing",
            "roc_auc": None,
            "source_score_rank": None,
            "source_margin": None,
            "overall_model_fpr": None,
            "error": "",
            "master_last_event": events[-1]["event"] if events else "",
        }
        if paths:
            batch = choose_batch(paths)
            config, config_error = safe_json(batch / "fingerprint_config.json")
            row["batch"] = str(batch.relative_to(results_root))
            if config is None:
                row["error"] = config_error or "invalid fingerprint_config.json"
            else:
                row["method"] = config.get("fingerprint_method", "")
                row["source_model"] = (config.get("source_model") or {}).get("model_name", "")
                expected, actual, query_count = expected_and_actual(config, batch)
                row["expected_artifacts"] = expected
                row["actual_artifacts"] = actual
                row["query_target_pairs"] = query_count

                errors = []
                for report in REPORTS:
                    value, parse_error = safe_json(batch / f"{report}.json")
                    status = "invalid" if parse_error else (value or {}).get("status", "missing")
                    row[f"{report}_status"] = status
                    message = error_message(value, parse_error)
                    if message:
                        errors.append(f"{report}: {message}")

                    if report == "deployment_robustness" and value and status == "completed":
                        for condition in value.get("summary", []):
                            if isinstance(condition, Mapping):
                                deployment_rows.append(
                                    {
                                        "job_id": job_id,
                                        "method": row["method"],
                                        "source_model": row["source_model"],
                                        **scalar_metrics(condition),
                                    }
                                )

                    if report == "model_specificity" and value and status == "completed":
                        summary = value.get("summary") or {}
                        row["roc_auc"] = summary.get("roc_auc")
                        row["source_score_rank"] = summary.get("source_score_rank")
                        row["source_margin"] = summary.get(
                            "source_score_margin_over_best_negative"
                        )
                        row["overall_model_fpr"] = summary.get("overall_model_fpr")
                        for model in value.get("models", []):
                            if not isinstance(model, Mapping):
                                continue
                            model_rows.append(
                                {
                                    "job_id": job_id,
                                    "method": row["method"],
                                    "source_model": row["source_model"],
                                    "evaluation_model": model.get("evaluation_model"),
                                    "role": model.get("role"),
                                    "label": model.get("label"),
                                    "negative_type": model.get("negative_type"),
                                    "score": model.get("score"),
                                    "predicted_positive": model.get("predicted_positive"),
                                    **scalar_metrics(model.get("metrics") or {}, "metric."),
                                }
                            )
                row["error"] = "; ".join(filter(None, [row["error"], *errors]))

        row["status"] = status_for(row, events)
        if row["status"] != "completed":
            failed_tails[job_id] = log_tail(logs_root, job_id)
        jobs.append(row)

    warnings = list(discovery_warnings)
    for row in jobs:
        if row["status"] == "started_without_saved_batch":
            warnings.append(
                f"{row['job_id']}: worker started but no saved fingerprint batch exists; "
                "inspect its log before treating it as still running."
            )
    completed_sources: dict[str, set[str]] = {}
    for row in jobs:
        if row["status"] == "completed":
            completed_sources.setdefault(str(row["method"]), set()).add(str(row["source_model"]))
    methods = sorted(completed_sources)
    comparable_sources = (
        sorted(set.intersection(*(completed_sources[method] for method in methods)))
        if len(methods) >= 2
        else []
    )

    return {
        "results_root": str(results_root),
        "logs_root": str(logs_root) if logs_root else None,
        "counts_by_status": dict(Counter(row["status"] for row in jobs)),
        "completed_sources_by_method": {
            method: sorted(sources) for method, sources in completed_sources.items()
        },
        "paired_comparable_sources": comparable_sources,
        "warnings": warnings,
        "jobs": jobs,
        "model_scores": model_rows,
        "deployment_conditions": deployment_rows,
        "failed_log_tails": failed_tails,
    }


def markdown_report(bundle: Mapping[str, Any]) -> str:
    jobs = bundle["jobs"]
    lines = [
        "# TRAP / PlugAE experiment analysis inventory",
        "",
        f"- Results: `{bundle['results_root']}`",
        f"- Logs: `{bundle['logs_root'] or 'not supplied'}`",
        f"- Jobs discovered: {len(jobs)}",
        "- Status counts: "
        + ", ".join(f"{key}={value}" for key, value in bundle["counts_by_status"].items()),
        "- Paired, directly comparable completed sources: "
        + (", ".join(bundle["paired_comparable_sources"]) or "none yet"),
        "",
        "## Job summary",
        "",
        "| job | method | source | artifacts | deployment | specificity | AUC | margin | overall FPR | final status |",
        "|---|---|---|---:|---|---|---:|---:|---:|---|",
    ]
    for row in jobs:
        artifacts = f"{row['actual_artifacts']}/{row['expected_artifacts']}"
        values = [
            row["job_id"], row["method"] or "?", row["source_model"] or "?", artifacts,
            row["deployment_robustness_status"], row["model_specificity_status"],
            row["roc_auc"], row["source_margin"], row["overall_model_fpr"], row["status"],
        ]
        lines.append("| " + " | ".join("" if value is None else str(value) for value in values) + " |")

    errors = [row for row in jobs if row.get("error")]
    if errors:
        lines.extend(["", "## Recorded evaluation errors", ""])
        lines.extend(f"- `{row['job_id']}`: {row['error']}" for row in errors)
    if bundle["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in bundle["warnings"])
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Compare TRAP and PlugAE only on source models listed as paired and completed.",
            "- A generated batch is not a completed experiment until both enabled evaluation reports are completed.",
            "- The reported Youden operating point is calibrated and measured on the same suspect-model panel; treat its FPR as descriptive, while ROC AUC is the threshold-free primary statistic.",
            "- Multiple seeds with greedy decoding are deterministic repeats, not independent stochastic trials.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path, help="One run-ID result directory")
    parser.add_argument("--logs-root", type=Path, help="Matching run-ID log directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Default: RESULTS_ROOT/analysis",
    )
    args = parser.parse_args()

    results_root = args.results_root.expanduser().resolve()
    logs_root = args.logs_root.expanduser().resolve() if args.logs_root else None
    output_dir = (args.output_dir or (results_root / "analysis")).expanduser().resolve()
    if not results_root.is_dir():
        parser.error(f"results root does not exist: {results_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = analyze(results_root, logs_root)
    summary_path = output_dir / "summary.csv"
    models_path = output_dir / "model_scores.csv"
    deployment_path = output_dir / "deployment_conditions.csv"
    json_path = output_dir / "analysis_bundle.json"
    markdown_path = output_dir / "analysis_report.md"

    write_csv(
        summary_path,
        bundle["jobs"],
        (
            "job_id", "method", "source_model", "status", "actual_artifacts",
            "expected_artifacts", "query_target_pairs", "deployment_robustness_status",
            "model_specificity_status", "roc_auc", "source_score_rank", "source_margin",
            "overall_model_fpr", "error", "batch", "master_last_event",
        ),
    )
    write_csv(
        models_path,
        bundle["model_scores"],
        ("job_id", "method", "source_model", "evaluation_model", "role", "label", "negative_type", "score", "predicted_positive"),
    )
    write_csv(
        deployment_path,
        bundle["deployment_conditions"],
        ("job_id", "method", "source_model", "component", "condition_id", "num_runs"),
    )
    json_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown_report(bundle), encoding="utf-8")

    archive = output_dir / "analysis_bundle.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for path in (summary_path, models_path, deployment_path, json_path, markdown_path):
            target.write(path, arcname=path.name)

    print(markdown_report(bundle))
    print(f"\nAnalysis files: {output_dir}")
    print(f"Upload this one file for review: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
