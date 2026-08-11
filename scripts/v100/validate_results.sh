#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${LEAFBENCH_ROOT:-/raid/chj/fingerprint/LeaFBench}"
RESULTS_ROOT="${1:-${RESULTS_ROOT:-${PROJECT_ROOT}/results/v100_validation}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" - "${RESULTS_ROOT}" <<'PY'
import json
import math
import sys
from pathlib import Path

import evaluation

root = Path(sys.argv[1])
methods = ("trap", "plugae", "zeroprint", "reef")
reports = (
    "model_modification_robustness",
    "deployment_robustness",
    "model_specificity",
    "prompt_stealthiness",
)
expected_not_applicable = {("reef", "prompt_stealthiness")}
errors = []

for method in methods:
    batches = list((root / method).glob("exp_*"))
    batches = [p for p in batches if (p / "fingerprint_config.json").is_file()]
    if not batches:
        errors.append(f"{method}: no fingerprint batch")
        continue
    # Evaluation may clone a batch when its config changes. Select the latest
    # batch containing result files rather than assuming the '_a' suffix.
    evaluated = [p for p in batches if any((p / f"{r}.json").is_file() for r in reports)]
    candidates = evaluated or batches
    batch = max(
        candidates,
        key=lambda path: evaluation.variant_to_number(
            evaluation.experiment_variant(path)[1]
        ),
    )
    with (batch / "fingerprint_config.json").open(encoding="utf-8") as stream:
        fingerprint = json.load(stream)
    artifacts = sorted(batch.glob("[0-9][0-9][0-9].json"))
    if len(artifacts) != fingerprint.get("artifact_count"):
        errors.append(f"{method}: artifact count mismatch in {batch.name}")

    for report in reports:
        path = batch / f"{report}.json"
        if not path.is_file():
            errors.append(f"{method}: missing {report}.json")
            continue
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
        expected = "not_applicable" if (method, report) in expected_not_applicable else "completed"
        if value.get("status") != expected:
            message = value.get("error", {}).get("message", "no error message")
            errors.append(f"{method}/{report}: {value.get('status')} ({message})")
        if report == "model_specificity" and value.get("status") == "completed":
            threshold = value.get("threshold", {}).get("value")
            if not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
                errors.append(f"{method}: specificity threshold is not finite")

    print(f"{method}: checked {batch}")

if errors:
    print("\nValidation failed:", file=sys.stderr)
    for error in errors:
        print(f"- {error}", file=sys.stderr)
    raise SystemExit(1)

print("All four fingerprint batches and evaluation reports passed structural validation.")
PY
