#!/usr/bin/env python3
"""Create a compact TSV inventory from completed paper-scale result folders."""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/v100_fingerprint_paper")
    rows = []
    for config_path in sorted(root.glob("**/fingerprint_config.json")):
        batch = config_path.parent
        config = json.loads(config_path.read_text(encoding="utf-8"))
        row = {
            "experiment_id": batch.name,
            "method": config["fingerprint_method"],
            "source_model": config["source_model"]["model_name"],
            "seed": config["optimization_seed"],
            "artifacts": config["artifact_count"],
            "generation_status": config.get("status", "completed"),
        }
        for report in (
            "model_modification_robustness",
            "deployment_robustness",
            "model_specificity",
            "prompt_stealthiness",
        ):
            path = batch / f"{report}.json"
            if not path.exists():
                row[report] = "missing"
                continue
            value = json.loads(path.read_text(encoding="utf-8"))
            row[report] = value.get("status", "unknown")
            if report == "model_specificity" and value.get("status") in {
                "completed", "partial_failed", "running"
            }:
                summary = value.get("summary", {})
                row["roc_auc"] = summary.get("roc_auc")
                row["source_margin"] = summary.get("source_score_margin_over_best_negative")
                row["in_sample_descriptive_fpr"] = summary.get(
                    "in_sample_descriptive_fpr"
                )
                row["held_out_fpr"] = summary.get("held_out_fpr")
        rows.append(row)

    seeds_by_method_source = defaultdict(set)
    for row in rows:
        seeds_by_method_source[(row["method"], row["source_model"])].add(row["seed"])
    for row in rows:
        row["optimization_seed_count"] = len(
            seeds_by_method_source[(row["method"], row["source_model"])]
        )

    output = root / "paper_results_inventory.tsv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "experiment_id", "method", "source_model", "seed", "optimization_seed_count",
        "generation_status", "artifacts",
        "model_modification_robustness", "deployment_robustness",
        "model_specificity", "prompt_stealthiness", "roc_auc", "source_margin",
        "in_sample_descriptive_fpr", "held_out_fpr",
    ]
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")
    single_seed = sorted(
        f"{method}/{source}"
        for (method, source), seeds in seeds_by_method_source.items()
        if len(seeds) < 2
    )
    if single_seed:
        print(
            "WARNING: seed-level stability is unavailable for: "
            + ", ".join(single_seed)
        )
    return 1 if len(rows) != 15 else 0


if __name__ == "__main__":
    raise SystemExit(main())
