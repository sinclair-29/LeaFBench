from __future__ import annotations

import importlib.util
import json
import os
import socket
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/v100/trap_plugae_20/summarize_results.py"
)
SPEC = importlib.util.spec_from_file_location("summarize_results", SCRIPT)
assert SPEC and SPEC.loader
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class TrapPlugAESummaryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.results = root / "results" / "run"
        self.logs = root / "logs" / "run"
        self.results.mkdir(parents=True)
        self.logs.mkdir(parents=True)

    def tearDown(self):
        self.temporary.cleanup()

    def test_completed_trap_and_failed_worker_are_both_reported(self):
        job = "gpu01_trap_llama2_7b"
        batch = self.results / job / "exp_llama_trap_seed_042_a"
        write_json(
            batch / "fingerprint_config.json",
            {
                "fingerprint_method": "trap",
                "source_model": {"model_name": "Llama-2-7B"},
                "artifact_count": 20,
                "fingerprint_parameters": {"goal_count": 20},
            },
        )
        for index in range(1, 21):
            write_json(batch / f"{index:03d}.json", {"payload": {"kind": "trap"}})
        write_json(
            batch / "deployment_robustness.json",
            {
                "status": "completed",
                "summary": [
                    {
                        "component": "temperature",
                        "condition_id": "temperature_0.7",
                        "num_runs": 10,
                        "score": 0.8,
                        "metrics": {"target_hit_rate": 0.8},
                    }
                ],
            },
        )
        write_json(
            batch / "model_specificity.json",
            {
                "status": "completed",
                "summary": {
                    "roc_auc": 1.0,
                    "source_score_rank": 1,
                    "source_score_margin_over_best_negative": 0.7,
                    "overall_model_fpr": 0.0,
                },
                "models": [
                    {
                        "evaluation_model": "Llama-2-7B",
                        "role": "original",
                        "label": 1,
                        "score": 0.9,
                        "predicted_positive": 1,
                        "metrics": {"target_hit_rate": 0.9},
                    }
                ],
            },
        )
        failed = "gpu00_trap_llama_7b"
        (self.logs / "master.log").write_text(
            f"[time] START {job}\n[time] DONE {job}\n[time] START {failed}\n",
            encoding="utf-8",
        )
        (self.logs / f"{failed}.log").write_text("loading\nValueError: no weights\n", encoding="utf-8")

        bundle = SUMMARY.analyze(self.results, self.logs)
        by_job = {row["job_id"]: row for row in bundle["jobs"]}
        self.assertEqual(by_job[job]["status"], "completed")
        self.assertEqual(by_job[job]["actual_artifacts"], 20)
        self.assertEqual(by_job[job]["roc_auc"], 1.0)
        self.assertEqual(by_job[failed]["status"], "interrupted")
        self.assertIn("ValueError: no weights", bundle["failed_log_tails"][failed])
        self.assertEqual(len(bundle["deployment_conditions"]), 1)
        self.assertEqual(len(bundle["model_scores"]), 1)

    def test_plugae_counts_one_embedding_and_twenty_query_pairs(self):
        job = "gpu14_plugae_gemma2_2b"
        batch = self.results / job / "exp_gemma_plugae_seed_042_a"
        write_json(
            batch / "fingerprint_config.json",
            {
                "fingerprint_method": "plugae",
                "source_model": {"model_name": "Gemma-2-2B"},
                "artifact_count": 1,
                "fingerprint_parameters": {"num_queries": 20},
            },
        )
        write_json(
            batch / "001.json",
            {"payload": {"kind": "plugae", "queries": [f"q{i}" for i in range(20)]}},
        )
        for report in SUMMARY.REPORTS:
            payload = {"status": "completed"}
            if report == "model_specificity":
                payload.update(
                    {
                        "summary": {
                            "roc_auc": 0.75,
                            "source_score_rank": 1,
                            "source_score_margin_over_best_negative": 0.2,
                            "overall_model_fpr": 0.25,
                        },
                        "models": [],
                    }
                )
            else:
                payload["summary"] = []
            write_json(batch / f"{report}.json", payload)

        bundle = SUMMARY.analyze(self.results, self.logs)
        row = bundle["jobs"][0]
        self.assertEqual(row["expected_artifacts"], 1)
        self.assertEqual(row["actual_artifacts"], 1)
        self.assertEqual(row["query_target_pairs"], 20)
        self.assertEqual(row["status"], "completed")

    def test_fresh_live_heartbeat_is_running_and_failed_phase_is_terminal(self):
        running = "gpu00_trap_running"
        failed = "gpu01_trap_failed"
        write_json(
            self.logs / "status" / f"{running}.json",
            {
                "job_id": running,
                "state": "running",
                "phase": "generate",
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "exit_code": None,
            },
        )
        write_json(
            self.logs / "status" / f"{failed}.json",
            {
                "job_id": failed,
                "state": "failed",
                "phase": "generate",
                "pid": 999999,
                "host": socket.gethostname(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "exit_code": 1,
            },
        )
        bundle = SUMMARY.analyze(self.results, self.logs)
        by_job = {row["job_id"]: row for row in bundle["jobs"]}
        self.assertEqual(by_job[running]["status"], "running")
        self.assertEqual(by_job[failed]["status"], "generation_failed")


if __name__ == "__main__":
    unittest.main()
