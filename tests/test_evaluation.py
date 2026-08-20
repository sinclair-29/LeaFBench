from __future__ import annotations

import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
from pathlib import Path

import evaluation


class DummyMethod:
    evaluation_capabilities = {
        "deployment_robustness": {
            "system_prompts": True,
            "sampling": False,
        }
    }

    def supports_evaluation(self, name, component=None):
        capability = self.evaluation_capabilities.get(name, False)
        if component is None:
            return bool(capability)
        return bool(capability.get(component, False))


class EvaluationUtilitiesTest(unittest.TestCase):
    def test_generation_seed_controls_python_rng_used_by_fingerprint_preparation(self):
        import random
        import numpy as np

        evaluation.seed_fingerprint_generation(17)
        first = (random.random(), np.random.random())
        evaluation.seed_fingerprint_generation(17)
        second = (random.random(), np.random.random())
        self.assertEqual(first, second)

    def test_variant_letters_round_trip(self):
        for number, variant in ((1, "a"), (26, "z"), (27, "aa"), (53, "ba")):
            self.assertEqual(evaluation.number_to_variant(number), variant)
            self.assertEqual(evaluation.variant_to_number(variant), number)

    def test_youden_threshold_is_finite_and_separates_models(self):
        threshold = evaluation.youden_threshold(
            scores=[0.9, 0.8, 0.3, 0.2],
            labels=[1, 1, 0, 0],
        )
        self.assertEqual(threshold, 0.8)
        self.assertTrue(threshold < float("inf"))

    def test_youden_requires_positive_and_negative_models(self):
        with self.assertRaisesRegex(ValueError, "both positive and negative"):
            evaluation.youden_threshold([0.7, 0.8], [1, 1])

    def test_greedy_decoding_collapses_requested_seeds(self):
        requested, effective = evaluation.effective_decoding_seeds(
            [0, 1, 2], {"do_sample": False}
        )
        self.assertEqual(requested, [0, 1, 2])
        self.assertEqual(effective, [0])

    def test_model_groups_requires_original_to_equal_source(self):
        config = {
            "source_model": {"model_name": "source"},
            "model_groups": {
                "original": ["other"],
                "derivatives": [],
                "negatives": [],
            },
        }
        with self.assertRaisesRegex(ValueError, "must be the source model"):
            evaluation.model_groups(config)

    def test_reef_like_capabilities_skip_sampling_but_keep_system_prompt(self):
        conditions, not_applicable = evaluation.deployment_conditions(
            {
                "generation": {"do_sample": False},
                "system_prompts": [{"id": "default", "prompt": None}],
                "sampling": {
                    "seeds": [0, 1],
                    "temperature_values": [0.7],
                    "temperature_top_p": 1.0,
                },
            },
            DummyMethod(),
        )
        self.assertEqual(len(conditions), 1)
        self.assertEqual(conditions[0]["component"], "system_prompts")
        self.assertEqual(
            not_applicable,
            [{"component": "sampling", "reason": "unsupported_by_method"}],
        )


class EvaluationArtifactTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.results = Path(self.temporary.name)
        self.batch = self.results / "exp_model_trap_seed_004_a"
        self.identity = {
            "fingerprint_method": "trap",
            "optimization_seed": 4,
            "source_model": {
                "model_name": "model",
                "model_path": "checkpoint",
                "model_family": "family",
                "model_type": "instruct",
            },
            "fingerprint_parameters": {
                "fingerprint_method": "trap",
                "seed": 4,
            },
            "benchmark_config": {"path": "benchmark.yaml", "sha256": "abc"},
        }
        self.records = [
            {
                "schema_version": 1,
                "fingerprint_id": f"{self.batch.name}:{index:03d}",
                "item_index": index,
                "source_model": "model",
                "payload": {"kind": "value", "value": f"fingerprint-{index}"},
                "metadata": {},
            }
            for index in range(1, 4)
        ]
        evaluation.write_fingerprint_batch(self.batch, self.records, self.identity)

    def tearDown(self):
        self.temporary.cleanup()

    def test_numbered_batch_is_loaded_without_regeneration(self):
        self.assertEqual(
            [path.name for path in evaluation.artifact_paths(self.batch)],
            ["001.json", "002.json", "003.json"],
        )
        self.assertEqual(evaluation.load_records(self.batch), self.records)
        config = evaluation.read_json(self.batch / "fingerprint_config.json")
        self.assertEqual(config["artifact_count"], 3)
        self.assertEqual(config["config_hash"], evaluation.stable_hash(self.identity))

    def test_non_contiguous_artifacts_fail(self):
        (self.batch / "002.json").unlink()
        with self.assertRaisesRegex(ValueError, "contiguous"):
            evaluation.load_records(self.batch)

    def test_changed_evaluation_creates_and_reuses_one_whole_folder_variant(self):
        evaluation.atomic_write_json(
            self.batch / "model_specificity.json",
            {"status": "completed", "evaluation_config_hash": "config-a"},
        )
        variant_b = evaluation.resolve_evaluation_directory(self.batch, "config-b")
        self.assertEqual(variant_b.name, "exp_model_trap_seed_004_b")
        self.assertEqual(
            sorted(path.name for path in evaluation.artifact_paths(variant_b)),
            ["001.json", "002.json", "003.json"],
        )
        cloned_record = evaluation.read_json(variant_b / "001.json")
        self.assertEqual(
            cloned_record["fingerprint_id"],
            "exp_model_trap_seed_004_b:001",
        )

        evaluation.atomic_write_json(
            variant_b / "model_specificity.json",
            {"status": "completed", "evaluation_config_hash": "config-b"},
        )
        reused = evaluation.resolve_evaluation_directory(self.batch, "config-b")
        self.assertEqual(reused, variant_b)
        self.assertFalse((self.results / "exp_model_trap_seed_004_c").exists())

    def test_failed_result_requires_explicit_retry(self):
        path = self.batch / "deployment_robustness.json"
        evaluation.atomic_write_json(
            path,
            {"status": "failed", "evaluation_config_hash": "same"},
        )
        self.assertFalse(
            evaluation.should_run_result(path, "same", retry_failed=False, overwrite=False)
        )
        self.assertTrue(
            evaluation.should_run_result(path, "same", retry_failed=True, overwrite=False)
        )

    def test_partial_failed_result_requires_explicit_retry(self):
        path = self.batch / "model_specificity.json"
        evaluation.atomic_write_json(
            path,
            {"status": "partial_failed", "evaluation_config_hash": "same"},
        )
        self.assertFalse(
            evaluation.should_run_result(path, "same", retry_failed=False, overwrite=False)
        )
        self.assertTrue(
            evaluation.should_run_result(path, "same", retry_failed=True, overwrite=False)
        )

    def test_schema_v2_incomplete_batch_can_resume_but_cannot_evaluate(self):
        config = evaluation.read_json(self.batch / "fingerprint_config.json")
        config.update(
            {
                "status": "generating",
                "expected_artifact_count": 4,
                "artifact_count": 3,
            }
        )
        evaluation.atomic_write_json(self.batch / "fingerprint_config.json", config)
        self.assertEqual(len(evaluation.load_records(self.batch, allow_incomplete=True)), 3)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            evaluation.load_records(self.batch)

    def test_manifest_preserves_nineteen_checkpoints_for_resume(self):
        batch = self.results / "exp_resume_trap_seed_004_a"
        evaluation.initialize_fingerprint_manifest(batch, self.identity, 20)
        for index in range(1, 20):
            evaluation.atomic_write_json(
                batch / f"{index:03d}.json",
                {
                    "schema_version": 2,
                    "fingerprint_id": f"{batch.name}:{index:03d}",
                    "item_index": index,
                    "source_model": "model",
                    "payload": {"kind": "trap", "target": str(index)},
                    "metadata": {},
                },
            )
            evaluation.update_fingerprint_manifest(
                batch, status="generating", artifact_count=index
            )
        evaluation.update_fingerprint_manifest(
            batch,
            status="generation_failed",
            artifact_count=19,
            error={"type": "InterruptedError", "message": "stopped"},
        )
        manifest = evaluation.initialize_fingerprint_manifest(batch, self.identity, 20)
        self.assertEqual(manifest["artifact_count"], 19)
        records = evaluation.load_records(batch, allow_incomplete=True)
        self.assertEqual(len(records), 19)
        self.assertEqual(records[-1]["item_index"], 19)


class SpecificityStatisticsTest(unittest.TestCase):
    def setUp(self):
        self.source = SimpleNamespace(model_name="source")
        self.models = {
            name: SimpleNamespace(model_name=name)
            for name in ("source", "derivative", "negative-a", "negative-b")
        }
        self.benchmark = SimpleNamespace(get_all_models=lambda: self.models)

    @staticmethod
    def groups(split=False):
        negatives = [
            {
                "model_name": "negative-a",
                "role": "negative",
                "negative_type": "unrelated",
            },
            {
                "model_name": "negative-b",
                "role": "negative",
                "negative_type": "unrelated",
            },
        ]
        if split:
            negatives[0]["threshold_split"] = "calibration"
            negatives[1]["threshold_split"] = "held_out"
        return {
            "original": [{"model_name": "source", "role": "original"}],
            "derivatives": [
                {"model_name": "derivative", "role": "derivative"}
            ],
            "negatives": negatives,
        }

    @staticmethod
    def result(score):
        return {"score": score, "metrics": {"hit": score}, "trials": []}

    def test_auc_uses_source_only_and_keeps_derivative_separate(self):
        scores = {"source": 0.8, "derivative": 0.0, "negative-a": 0.0, "negative-b": 0.0}
        calls = []

        def verify(method, source, model, generation):
            calls.append((model.model_name, generation["seed"]))
            return self.result(scores[model.model_name])

        with mock.patch.object(evaluation, "verify_model", side_effect=verify):
            payload = evaluation.run_specificity(
                object(),
                self.source,
                self.benchmark,
                self.groups(),
                {"seeds": list(range(10)), "generation": {"do_sample": False}},
            )
        self.assertEqual(payload["summary"]["roc_auc"], 1.0)
        self.assertEqual(payload["effective_seeds"], [0])
        self.assertEqual(len(calls), 4)
        self.assertEqual(
            payload["summary"]["derivative_robustness"][0]["score"], 0.0
        )

    def test_all_zero_panel_has_no_rank_threshold_or_fpr(self):
        with mock.patch.object(
            evaluation, "verify_model", return_value=self.result(0.0)
        ):
            payload = evaluation.run_specificity(
                object(), self.source, self.benchmark, self.groups(),
                {"seeds": [0], "generation": {"do_sample": False}},
            )
        self.assertEqual(payload["summary"]["roc_auc"], 0.5)
        self.assertIsNone(payload["summary"]["source_score_rank"])
        self.assertEqual(payload["threshold"]["status"], "degenerate_all_equal")
        self.assertIsNone(payload["threshold"]["value"])
        self.assertIsNone(payload["summary"]["overall_model_fpr"])

    def test_held_out_fpr_is_separate_from_calibration(self):
        scores = {"source": 0.8, "derivative": 0.7, "negative-a": 0.2, "negative-b": 0.9}

        def verify(method, source, model, generation):
            return self.result(scores[model.model_name])

        with mock.patch.object(evaluation, "verify_model", side_effect=verify):
            payload = evaluation.run_specificity(
                object(), self.source, self.benchmark, self.groups(split=True),
                {"seeds": [0], "generation": {"do_sample": False}},
            )
        self.assertEqual(payload["threshold"]["status"], "held_out")
        self.assertEqual(payload["threshold"]["in_sample_descriptive_fpr"], 0.0)
        self.assertEqual(payload["threshold"]["held_out_fpr"], 1.0)

    def test_partial_model_failure_is_checkpointed_and_retry_reuses_successes(self):
        checkpoints = []

        def first_verify(method, source, model, generation):
            if model.model_name == "negative-b":
                raise OSError("broken checkpoint")
            return self.result(0.5)

        with mock.patch.object(evaluation, "verify_model", side_effect=first_verify):
            first = evaluation.run_specificity(
                object(), self.source, self.benchmark, self.groups(),
                {"seeds": [0], "generation": {"do_sample": False}},
                checkpoint=lambda value: checkpoints.append(value),
            )
        self.assertEqual(first["failure_count"], 1)
        self.assertEqual(len(checkpoints[-1]["models"]), 4)
        completed = {
            row["evaluation_model"]
            for row in first["models"]
            if row["status"] == "completed"
        }
        self.assertIn("negative-a", completed)

        retried = []

        def retry_verify(method, source, model, generation):
            retried.append(model.model_name)
            return self.result(0.1)

        with mock.patch.object(evaluation, "verify_model", side_effect=retry_verify):
            second = evaluation.run_specificity(
                object(), self.source, self.benchmark, self.groups(),
                {"seeds": [0], "generation": {"do_sample": False}},
                existing=first,
            )
        self.assertEqual(retried, ["negative-b"])
        self.assertEqual(second["failure_count"], 0)


if __name__ == "__main__":
    unittest.main()
