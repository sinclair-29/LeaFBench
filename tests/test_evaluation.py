from __future__ import annotations

import json
import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
