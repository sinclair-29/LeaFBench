from __future__ import annotations

import unittest
from types import SimpleNamespace

try:
    from fingerprint.trap.trap import TRAPFingerprint
except ModuleNotFoundError as error:
    if error.name not in {"torch", "transformers", "pandas"}:
        raise
    TRAPFingerprint = None


class CapturingModel:
    def __init__(self, name="derivative"):
        self.model_name = name
        self.type = "instruct"
        self.calls = []

    def generate(self, prompts, **kwargs):
        self.calls.append((list(prompts), dict(kwargs)))
        return ["code 1234" for _ in prompts]


@unittest.skipIf(TRAPFingerprint is None, "TRAP runtime dependencies are unavailable")
class TrapCheckpointingTest(unittest.TestCase):
    def method(self):
        method = TRAPFingerprint(
            {
                "n_goals": 2,
                "goal_count": 2,
                "goal_offset": 0,
                "prompt_seed": 41,
                "seed": 42,
                "gcg_config": {"seed": 42},
            }
        )
        method.prompts = ["p1", "p2"]
        method.targets = ["t1", "t2"]
        method.string_target = ["1234", "5678"]
        return method

    def test_item_seed_is_stable_and_item_specific(self):
        method = self.method()
        self.assertEqual(method._item_seed(0), method._item_seed(0))
        self.assertNotEqual(method._item_seed(0), method._item_seed(1))

    def test_partial_records_are_checked_against_prepared_targets(self):
        method = self.method()
        record = {
            "item_index": 1,
            "payload": {"kind": "trap", "instruction": "p1", "target": "1234"},
        }
        method.validate_partial_records([record])
        record["payload"]["target"] = "9999"
        with self.assertRaisesRegex(ValueError, "target changed"):
            method.validate_partial_records([record])

    def test_cross_model_verification_uses_suspect_renderer(self):
        method = self.method()
        source = SimpleNamespace(
            model_name="source",
            fingerprint_records=[
                {
                    "fingerprint_id": "fp:001",
                    "payload": {
                        "kind": "trap",
                        "raw_user_prompt": "raw prompt",
                        "rendered_prompt": "source rendered",
                        "target": "1234",
                    },
                }
            ],
        )
        suspect = CapturingModel()
        result = method.verify_fingerprint(
            source, suspect, {"input_mode": "model_rendered", "seed": 0}
        )
        self.assertEqual(result.score, 1.0)
        prompts, kwargs = suspect.calls[0]
        self.assertEqual(prompts, ["raw prompt"])
        self.assertFalse(kwargs["prompts_are_rendered"])
        with self.assertRaisesRegex(ValueError, "cross-model"):
            method.verify_fingerprint(
                source, suspect, {"input_mode": "source_rendered", "seed": 0}
            )


if __name__ == "__main__":
    unittest.main()
