from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/v100/trap_plugae_smoke/validate_results.py"
)
SPEC = importlib.util.spec_from_file_location("smoke_validation", SCRIPT)
smoke_validation = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = smoke_validation
SPEC.loader.exec_module(smoke_validation)


class SmokeValidationTest(unittest.TestCase):
    def test_checked_in_configs_keep_full_optimization_parameters(self):
        verdict = smoke_validation.Verdict()
        root = Path(__file__).resolve().parents[1] / "config/v100/trap_plugae_smoke"
        smoke_validation.validate_configs(root, verdict)
        self.assertEqual(verdict.protocol_errors, [])

    def test_trial_flags_are_recomputed_from_raw_output(self):
        valid = smoke_validation.Verdict()
        smoke_validation.validate_trial_flags(
            "trap",
            {
                "target": "6532",
                "output": "answer: 6532",
                "parsed_target": "6532",
                "success": 1,
                "invalid": 0,
            },
            "valid trial",
            valid,
        )
        self.assertEqual(valid.protocol_errors, [])

        broken = smoke_validation.Verdict()
        smoke_validation.validate_trial_flags(
            "plugae",
            {
                "keyword": "6532",
                "output": "no numeric answer",
                "parsed_output": None,
                "success": 0,
                "invalid": 0,
            },
            "broken trial",
            broken,
        )
        self.assertTrue(any("invalid flag" in item for item in broken.protocol_errors))

    def test_missing_results_are_protocol_failure(self):
        verdict = smoke_validation.Verdict()
        with tempfile.TemporaryDirectory() as temporary:
            batch = smoke_validation.discover_batch(
                Path(temporary), "trap", verdict
            )
        self.assertIsNone(batch)
        self.assertTrue(verdict.protocol_errors)


if __name__ == "__main__":
    unittest.main()
