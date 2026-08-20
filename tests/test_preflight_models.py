from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/v100/preflight_models.py"
SPEC = importlib.util.spec_from_file_location("preflight_models", SCRIPT)
assert SPEC and SPEC.loader
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


class PreflightModelsTest(unittest.TestCase):
    def test_index_requires_every_referenced_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"a": "missing.safetensors"}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FileNotFoundError, "Missing indexed"):
                PREFLIGHT.weight_files(root)

    def test_referenced_models_include_source_groups_and_deployment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.yaml"
            path.write_text(
                """
source_model: {model_name: source}
model_groups:
  original: [{model_name: source}]
  derivatives: [{model_name: derivative}]
  negatives: [{model_name: negative}]
evaluations:
  deployment_robustness: {model_name: deployed}
""",
                encoding="utf-8",
            )
            self.assertEqual(
                PREFLIGHT.referenced_models([path]),
                {"source", "derivative", "negative", "deployed"},
            )


if __name__ == "__main__":
    unittest.main()
