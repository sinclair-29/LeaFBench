from __future__ import annotations

import unittest

try:
    from fingerprint.plugae.plugae import PlugAEFingerprint, PROFLINGO_TEMPLATES
except ModuleNotFoundError as error:
    if error.name not in {"torch", "transformers"}:
        raise
    PlugAEFingerprint = None
    PROFLINGO_TEMPLATES = ()


class AtomicTokenTokenizer:
    def __init__(self, token="mkahg"):
        self.token = token

    def encode(self, text, add_special_tokens=False):
        ids = [1] if add_special_tokens else []
        index = 0
        while index < len(text):
            if text.startswith(self.token, index):
                ids.append(999)
                index += len(self.token)
            else:
                ids.append(10 + ord(text[index]))
                index += 1
        return ids


@unittest.skipIf(PlugAEFingerprint is None, "PlugAE runtime dependencies are unavailable")
class PlugAEProtocolTest(unittest.TestCase):
    def method(self):
        method = PlugAEFingerprint({"num_queries": 2})
        method.queries = ["Question one?", "Question two?"]
        method.targets = ["1234", "5678"]
        method.keywords = ["1234", "5678"]
        method.output_parser = method._infer_output_parser(method.keywords)
        return method

    def test_validation_text_reproduces_training_embedding_position(self):
        method = self.method()
        specs = method._prompt_specs(method.copyright_token)
        self.assertEqual(len(specs), 2 * len(PROFLINGO_TEMPLATES))
        method._validate_template_round_trip(
            AtomicTokenTokenizer(), method.copyright_token, specs
        )
        self.assertEqual(
            {item["template_id"] for item in specs}, {"alpaca", "zero_shot"}
        )

    def test_fixed_digit_parser_distinguishes_invalid_and_wrong_valid_output(self):
        method = self.method()
        parsed, success, invalid = method._parse_output("answer: 9999", "1234")
        self.assertEqual(parsed, "9999")
        self.assertFalse(success)
        self.assertFalse(invalid)
        parsed, success, invalid = method._parse_output("no digits", "1234")
        self.assertIsNone(parsed)
        self.assertFalse(success)
        self.assertTrue(invalid)

    def test_natural_text_only_marks_blank_output_invalid(self):
        method = self.method()
        method.keywords = ["north", "honey"]
        method.output_parser = method._infer_output_parser(method.keywords)
        self.assertEqual(method.output_parser["kind"], "nonempty_text")
        self.assertEqual(method._parse_output("south", "north"), ("south", False, False))
        self.assertEqual(method._parse_output("  ", "north"), (None, False, True))

    def test_metrics_report_both_templates_and_query_aggregates(self):
        method = self.method()
        outputs = ["1234", "none", "5678", "5678"]
        trials, metrics = method._trials_and_metrics(outputs, seed=0)
        self.assertEqual(len(trials), 4)
        self.assertEqual(metrics["transfer_response_rate"], 0.75)
        self.assertEqual(metrics["query_any_template_hit_rate"], 1.0)
        self.assertEqual(metrics["query_all_templates_hit_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
