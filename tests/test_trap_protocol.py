from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    from fingerprint.trap.gcg import GCGOptimizer
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    GCGOptimizer = None


class TinyTokenizer:
    def __init__(self):
        self._vocab = {
            "apple": 0,
            "One": 1,
            "▁two": 2,
            "triangles": 3,
            "XIV": 4,
            "hello": 5,
        }

    def get_vocab(self):
        return self._vocab


@unittest.skipIf(GCGOptimizer is None, "PyTorch is not installed in this environment")
class TrapProtocolTest(unittest.TestCase):
    def test_number_filter_matches_words_plurals_sentencepiece_and_roman(self):
        optimizer = object.__new__(GCGOptimizer)
        optimizer.tokenizer = TinyTokenizer()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "words.csv"
            path.write_text("one\ntwo\ntriangle\n", encoding="utf-8")
            filtered = set(optimizer._get_number_filter_token_ids(path))
        self.assertEqual(filtered, {1, 2, 3, 4})

    def test_roman_filter_rejects_empty_and_accepts_valid_numerals(self):
        self.assertFalse(GCGOptimizer._is_roman_numeral(""))
        self.assertFalse(GCGOptimizer._is_roman_numeral("hello"))
        self.assertTrue(GCGOptimizer._is_roman_numeral("▁XIV"))


if __name__ == "__main__":
    unittest.main()
