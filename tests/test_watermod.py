from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch

from deploying_techniques.watermark.config import WatermarkConfig
from deploying_techniques.watermark.detector import WatermarkDetector
from deploying_techniques.watermark.model import WatermarkedModel
from deploying_techniques.watermark.processor import WatermarkLogitsProcessor, WaterModUtils


REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ZERO_BIT = REPO_ROOT / "reference" / "WaterMod" / "zero_bit"


def load_official_sam_classes():
    """Import the checked-in released SAM implementation for equivalence tests."""

    sys.path.insert(0, str(OFFICIAL_ZERO_BIT))
    try:
        # Only SAMUtils and SAMLogitsProcessor are under test. Stub the
        # framework base classes so the released visualization module's
        # Python-version-specific annotations do not affect this unit test.
        official_base = ModuleType("watermark.base")
        official_base.BaseWatermark = type("BaseWatermark", (), {})
        official_base.BaseConfig = type("BaseConfig", (), {})
        sys.modules["watermark.base"] = official_base
        from watermark.sam.sam import SAMLogitsProcessor, SAMUtils
    finally:
        sys.path.pop(0)
    return SAMLogitsProcessor, SAMUtils


def watermod_config(**overrides) -> WatermarkConfig:
    values = {
        "method": "watermod",
        "delta": 1.0,
        "hash_key": 15485863,
        "prefix_length": 1,
        "z_threshold": 4.0,
        "f_scheme": "time",
        "entropy_type": "shannon",
        "tau": 1.0,
        "H_scale": 1.2,
    }
    values.update(overrides)
    return WatermarkConfig.from_mapping(values)


class FixedLogitModel(torch.nn.Module):
    """Small deterministic causal model used for CPU-only integration checks."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(vocab_size=vocab_size)
        self.seen_contexts: list[list[int]] = []

    @property
    def device(self):
        return self.anchor.device

    def forward(self, input_ids):
        self.seen_contexts.append(input_ids[0].detach().cpu().tolist())
        batch_size, sequence_length = input_ids.shape
        vocabulary = torch.arange(self.vocab_size, device=input_ids.device).float()
        rows = []
        for position in range(sequence_length):
            center = (input_ids[:, position].float() + position + 1) % self.vocab_size
            rows.append(-torch.abs(vocabulary.unsqueeze(0) - center.unsqueeze(1)))
        logits = torch.stack(rows, dim=1) + self.anchor
        return {"logits": logits.expand(batch_size, -1, -1)}

    def generate(
        self,
        input_ids,
        attention_mask=None,
        logits_processor=None,
        max_new_tokens=8,
        **_,
    ):
        sequence = input_ids.clone()
        for _ in range(max_new_tokens):
            scores = self(input_ids=sequence)["logits"][:, -1]
            if logits_processor is not None:
                scores = logits_processor(sequence, scores)
            next_token = torch.argmax(scores, dim=-1, keepdim=True)
            sequence = torch.cat((sequence, next_token), dim=-1)
        return sequence


class TinyTokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.padding_side = "right"

    def __len__(self):
        return self.vocab_size

    def encode(self, text, add_special_tokens=False):
        if text and all(piece.isdigit() for piece in text.split()):
            return [int(piece) for piece in text.split()]
        return [2 + (ord(character) % (self.vocab_size - 2)) for character in text]

    def __call__(self, texts, return_tensors, padding, truncation, max_length):
        encoded = [self.encode(text)[-max_length:] for text in texts]
        width = max(len(tokens) for tokens in encoded)
        input_ids = []
        attention_mask = []
        for tokens in encoded:
            padding_length = width - len(tokens)
            input_ids.append(([self.pad_token_id] * padding_length) + tokens)
            attention_mask.append(([0] * padding_length) + ([1] * len(tokens)))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens=True):
        excluded = {self.pad_token_id, self.eos_token_id} if skip_special_tokens else set()
        return " ".join(str(int(token)) for token in token_ids if int(token) not in excluded)


class TinyModelPool:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def get_model(self, *_args, **_kwargs):
        return self.model

    def get_tokenizer(self, *_args, **_kwargs):
        return self.tokenizer


class RawPromptSource:
    @staticmethod
    def render_prompts(prompts, tokenizer):
        return list(prompts)


class WaterModOfficialEquivalenceTest(unittest.TestCase):
    def setUp(self):
        self.config = watermod_config()
        self.vocab_size = 10
        self.prefix = torch.tensor([17, 3], dtype=torch.long)
        self.logits = torch.tensor(
            [2.7, -0.4, 1.3, 0.2, -1.1, 3.0, 0.8, -0.2, 1.9, 0.5],
            dtype=torch.float32,
        )

    def official_config(self):
        return SimpleNamespace(
            delta=self.config.delta,
            hash_key=self.config.hash_key,
            prefix_length=self.config.prefix_length,
            z_threshold=self.config.z_threshold,
            f_scheme=self.config.f_scheme,
            entropy_type=self.config.entropy_type,
            tau=self.config.tau,
            H_scale=self.config.H_scale,
            vocab_size=self.vocab_size,
        )

    def test_group_mask_logits_and_z_score_match_released_sam(self):
        OfficialProcessor, OfficialUtils = load_official_sam_classes()
        official_utils = OfficialUtils(self.official_config())
        official_processor = OfficialProcessor(self.official_config(), official_utils)
        native_utils = WaterModUtils(self.config, self.vocab_size)
        native_processor = WatermarkLogitsProcessor(self.config, self.vocab_size)

        official_group = official_utils.choose_group(self.logits, self.prefix)
        ranked = torch.argsort(self.logits, descending=True)
        ranks = torch.arange(len(ranked))
        official_green = ranked[((ranks + 1) % 2) == official_group]

        native_parity = native_utils.green_parity(self.logits, self.prefix)
        native_green = native_utils.green_ids(self.logits, self.prefix)
        self.assertEqual(native_parity, 1 if official_group == 0 else 0)
        self.assertTrue(torch.equal(native_green, official_green))

        input_ids = self.prefix.unsqueeze(0)
        scores = self.logits.unsqueeze(0)
        official_biased = official_processor(input_ids, scores.clone())
        native_biased = native_processor(input_ids, scores.clone())
        torch.testing.assert_close(native_biased, official_biased, rtol=0.0, atol=0.0)

        for green_count, num_scored in ((0, 1), (3, 8), (9, 10)):
            self.assertEqual(
                native_utils.z_score(green_count, num_scored),
                official_utils.z_score(green_count, num_scored),
            )

    def test_official_mixed_log_entropy_probability_is_retained(self):
        uniform_logits = torch.zeros(self.vocab_size)
        probability = WaterModUtils(self.config, self.vocab_size).odd_probability(uniform_logits)
        expected = math.log(2.0) ** self.config.H_scale
        self.assertAlmostEqual(probability, expected, places=6)


class WaterModIntegrationTest(unittest.TestCase):
    def test_leafbench_wrapper_smoke_generation_and_detection(self):
        vocab_size = 12
        model = FixedLogitModel(vocab_size)
        tokenizer = TinyTokenizer(vocab_size)
        wrapper = WatermarkedModel(
            {
                "model_family": "tiny",
                "pretrained_model": "tiny",
                "instruct_model": None,
                "base_model": "tiny",
                "model_name": "tiny_watermark_watermod_0",
                "model_path": "unused",
                "type": "watermark",
                "params": {
                    "max_input_length": 64,
                    "max_new_tokens": 8,
                    "temperature": 0.0,
                    "do_sample": False,
                },
                "watermark": {"method": "watermod"},
            },
            source_model=RawPromptSource(),
            model_pool=TinyModelPool(model, tokenizer),
            accelerator=None,
        )

        outputs = wrapper.generate(["WaterMod"])
        detections = wrapper.detect(outputs)

        self.assertEqual(len(outputs), 1)
        self.assertTrue(outputs[0])
        self.assertEqual(detections[0]["num_tokens_scored"], 8)
        self.assertTrue(math.isfinite(float(detections[0]["z_score"])))

    def test_embedding_and_completion_only_detection_produce_finite_z_score(self):
        config = watermod_config()
        vocab_size = 12
        model = FixedLogitModel(vocab_size)
        processor = WatermarkLogitsProcessor(config, vocab_size)
        prompt = [4, 7]
        generated: list[int] = []

        for _ in range(8):
            context = prompt + generated
            raw_logits = model(input_ids=torch.tensor([context]))["logits"][0, -1]
            biased = processor(
                torch.tensor([context]),
                raw_logits.unsqueeze(0).clone(),
            )[0]
            generated.append(int(torch.argmax(biased).item()))

        model.seen_contexts.clear()
        detector = WatermarkDetector(config, vocab_size, "cpu", model=model)
        result = detector.detect_token_ids(generated, prompt)

        self.assertEqual(result["num_tokens_scored"], len(generated))
        self.assertTrue(math.isfinite(float(result["z_score"])))
        self.assertEqual(
            model.seen_contexts,
            [prompt + generated[:index] for index in range(len(generated))],
        )

    def test_default_configuration_is_official_sam(self):
        config = WatermarkConfig.from_mapping({"method": "watermod"})
        self.assertEqual(config.delta, 1.0)
        self.assertEqual(config.entropy_type, "shannon")
        self.assertEqual(config.H_scale, 1.2)
        self.assertEqual(config.prefix_length, 1)
        self.assertEqual(config.z_threshold, 4.0)
        self.assertEqual(config.gamma, 0.5)

    def test_greedy_generation_does_not_forward_sampling_only_arguments(self):
        wrapper = object.__new__(WatermarkedModel)
        wrapper.params = {
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "max_new_tokens": 64,
        }
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1)
        params = wrapper._generation_params(tokenizer, {})

        self.assertFalse(params["do_sample"])
        self.assertNotIn("temperature", params)
        self.assertNotIn("top_p", params)
        self.assertNotIn("top_k", params)


if __name__ == "__main__":
    unittest.main()
