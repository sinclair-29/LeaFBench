from __future__ import annotations

import math
from typing import Sequence

import torch
from scipy.stats import binom, norm

from deploying_techniques.watermark.config import WatermarkConfig
from deploying_techniques.watermark.processor import Greenlist


class WatermarkDetector:
    """Extract KGW, OPT, or MorphMark from exact completion token IDs."""

    def __init__(
        self,
        config: WatermarkConfig,
        vocab_size: int,
        device: torch.device | str,
    ):
        config.validate()
        self.config = config
        self.vocab_size = int(vocab_size)
        self.device = torch.device(device)
        self.greenlist = Greenlist(config, self.vocab_size, self.device)

    def detect_token_ids(
        self,
        completion_ids: Sequence[int],
        prefix_ids: Sequence[int] = (),
    ) -> dict[str, float | int | bool | str]:
        completion = [int(token) for token in completion_ids]
        prefix = [int(token) for token in prefix_ids]
        first_index = 0 if prefix else 1
        green_count = 0
        scored = 0

        for index in range(first_index, len(completion)):
            context = prefix + completion[:index]
            token = completion[index]
            ids = self.greenlist.ids(context)
            if bool((ids == token).any().item()):
                green_count += 1
            scored += 1

        if scored == 0:
            return {
                "is_watermarked": False,
                "score": 0.0,
                "z_score": 0.0,
                "p_value": 1.0,
                "num_tokens_scored": 0,
                "num_green_tokens": 0,
                "green_fraction": 0.0,
                "decision_rule": "green_count" if self.config.method == "opt" else "z_score",
            }

        expected = scored * self.config.gamma
        variance = scored * self.config.gamma * (1.0 - self.config.gamma)
        z_score = (green_count - expected) / math.sqrt(variance)
        if self.config.method == "opt":
            p_value = float(binom.sf(green_count - 1, scored, self.config.gamma))
            is_watermarked = p_value <= self.config.significance_level
            decision_rule = "exact_binomial_tail"
        else:
            p_value = float(norm.sf(z_score))
            is_watermarked = z_score > self.config.z_threshold
            decision_rule = "z_score"

        return {
            "is_watermarked": bool(is_watermarked),
            "score": float(z_score),
            "z_score": float(z_score),
            "p_value": p_value,
            "num_tokens_scored": scored,
            "num_green_tokens": green_count,
            "green_fraction": green_count / scored,
            "expected_green_fraction": self.config.gamma,
            "decision_rule": decision_rule,
        }

    def detect_text(self, text: str, tokenizer) -> dict[str, float | int | bool | str]:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        return self.detect_token_ids(token_ids)
