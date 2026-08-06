from __future__ import annotations

import math
from typing import Sequence

import torch
from scipy.stats import binom, norm

from deploying_techniques.watermark.config import WatermarkConfig
from deploying_techniques.watermark.processor import Greenlist, WaterModUtils


class WatermarkDetector:
    """Extract a supported watermark from exact completion token IDs."""

    def __init__(
        self,
        config: WatermarkConfig,
        vocab_size: int,
        device: torch.device | str,
        model=None,
    ):
        config.validate()
        self.config = config
        self.vocab_size = int(vocab_size)
        self.device = torch.device(device)
        self.model = model
        self.greenlist = (
            None
            if config.method == "watermod"
            else Greenlist(config, self.vocab_size, self.device)
        )
        self.watermod = (
            WaterModUtils(config, self.vocab_size)
            if config.method == "watermod"
            else None
        )

    def _watermod_green_ids(self, context: Sequence[int]) -> torch.Tensor:
        if self.model is None or self.watermod is None:
            raise ValueError("WaterMod detection requires the generation model")
        input_ids = torch.tensor([list(context)], dtype=torch.long, device=self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
        logits = outputs["logits"][0, -1]
        return self.watermod.green_ids(logits, context)

    def detect_token_ids(
        self,
        completion_ids: Sequence[int],
        prefix_ids: Sequence[int] = (),
    ) -> dict[str, float | int | bool | str]:
        completion = [int(token) for token in completion_ids]
        prefix = [int(token) for token in prefix_ids]
        green_count = 0
        scored = 0

        for index in range(len(completion)):
            context = prefix + completion[:index]
            required_prefix = self.config.prefix_length if self.config.method == "watermod" else 1
            if len(context) < required_prefix:
                continue
            token = completion[index]
            if self.config.method == "watermod":
                ids = self._watermod_green_ids(context)
            else:
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
        if self.config.method == "watermod":
            z_score = self.watermod.z_score(green_count, scored)
        else:
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
