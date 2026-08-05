from __future__ import annotations

import math
from typing import Sequence

import torch
from transformers import LogitsProcessor

from deploying_techniques.watermark.config import WatermarkConfig


class Greenlist:
    """Device-local PRFs used by the reference implementations."""

    def __init__(self, config: WatermarkConfig, vocab_size: int, device: torch.device):
        self.config = config
        self.vocab_size = int(vocab_size)
        self.device = torch.device(device)
        self.rng = torch.Generator(device=self.device)
        self.prf = None
        if config.method == "morphmark":
            self.rng.manual_seed(config.hash_key)
            self.prf = torch.randperm(
                self.vocab_size,
                generator=self.rng,
                device=self.device,
            )

    def seed(self, prefix: Sequence[int]) -> int:
        if not prefix:
            raise ValueError("At least one prefix token is required to construct a green list")
        previous_token = int(prefix[-1])
        if self.config.method == "morphmark":
            prf_value = int(self.prf[previous_token % self.vocab_size].item())
            return (self.config.hash_key * prf_value) % self.vocab_size
        # KGW simple_1, also used by the OPT experiments.
        return self.config.hash_key * previous_token

    def ids(self, prefix: Sequence[int]) -> torch.Tensor:
        self.rng.manual_seed(self.seed(prefix))
        permutation = torch.randperm(
            self.vocab_size,
            generator=self.rng,
            device=self.device,
        )
        return permutation[: int(self.config.gamma * self.vocab_size)]


def green_mask(vocab_size: int, green_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    mask[green_ids] = True
    return mask


def opt_damage(logits: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    """The B(p_t, G_t) damage criterion from Wouters (2024)."""

    eps = 1e-12
    probabilities = torch.softmax(logits.float(), dim=-1).clamp_min(eps)
    gamma_t = probabilities[mask].sum().clamp(eps, 1.0 - eps)
    indicator = mask.to(probabilities.dtype)
    coefficient = (gamma_t - indicator) / (gamma_t * (1.0 - gamma_t))
    damage = torch.sum(coefficient * probabilities * torch.log(probabilities))
    return float(damage.item()), float(gamma_t.item())


def morph_strength(config: WatermarkConfig, p_green: float) -> float:
    # MarkLLM's MorphMark implementation applies no shift below p0.
    if p_green < config.p0:
        return 0.0
    if config.variant == "linear":
        return config.k_linear * p_green
    if config.variant == "exp":
        return math.exp(config.k_exp * p_green) - 1.0
    return math.log(config.k_log * p_green + 1.0)


def apply_morphmark(
    logits: torch.Tensor,
    mask: torch.Tensor,
    config: WatermarkConfig,
) -> tuple[torch.Tensor, dict[str, float | bool]]:
    # This mirrors MarkLLM's probability-mass transfer, including its clamp and
    # renormalization after large adaptive shifts.
    probabilities = torch.softmax(logits.float(), dim=-1)
    p_green = float(probabilities[mask].sum().item())
    strength = morph_strength(config, p_green)
    beta = strength * (1.0 - p_green)

    adjusted = probabilities.clone()
    green_weights = probabilities[mask]
    red_weights = probabilities[~mask]
    adjusted[mask] += (green_weights / green_weights.sum()) * beta
    adjusted[~mask] -= (red_weights / red_weights.sum()) * beta
    adjusted = torch.nan_to_num(adjusted, nan=0.0).clamp_min(0.0)
    adjusted /= adjusted.sum().clamp_min(torch.finfo(adjusted.dtype).tiny)
    return torch.log(adjusted), {
        "p_green": p_green,
        "strength": strength,
        "applied": strength != 0.0,
    }


class WatermarkLogitsProcessor(LogitsProcessor):
    """One Hugging Face logits processor for all three transferred methods."""

    def __init__(self, config: WatermarkConfig, vocab_size: int):
        config.validate()
        self.config = config
        self.vocab_size = int(vocab_size)
        self._greenlists: dict[tuple[int, str], Greenlist] = {}
        self.step_metadata: list[list[dict[str, float | int | bool | str]]] = []

    def _greenlist(self, vocab_size: int, device: torch.device) -> Greenlist:
        key = (vocab_size, str(device))
        if key not in self._greenlists:
            self._greenlists[key] = Greenlist(self.config, vocab_size, device)
        return self._greenlists[key]

    def _apply_row(
        self,
        prefix: Sequence[int],
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float | int | bool | str]]:
        vocab_size = min(self.vocab_size, logits.shape[-1])
        utility = self._greenlist(vocab_size, logits.device)
        ids = utility.ids(prefix)
        mask = green_mask(logits.shape[-1], ids, logits.device)
        info: dict[str, float | int | bool | str] = {
            "method": self.config.method,
            "seed": utility.seed(prefix),
            "greenlist_size": int(ids.numel()),
            "applied": False,
        }

        if self.config.method == "kgw":
            result = logits.clone()
            result[mask] += self.config.delta
            info["applied"] = True
            return result, info

        if self.config.method == "opt":
            damage, gamma_t = opt_damage(logits, mask)
            info.update({"damage": damage, "gamma_t": gamma_t})
            if damage <= self.config.beta:
                result = logits.clone()
                result[~mask] = -torch.inf
                info["applied"] = True
                return result, info
            return logits, info

        result, morph_info = apply_morphmark(logits, mask, self.config)
        info.update(morph_info)
        return result.to(logits.dtype), info

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[-1] < 1:
            return scores
        output = scores.clone()
        batch_metadata = []
        for batch_index in range(scores.shape[0]):
            prefix = input_ids[batch_index].detach().cpu().tolist()
            output[batch_index], info = self._apply_row(prefix, scores[batch_index])
            batch_metadata.append(info)
        self.step_metadata.append(batch_metadata)
        return output
