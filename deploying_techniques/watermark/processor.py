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


class WaterModUtils:
    """Official zero_bit/SAM entropy gate, PRF, parity split, and z-score."""

    def __init__(self, config: WatermarkConfig, vocab_size: int):
        if config.method != "watermod":
            raise ValueError("WaterModUtils requires method='watermod'")
        self.config = config
        self.vocab_size = int(vocab_size)

    def seed(self, prefix: Sequence[int] | torch.Tensor) -> int:
        values = torch.as_tensor(prefix, dtype=torch.long).flatten()
        window = values[-self.config.prefix_length :]
        if window.numel() < self.config.prefix_length:
            raise ValueError("Not enough prefix tokens for WaterMod's configured prefix_length")
        if self.config.f_scheme == "additive":
            return int(window.sum())
        if self.config.f_scheme == "time":
            return int(torch.prod(window))
        if self.config.f_scheme == "skip":
            return int(window[0].item())
        return int(window.min())

    def hash_to_uniform(self, seed: int) -> float:
        # The released SAM implementation deliberately uses a CPU generator,
        # independent of the model device and global RNG state.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed ^ self.config.hash_key)
        return float(torch.rand((), generator=generator))

    def odd_probability(self, logits: torch.Tensor) -> float:
        probabilities = torch.softmax(logits.float(), dim=-1)
        if self.config.entropy_type == "shannon":
            entropy = float(
                -(probabilities * torch.log(probabilities + 1e-12)).sum()
            )
            # Official SAM mixes natural-log entropy with a base-2 maximum.
            # This is retained intentionally as the executable baseline.
            maximum_entropy = math.log2(self.vocab_size)
        else:
            tau = self.config.tau
            entropy = float((probabilities / (1.0 + tau * probabilities)).sum())
            maximum_entropy = 1.0 / (1.0 + (tau / self.vocab_size))
        normalized = entropy / maximum_entropy
        return min(1.0, max(0.0, normalized**self.config.H_scale))

    def green_parity(self, logits: torch.Tensor, prefix: Sequence[int] | torch.Tensor) -> int:
        probability = self.odd_probability(logits)
        uniform = self.hash_to_uniform(self.seed(prefix))
        # Clear zero-based semantics equivalent to official SAM's inverted
        # group label combined with its one-based rank expression.
        return 1 if uniform < probability else 0

    def green_ids(
        self,
        logits: torch.Tensor,
        prefix: Sequence[int] | torch.Tensor,
    ) -> torch.Tensor:
        parity = self.green_parity(logits, prefix)
        ranked_ids = torch.argsort(logits, descending=True)
        zero_based_ranks = torch.arange(len(ranked_ids), device=ranked_ids.device)
        return ranked_ids[zero_based_ranks % 2 == parity]

    @staticmethod
    def z_score(green_count: int, num_scored: int) -> float:
        if num_scored <= 0:
            return 0.0
        return (green_count - (0.5 * num_scored)) / math.sqrt(0.25 * num_scored)


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
    """One Hugging Face logits processor for LeaFBench watermark methods."""

    def __init__(self, config: WatermarkConfig, vocab_size: int):
        config.validate()
        self.config = config
        self.vocab_size = int(vocab_size)
        self._greenlists: dict[tuple[int, str], Greenlist] = {}
        self._watermod_utils: dict[int, WaterModUtils] = {}
        self.step_metadata: list[list[dict[str, float | int | bool | str]]] = []

    def _greenlist(self, vocab_size: int, device: torch.device) -> Greenlist:
        key = (vocab_size, str(device))
        if key not in self._greenlists:
            self._greenlists[key] = Greenlist(self.config, vocab_size, device)
        return self._greenlists[key]

    def _watermod(self, vocab_size: int) -> WaterModUtils:
        if vocab_size not in self._watermod_utils:
            self._watermod_utils[vocab_size] = WaterModUtils(self.config, vocab_size)
        return self._watermod_utils[vocab_size]

    def _apply_row(
        self,
        prefix: Sequence[int],
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float | int | bool | str]]:
        vocab_size = min(self.vocab_size, logits.shape[-1])
        if self.config.method == "watermod":
            utility = self._watermod(vocab_size)
            seed = utility.seed(prefix)
            probability = utility.odd_probability(logits)
            uniform = utility.hash_to_uniform(seed)
            green_parity = 1 if uniform < probability else 0
            ranked_ids = torch.argsort(logits, descending=True)
            ranks = torch.arange(len(ranked_ids), device=ranked_ids.device)
            green_ids = ranked_ids[ranks % 2 == green_parity]
            result = logits.clone()
            result[green_ids] += self.config.delta
            return result, {
                "method": self.config.method,
                "seed": seed,
                "uniform": uniform,
                "odd_probability": probability,
                "green_parity": green_parity,
                "greenlist_size": int(green_ids.numel()),
                "applied": True,
            }

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
        required_prefix = self.config.prefix_length if self.config.method == "watermod" else 1
        if input_ids.shape[-1] < required_prefix:
            return scores
        output = scores.clone()
        batch_metadata = []
        for batch_index in range(scores.shape[0]):
            prefix = input_ids[batch_index].detach().cpu().tolist()
            output[batch_index], info = self._apply_row(prefix, scores[batch_index])
            batch_metadata.append(info)
        self.step_metadata.append(batch_metadata)
        return output
