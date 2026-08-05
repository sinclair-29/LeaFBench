from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping


@dataclass(frozen=True)
class WatermarkConfig:
    """Configuration shared by embedding and extraction."""

    method: str
    gamma: float
    hash_key: int = 15485863
    z_threshold: float = 4.0

    # KGW
    delta: float = 2.0

    # OPT
    beta: float = 0.0
    significance_level: float = 0.01

    # MorphMark
    variant: str = "exp"
    p0: float = 0.15
    k_linear: float = 1.55
    k_exp: float = 1.30
    k_log: float = 2.15

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "WatermarkConfig":
        aliases = {"watermark_type": "method", "morph_variant": "variant", "morph_p0": "p0"}
        normalized = {aliases.get(key, key): value for key, value in values.items()}
        allowed = {field.name for field in fields(cls)}
        config = cls(**{key: value for key, value in normalized.items() if key in allowed})
        config.validate()
        return config

    def validate(self) -> None:
        if self.method not in {"kgw", "opt", "morphmark"}:
            raise ValueError(f"Unsupported watermark method: {self.method!r}")
        if not 0.0 < self.gamma < 1.0:
            raise ValueError("gamma must lie strictly between zero and one")
        if self.method == "morphmark" and self.variant not in {"linear", "exp", "log"}:
            raise ValueError(f"Unsupported MorphMark variant: {self.variant!r}")
        if not 0.0 < self.significance_level < 1.0:
            raise ValueError("significance_level must lie strictly between zero and one")
