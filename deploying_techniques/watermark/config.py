from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping


@dataclass(frozen=True)
class WatermarkConfig:
    """Configuration shared by embedding and extraction."""

    method: str
    gamma: float = 0.5
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

    # WaterMod zero_bit/SAM
    prefix_length: int = 1
    f_scheme: str = "time"
    entropy_type: str = "shannon"
    tau: float = 1.0
    H_scale: float = 1.2

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "WatermarkConfig":
        aliases = {"watermark_type": "method", "morph_variant": "variant", "morph_p0": "p0"}
        normalized = {aliases.get(key, key): value for key, value in values.items()}
        if normalized.get("method") == "watermod":
            # SAM partitions token ranks into two parity classes, so gamma is
            # fixed rather than an independently tunable WaterMod parameter.
            normalized.setdefault("gamma", 0.5)
            normalized.setdefault("delta", 1.0)
        allowed = {field.name for field in fields(cls)}
        config = cls(**{key: value for key, value in normalized.items() if key in allowed})
        config.validate()
        return config

    def validate(self) -> None:
        if self.method not in {"kgw", "opt", "morphmark", "watermod"}:
            raise ValueError(f"Unsupported watermark method: {self.method!r}")
        if not 0.0 < self.gamma < 1.0:
            raise ValueError("gamma must lie strictly between zero and one")
        if self.method == "morphmark" and self.variant not in {"linear", "exp", "log"}:
            raise ValueError(f"Unsupported MorphMark variant: {self.variant!r}")
        if not 0.0 < self.significance_level < 1.0:
            raise ValueError("significance_level must lie strictly between zero and one")
        if self.method == "watermod":
            if self.gamma != 0.5:
                raise ValueError("WaterMod fixes gamma at 0.5 through rank-parity partitioning")
            if self.prefix_length < 1:
                raise ValueError("prefix_length must be positive")
            if self.f_scheme not in {"additive", "time", "skip", "min"}:
                raise ValueError(f"Unsupported WaterMod f_scheme: {self.f_scheme!r}")
            if self.entropy_type not in {"shannon", "spike"}:
                raise ValueError(f"Unsupported WaterMod entropy_type: {self.entropy_type!r}")
            if self.tau <= 0.0:
                raise ValueError("tau must be positive")
            if self.H_scale <= 0.0:
                raise ValueError("H_scale must be positive")
