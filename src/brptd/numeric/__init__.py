"""BRPTD 定点残差与公开分桶核心。"""

from .bins import ResidualBin, bucket_interval, compute_residual_bin
from .config import ResidualBinConfig
from .fixed_point import (
    COEFFICIENT_SCALE,
    MEASUREMENT_SCALE,
    NORMALIZATION_SCALE,
    RISTRETTO_SCALAR_ORDER,
    U64_MAX,
    FixedPointConfig,
    canonical_split,
    decode,
    encode,
    round_half_away_from_zero,
)
from .residual import ResidualComponent, exact_residual, residual_components

__all__ = [
    "COEFFICIENT_SCALE",
    "MEASUREMENT_SCALE",
    "NORMALIZATION_SCALE",
    "RISTRETTO_SCALAR_ORDER",
    "U64_MAX",
    "FixedPointConfig",
    "ResidualBin",
    "ResidualBinConfig",
    "ResidualComponent",
    "bucket_interval",
    "canonical_split",
    "compute_residual_bin",
    "decode",
    "encode",
    "exact_residual",
    "residual_components",
    "round_half_away_from_zero",
]
