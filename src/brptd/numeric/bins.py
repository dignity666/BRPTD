"""固定宽度残差桶的区间与保守代理。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .config import ResidualBinConfig, _checked_product
from .fixed_point import _integer, decode


@dataclass(frozen=True)
class ResidualBin:
    """公开桶标签、闭区间和归一化上端点代理。"""

    label: int
    lo: int
    hi: int
    proxy: Decimal

    def __post_init__(self) -> None:
        label = _integer(self.label, "label")
        lo = _integer(self.lo, "lo")
        hi = _integer(self.hi, "hi")
        if label < 0 or lo < 0 or hi < lo:
            raise ValueError("residual bin label and interval must be nonnegative and ordered")
        if not isinstance(self.proxy, Decimal) or not self.proxy.is_finite():
            raise ValueError("proxy must be a finite Decimal")
        if self.proxy < 0:
            raise ValueError("proxy must be nonnegative")

        object.__setattr__(self, "label", label)
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)

    @property
    def lower(self) -> int:
        """返回桶闭区间下端点的描述性别名。"""

        return self.lo

    @property
    def upper(self) -> int:
        """返回桶闭区间上端点的描述性别名。"""

        return self.hi

    @property
    def normalized_upper(self) -> Decimal:
        """返回保守上端点代理的描述性别名。"""

        return self.proxy


def bucket_interval(label: int, config: ResidualBinConfig) -> tuple[int, int]:
    """返回合法桶标签对应的闭区间，并在 ``err_max`` 处截断。"""

    if not isinstance(config, ResidualBinConfig):
        raise ValueError("config must be a ResidualBinConfig")
    integer_label = _integer(label, "label")
    if integer_label < 0 or integer_label > config.max_label:
        raise ValueError("label is outside the public residual bin domain")
    lower = _checked_product(
        integer_label,
        config.delta_bin,
        config.fixed_point,
        "bin lower endpoint",
    )
    upper = min(
        (integer_label + 1) * config.delta_bin - 1,
        config.err_max,
    )
    return lower, upper


def compute_residual_bin(residual: int, config: ResidualBinConfig) -> ResidualBin:
    """把一个合法精确残差映射为唯一公开桶。"""

    if not isinstance(config, ResidualBinConfig):
        raise ValueError("config must be a ResidualBinConfig")
    integer_residual = _integer(residual, "residual")
    if integer_residual < 0 or integer_residual > config.err_max:
        raise ValueError("residual is outside the public residual domain")
    label = integer_residual // config.delta_bin
    lower, upper = bucket_interval(label, config)
    return ResidualBin(
        label=label,
        lo=lower,
        hi=upper,
        proxy=decode(upper, config.normalization_scale),
    )
