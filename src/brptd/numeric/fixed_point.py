"""协议数值的十进制定点编码。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation, localcontext
from numbers import Integral
from typing import TypeAlias

MEASUREMENT_SCALE = 1_000_000
COEFFICIENT_SCALE = 1_000_000
NORMALIZATION_SCALE = MEASUREMENT_SCALE * COEFFICIENT_SCALE
RISTRETTO_SCALAR_ORDER = 2**252 + 27742317777372353535851937790883648493
U64_MAX = 2**64 - 1

DecimalInput: TypeAlias = Decimal | int | float | str


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _finite_decimal(value: object, name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not decimal_value.is_finite():
        raise ValueError(f"{name} must be finite")
    return decimal_value


@dataclass(frozen=True)
class FixedPointConfig:
    """固定点尺度、范围证明位宽和标量域公开参数。"""

    measurement_scale: int = MEASUREMENT_SCALE
    coefficient_scale: int = COEFFICIENT_SCALE
    range_bits: int = 64
    scalar_order: int = RISTRETTO_SCALAR_ORDER

    def __post_init__(self) -> None:
        measurement_scale = _positive_integer(self.measurement_scale, "measurement_scale")
        coefficient_scale = _positive_integer(self.coefficient_scale, "coefficient_scale")
        range_bits = _positive_integer(self.range_bits, "range_bits")
        scalar_order = _positive_integer(self.scalar_order, "scalar_order")

        # 先按位长排除明显不可能的关系，避免为异常位宽构造巨型整数。
        if range_bits >= scalar_order.bit_length():
            raise ValueError("range_bits must satisfy 2 * (2**range_bits - 1) < scalar_order")
        range_maximum = (1 << range_bits) - 1
        if 2 * range_maximum >= scalar_order:
            raise ValueError("range_bits must satisfy 2 * (2**range_bits - 1) < scalar_order")

        object.__setattr__(self, "measurement_scale", measurement_scale)
        object.__setattr__(self, "coefficient_scale", coefficient_scale)
        object.__setattr__(self, "range_bits", range_bits)
        object.__setattr__(self, "scalar_order", scalar_order)

    @property
    def normalization_scale(self) -> int:
        """返回测量尺度与系数尺度的乘积。"""

        return self.measurement_scale * self.coefficient_scale

    @property
    def range_maximum(self) -> int:
        """返回公开范围证明可表达的最大非负整数。"""

        return (1 << self.range_bits) - 1


def round_half_away_from_zero(value: DecimalInput) -> int:
    """将有限 Decimal 数值舍入到最近整数，半值向远离零方向处理。"""

    decimal_value = _finite_decimal(value, "value")
    return int(decimal_value.to_integral_value(rounding=ROUND_HALF_UP))


def encode(value: DecimalInput, scale: int = MEASUREMENT_SCALE) -> int:
    """按给定正尺度编码单个有限数值。"""

    integer_scale = _positive_integer(scale, "scale")
    decimal_value = _finite_decimal(value, "value")
    required_precision = len(decimal_value.as_tuple().digits) + len(str(integer_scale)) + 2
    with localcontext() as context:
        context.prec = max(context.prec, required_precision)
        scaled = decimal_value * Decimal(integer_scale)
    return round_half_away_from_zero(scaled)


def decode(encoded: int, scale: int = MEASUREMENT_SCALE) -> Decimal:
    """将一个定点整数解码为 Decimal，避免二进制浮点损失。"""

    integer_value = _integer(encoded, "encoded")
    integer_scale = _positive_integer(scale, "scale")
    required_precision = len(str(abs(integer_value))) + len(str(integer_scale)) + 2
    with localcontext() as context:
        context.prec = max(context.prec, required_precision)
        return Decimal(integer_value) / Decimal(integer_scale)


def canonical_split(difference: int) -> tuple[int, int]:
    """返回唯一的非负正负分解，使两侧乘积为零。"""

    integer_difference = _integer(difference, "difference")
    return max(integer_difference, 0), max(-integer_difference, 0)
