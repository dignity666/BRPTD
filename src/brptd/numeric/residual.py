"""标准化 L1 整数残差及其规范分量。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .config import ResidualBinConfig, _checked_product, _checked_sum
from .fixed_point import U64_MAX, _integer, canonical_split


@dataclass(frozen=True)
class ResidualComponent:
    """单个坐标的有符号残差、规范正负分量和本轮公开上界。"""

    signed: int
    positive: int
    negative: int
    bound: int

    def __post_init__(self) -> None:
        signed = _integer(self.signed, "signed")
        positive = _integer(self.positive, "positive")
        negative = _integer(self.negative, "negative")
        bound = _integer(self.bound, "bound")
        if positive < 0 or negative < 0 or bound < 0:
            raise ValueError("positive, negative, and bound must be nonnegative")
        if max(positive, negative, bound, abs(signed)) > U64_MAX:
            raise OverflowError("residual component exceeds u64")
        if positive - negative != signed or positive * negative != 0:
            raise ValueError("positive and negative must canonically split signed")
        if positive + negative > bound:
            raise ValueError("residual component exceeds its round bound")

        object.__setattr__(self, "signed", signed)
        object.__setattr__(self, "positive", positive)
        object.__setattr__(self, "negative", negative)
        object.__setattr__(self, "bound", bound)

    @property
    def z_plus(self) -> int:
        """返回论文记号中的正残差分量。"""

        return self.positive

    @property
    def z_minus(self) -> int:
        """返回论文记号中的负残差分量。"""

        return self.negative

    @property
    def absolute(self) -> int:
        """返回该坐标对精确 L1 残差的贡献。"""

        return self.positive + self.negative


def _integer_vector(
    values: Sequence[int],
    expected_length: int,
    name: str,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be an integer sequence")
    if len(values) != expected_length:
        raise ValueError(f"{name} dimension must equal the configured domain dimension")
    return tuple(_integer(value, f"{name}[{index}]") for index, value in enumerate(values))


def residual_components(
    measurement: Sequence[int],
    truth: Sequence[int],
    config: ResidualBinConfig,
) -> tuple[ResidualComponent, ...]:
    """计算每维规范残差分量，并按当前 truth 推导本轮 bound。"""

    if not isinstance(config, ResidualBinConfig):
        raise ValueError("config must be a ResidualBinConfig")
    dimension = len(config.domains)
    measurements = _integer_vector(measurement, dimension, "measurement")
    truths = _integer_vector(truth, dimension, "truth")

    components: list[ResidualComponent] = []
    for index, ((lower, upper), eta, measured, center) in enumerate(
        zip(config.domains, config.etas, measurements, truths, strict=True)
    ):
        if measured < lower or measured > upper:
            raise ValueError(f"measurement[{index}] is outside the public domain")
        if center < lower or center > upper:
            raise ValueError(f"truth[{index}] is outside the public domain")

        difference = measured - center
        magnitude = _checked_product(
            eta,
            abs(difference),
            config.fixed_point,
            f"residual component {index}",
        )
        signed = magnitude if difference >= 0 else -magnitude
        positive, negative = canonical_split(signed)
        maximum_distance = max(abs(lower - center), abs(upper - center))
        bound = _checked_product(
            eta,
            maximum_distance,
            config.fixed_point,
            f"round bound {index}",
        )
        components.append(
            ResidualComponent(
                signed=signed,
                positive=positive,
                negative=negative,
                bound=bound,
            )
        )
    return tuple(components)


def exact_residual(
    measurement: Sequence[int],
    truth: Sequence[int],
    config: ResidualBinConfig,
) -> int:
    """返回规范分量绝对值之和形成的精确整数残差。"""

    total = 0
    for component in residual_components(measurement, truth, config):
        total = _checked_sum(
            total,
            component.absolute,
            config.fixed_point,
            "exact residual sum",
        )
    if total > config.err_max:
        raise OverflowError("exact residual exceeds err_max")
    return total
