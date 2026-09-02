"""残差分桶的公开不可变配置。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal

from .fixed_point import (
    U64_MAX,
    DecimalInput,
    FixedPointConfig,
    _finite_decimal,
    _integer,
    _positive_integer,
    encode,
)


def _sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    return value


def _checked_product(
    left: int,
    right: int,
    fixed_point: FixedPointConfig,
    name: str,
) -> int:
    product = left * right
    if product > U64_MAX:
        raise OverflowError(f"{name} exceeds u64")
    if product > fixed_point.range_maximum:
        raise OverflowError(f"{name} exceeds range_bits={fixed_point.range_bits} capacity")
    return product


def _checked_sum(
    total: int,
    term: int,
    fixed_point: FixedPointConfig,
    name: str,
) -> int:
    if term > U64_MAX - total:
        raise OverflowError(f"{name} exceeds u64")
    result = total + term
    if result > fixed_point.range_maximum:
        raise OverflowError(f"{name} exceeds range_bits={fixed_point.range_bits} capacity")
    return result


def _derive_err_max(
    domains: tuple[tuple[int, int], ...],
    etas: tuple[int, ...],
    fixed_point: FixedPointConfig,
) -> int:
    total = 0
    for index, (eta, (lower, upper)) in enumerate(zip(etas, domains, strict=True)):
        term = _checked_product(
            eta,
            upper - lower,
            fixed_point,
            f"domains[{index}] residual product",
        )
        total = _checked_sum(total, term, fixed_point, "err_max sum")
    return total


@dataclass(frozen=True)
class ResidualBinConfig:
    """已编码域、标准化系数和固定宽度桶的公开参数。"""

    domains: tuple[tuple[int, int], ...]
    etas: tuple[int, ...]
    err_max: int
    bin_count: int
    delta_bin: int
    fixed_point: FixedPointConfig = field(default_factory=FixedPointConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.fixed_point, FixedPointConfig):
            raise ValueError("fixed_point must be a FixedPointConfig")

        raw_domains = _sequence(self.domains, "domains")
        if not raw_domains:
            raise ValueError("domains must be nonempty")
        domains: list[tuple[int, int]] = []
        for index, raw_domain in enumerate(raw_domains):
            domain = _sequence(raw_domain, f"domains[{index}]")
            if len(domain) != 2:
                raise ValueError(f"domains[{index}] must contain two endpoints")
            lower = _integer(domain[0], f"domains[{index}][0]")
            upper = _integer(domain[1], f"domains[{index}][1]")
            if lower >= upper:
                raise ValueError(f"domains[{index}] must be strictly increasing")
            domains.append((lower, upper))

        raw_etas = _sequence(self.etas, "etas")
        if len(raw_etas) != len(domains):
            raise ValueError("domains and etas must have equal dimensions")
        etas = tuple(_positive_integer(value, f"etas[{index}]") for index, value in enumerate(raw_etas))
        canonical_domains = tuple(domains)
        expected_err_max = _derive_err_max(canonical_domains, etas, self.fixed_point)
        err_max = _positive_integer(self.err_max, "err_max")
        if err_max != expected_err_max:
            raise ValueError("err_max does not match domains and etas")

        bin_count = _positive_integer(self.bin_count, "bin_count")
        delta_bin = _positive_integer(self.delta_bin, "delta_bin")
        if delta_bin > U64_MAX:
            raise OverflowError("delta_bin exceeds u64")
        expected_delta = (err_max + bin_count) // bin_count
        if delta_bin != expected_delta:
            raise ValueError("delta_bin does not equal ceil((err_max + 1) / bin_count)")

        object.__setattr__(self, "domains", canonical_domains)
        object.__setattr__(self, "etas", etas)
        object.__setattr__(self, "err_max", err_max)
        object.__setattr__(self, "bin_count", bin_count)
        object.__setattr__(self, "delta_bin", delta_bin)

    @classmethod
    def from_standardized_domains(
        cls,
        domains: Sequence[tuple[DecimalInput, DecimalInput]],
        bin_count: int = 8192,
        *,
        etas: Sequence[DecimalInput] | None = None,
        fixed_point: FixedPointConfig | None = None,
    ) -> ResidualBinConfig:
        """编码实数域与 eta，并从全局上界推导统一桶宽。"""

        fixed = FixedPointConfig() if fixed_point is None else fixed_point
        if not isinstance(fixed, FixedPointConfig):
            raise ValueError("fixed_point must be a FixedPointConfig")
        requested_bin_count = _positive_integer(bin_count, "bin_count")
        raw_domains = _sequence(domains, "domains")
        if not raw_domains:
            raise ValueError("domains must be nonempty")

        encoded_domains: list[tuple[int, int]] = []
        for index, raw_domain in enumerate(raw_domains):
            domain = _sequence(raw_domain, f"domains[{index}]")
            if len(domain) != 2:
                raise ValueError(f"domains[{index}] must contain two endpoints")
            lower_value = _finite_decimal(domain[0], f"domains[{index}][0]")
            upper_value = _finite_decimal(domain[1], f"domains[{index}][1]")
            if lower_value >= upper_value:
                raise ValueError(f"domains[{index}] must be strictly increasing")
            lower = encode(lower_value, fixed.measurement_scale)
            upper = encode(upper_value, fixed.measurement_scale)
            if lower >= upper:
                raise ValueError(f"domains[{index}] collapses at measurement_scale")
            encoded_domains.append((lower, upper))

        if etas is None:
            raw_etas: Sequence[object] = (Decimal(1),) * len(encoded_domains)
        else:
            raw_etas = _sequence(etas, "etas")
        if len(raw_etas) != len(encoded_domains):
            raise ValueError("domains and etas must have equal dimensions")

        encoded_etas: list[int] = []
        for index, raw_eta in enumerate(raw_etas):
            eta_value = _finite_decimal(raw_eta, f"etas[{index}]")
            if eta_value <= 0:
                raise ValueError(f"etas[{index}] must be positive")
            eta = encode(eta_value, fixed.coefficient_scale)
            if eta <= 0:
                raise ValueError(f"etas[{index}] collapses at coefficient_scale")
            encoded_etas.append(eta)

        canonical_domains = tuple(encoded_domains)
        canonical_etas = tuple(encoded_etas)
        err_max = _derive_err_max(canonical_domains, canonical_etas, fixed)
        delta_bin = (err_max + requested_bin_count) // requested_bin_count
        if delta_bin > U64_MAX:
            raise OverflowError("delta_bin exceeds u64")
        return cls(
            domains=canonical_domains,
            etas=canonical_etas,
            err_max=err_max,
            bin_count=requested_bin_count,
            delta_bin=delta_bin,
            fixed_point=fixed,
        )

    @property
    def normalization_scale(self) -> int:
        """返回残差解码所用的归一化尺度。"""

        return self.fixed_point.normalization_scale

    @property
    def encoded_domains(self) -> tuple[tuple[int, int], ...]:
        """返回显式命名的已编码域别名。"""

        return self.domains

    @property
    def encoded_etas(self) -> tuple[int, ...]:
        """返回显式命名的已编码 eta 别名。"""

        return self.etas

    @property
    def max_label(self) -> int:
        """返回实际非空桶的最大标签。"""

        return self.err_max // self.delta_bin

    @property
    def actual_bin_count(self) -> int:
        """返回整数域中实际非空的桶数量。"""

        return self.max_label + 1
