"""BRPTD 实验的确定性指标、配对汇总和 BCa 区间。"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import rankdata

FloatArray = npt.NDArray[np.float64]
NORMAL = NormalDist()


class MetricError(ValueError):
    """指标输入或统计条件违反实验契约。"""


ROUND_METRIC_FIELDS = (
    "dataset",
    "attack",
    "trial_id",
    "fold",
    "block",
    "round_index",
    "nominal_malicious_ratio",
    "actual_malicious_ratio",
    "proof_mode",
    "exact_crse",
    "bucket_crse",
    "crse_ratio",
    "exact_worker_ranks",
    "proxy_worker_ranks",
    "spearman",
    "malicious_weight_share",
    "proof_acceptance_rate",
    "valid_update",
)
TRIAL_METRIC_FIELDS = (
    "dataset",
    "attack",
    "trial_id",
    "fold",
    "block",
    "nominal_malicious_ratio",
    "actual_malicious_ratio",
    "proof_mode",
    "exact_crse",
    "bucket_crse",
    "crse_ratio",
    "mean_spearman",
    "malicious_weight_share",
    "proof_acceptance_rate",
    "invalid_round_rate",
    "uncalculable_spearman_count",
    "uncalculable_ratio_count",
)
SUMMARY_FIELDS = (
    "dataset",
    "attack",
    "nominal_malicious_ratio",
    "proof_mode",
    "metric",
    "trial_count",
    "uncalculable_count",
    "mean",
    "sample_std",
    "ci95_lower",
    "ci95_upper",
    "bootstrap_resamples",
    "bootstrap_seed",
)


def _finite_array(value: npt.ArrayLike, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if not array.size or not np.all(np.isfinite(array)):
        raise MetricError(f"{name} 必须包含至少一个有限数值")
    return array


def _finite_nonnegative(value: float | int | None, name: str, *, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise MetricError(f"{name} 必须是有限数值")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise MetricError(f"{name} 必须是非负有限数值")
    return result


def standardized_crse(estimate: npt.ArrayLike, truth: npt.ArrayLike) -> float:
    """返回标准化空间的累计平方误差平方根。"""

    estimate_array = _finite_array(estimate, "estimate")
    truth_array = _finite_array(truth, "truth")
    if estimate_array.shape != truth_array.shape:
        raise MetricError("estimate 和 truth 的形状必须一致")
    return float(np.sqrt(np.sum(np.square(estimate_array - truth_array), dtype=np.float64)))


def crse_ratio(bucket_crse: float, exact_crse: float) -> float | None:
    """返回分桶相对精确版本的 cRSE 比值，零分母为不可计算。"""

    numerator = _finite_nonnegative(bucket_crse, "bucket_crse")
    denominator = _finite_nonnegative(exact_crse, "exact_crse")
    assert numerator is not None and denominator is not None
    return None if denominator == 0.0 else numerator / denominator


def average_ranks(values: npt.ArrayLike) -> tuple[float, ...]:
    """用平均并列秩生成可写入制品的 Worker 排名。"""

    array = _finite_array(values, "ranking values")
    if array.ndim != 1:
        raise MetricError("ranking values 必须是一维")
    return tuple(float(value) for value in rankdata(array, method="average"))


def worker_spearman(exact_values: npt.ArrayLike, bucket_values: npt.ArrayLike) -> float | None:
    """按平均并列秩计算 Spearman；常量或不足两名 Worker 返回 null。"""

    exact = _finite_array(exact_values, "exact_values")
    bucket = _finite_array(bucket_values, "bucket_values")
    if exact.ndim != 1 or bucket.ndim != 1 or exact.shape != bucket.shape:
        raise MetricError("两组 Worker 排名输入必须为同长度一维数组")
    if exact.size < 2:
        return None
    exact_ranks = np.asarray(average_ranks(exact), dtype=np.float64)
    bucket_ranks = np.asarray(average_ranks(bucket), dtype=np.float64)
    if np.all(exact_ranks == exact_ranks[0]) or np.all(bucket_ranks == bucket_ranks[0]):
        return None
    coefficient = float(np.corrcoef(exact_ranks, bucket_ranks)[0, 1])
    return coefficient if math.isfinite(coefficient) else None


def malicious_weight_share(weights: npt.ArrayLike, malicious: npt.ArrayLike) -> float | None:
    """返回恶意身份最终权重之和，占零总权重时为不可计算。"""

    values = _finite_array(weights, "weights")
    mask = np.asarray(malicious, dtype=np.bool_)
    if values.ndim != 1 or mask.ndim != 1 or values.shape != mask.shape:
        raise MetricError("weights 和 malicious 必须是同长度一维数组")
    if np.any(values < 0):
        raise MetricError("weights 不可为负")
    total = math.fsum(float(value) for value in values)
    malicious_total = math.fsum(float(value) for value in values[mask])
    return None if total == 0.0 else malicious_total / total


def proof_acceptance_rate(verified: npt.ArrayLike) -> float:
    """返回报告证明接纳率。"""

    values = np.asarray(verified, dtype=np.bool_)
    if not values.size:
        raise MetricError("verified 不能为空")
    return float(np.mean(values))


def invalid_round_rate(valid_updates: npt.ArrayLike) -> float:
    """返回无效更新轮次占比。"""

    values = np.asarray(valid_updates, dtype=np.bool_)
    if not values.size:
        raise MetricError("valid_updates 不能为空")
    return float(1.0 - np.mean(values))


@dataclass(frozen=True)
class RoundMetrics:
    """一轮分桶与精确残差参考的可审计记录。"""

    dataset: str
    attack: str
    trial_id: int
    fold: int
    block: int
    round_index: int
    nominal_malicious_ratio: float
    actual_malicious_ratio: float
    proof_mode: str
    exact_crse: float
    bucket_crse: float
    crse_ratio: float | None
    exact_worker_ranks: tuple[float, ...]
    proxy_worker_ranks: tuple[float, ...]
    spearman: float | None
    malicious_weight_share: float | None
    proof_acceptance_rate: float
    valid_update: bool

    def __post_init__(self) -> None:
        _validate_common(self.dataset, self.attack, self.trial_id, self.fold, self.block, self.proof_mode)
        if self.round_index < 0:
            raise MetricError("round_index 必须非负")
        _validate_ratios(self.nominal_malicious_ratio, self.actual_malicious_ratio)
        for name in ("exact_crse", "bucket_crse", "proof_acceptance_rate"):
            _finite_nonnegative(getattr(self, name), name)
        if not 0 <= self.proof_acceptance_rate <= 1:
            raise MetricError("proof_acceptance_rate 必须位于 [0, 1]")
        _finite_nonnegative(self.crse_ratio, "crse_ratio", allow_none=True)
        _finite_nonnegative(self.malicious_weight_share, "malicious_weight_share", allow_none=True)
        if self.malicious_weight_share is not None and self.malicious_weight_share > 1:
            raise MetricError("malicious_weight_share 必须位于 [0, 1]")
        if self.spearman is not None and (not math.isfinite(self.spearman) or not -1 <= self.spearman <= 1):
            raise MetricError("spearman 必须位于 [-1, 1] 或为 null")
        if len(self.exact_worker_ranks) != len(self.proxy_worker_ranks):
            raise MetricError("两组 Worker 排名长度必须一致")
        if any(not math.isfinite(value) for value in self.exact_worker_ranks + self.proxy_worker_ranks):
            raise MetricError("Worker 排名必须有限")

    def as_record(self) -> dict[str, Any]:
        """返回按固定字段顺序组织的 CSV/JSON 记录。"""

        return _record_in_order(asdict(self), ROUND_METRIC_FIELDS)


@dataclass(frozen=True)
class TrialMetrics:
    """一个 15 轮试验块的汇总指标。"""

    dataset: str
    attack: str
    trial_id: int
    fold: int
    block: int
    nominal_malicious_ratio: float
    actual_malicious_ratio: float
    proof_mode: str
    exact_crse: float
    bucket_crse: float
    crse_ratio: float | None
    mean_spearman: float | None
    malicious_weight_share: float | None
    proof_acceptance_rate: float
    invalid_round_rate: float
    uncalculable_spearman_count: int
    uncalculable_ratio_count: int

    def __post_init__(self) -> None:
        _validate_common(self.dataset, self.attack, self.trial_id, self.fold, self.block, self.proof_mode)
        _validate_ratios(self.nominal_malicious_ratio, self.actual_malicious_ratio)
        for name in ("exact_crse", "bucket_crse", "proof_acceptance_rate", "invalid_round_rate"):
            _finite_nonnegative(getattr(self, name), name)
        if self.proof_acceptance_rate > 1 or self.invalid_round_rate > 1:
            raise MetricError("比例指标必须位于 [0, 1]")
        _finite_nonnegative(self.crse_ratio, "crse_ratio", allow_none=True)
        _finite_nonnegative(self.malicious_weight_share, "malicious_weight_share", allow_none=True)
        if self.malicious_weight_share is not None and self.malicious_weight_share > 1:
            raise MetricError("malicious_weight_share 必须位于 [0, 1]")
        if self.mean_spearman is not None and (
            not math.isfinite(self.mean_spearman) or not -1 <= self.mean_spearman <= 1
        ):
            raise MetricError("mean_spearman 必须位于 [-1, 1] 或为 null")
        if self.uncalculable_spearman_count < 0 or self.uncalculable_ratio_count < 0:
            raise MetricError("不可计算计数不可为负")

    def as_record(self) -> dict[str, Any]:
        """返回按固定字段顺序组织的 CSV/JSON 记录。"""

        return _record_in_order(asdict(self), TRIAL_METRIC_FIELDS)


@dataclass(frozen=True)
class SummaryMetrics:
    """按试验编号配对的统计摘要。"""

    dataset: str
    attack: str
    nominal_malicious_ratio: float
    proof_mode: str
    metric: str
    trial_count: int
    uncalculable_count: int
    mean: float | None
    sample_std: float | None
    ci95_lower: float | None
    ci95_upper: float | None
    bootstrap_resamples: int
    bootstrap_seed: int

    def __post_init__(self) -> None:
        if not self.dataset or not self.attack or not self.proof_mode or not self.metric:
            raise MetricError("摘要标识字段不能为空")
        _validate_ratios(self.nominal_malicious_ratio, self.nominal_malicious_ratio)
        if self.trial_count < 0 or self.uncalculable_count < 0 or self.bootstrap_resamples <= 0:
            raise MetricError("摘要计数非法")
        for name in ("mean", "sample_std", "ci95_lower", "ci95_upper"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise MetricError(f"{name} 必须有限或为 null")

    def as_record(self) -> dict[str, Any]:
        """返回按固定字段顺序组织的 CSV/JSON 记录。"""

        return _record_in_order(asdict(self), SUMMARY_FIELDS)


def _validate_common(dataset: str, attack: str, trial_id: int, fold: int, block: int, proof_mode: str) -> None:
    if not dataset or not attack or not proof_mode:
        raise MetricError("dataset、attack 和 proof_mode 不能为空")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (trial_id, fold, block)):
        raise MetricError("trial_id、fold 和 block 必须为非负整数")


def _validate_ratios(nominal: float, actual: float) -> None:
    for name, value in (("nominal_malicious_ratio", nominal), ("actual_malicious_ratio", actual)):
        checked = _finite_nonnegative(value, name)
        assert checked is not None
        if checked > 1:
            raise MetricError(f"{name} 必须位于 [0, 1]")


def _record_in_order(raw: dict[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    return {field: raw[field] for field in fields}


def bca_interval(
    values: Sequence[float],
    *,
    resamples: int = 10_000,
    seed: int = 20260901,
) -> tuple[float | None, float | None]:
    """对样本均值计算确定性的百分位调整 BCa 95% 区间。"""

    sample = _finite_array(values, "values").reshape(-1)
    if sample.size < 2 or resamples <= 0:
        return None, None
    observed = float(np.mean(sample))
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, sample.size, size=(resamples, sample.size), endpoint=False)
    bootstrap = np.mean(sample[indices], axis=1)
    less = float(np.mean(bootstrap < observed))
    # 正态分位点在端点无定义，采用 BCa 通行的有限夹紧处理。
    epsilon = 1.0 / (2.0 * resamples)
    z0 = NORMAL.inv_cdf(min(1.0 - epsilon, max(epsilon, less)))
    jackknife = np.asarray([float(np.mean(np.delete(sample, index))) for index in range(sample.size)], dtype=np.float64)
    deviations = np.mean(jackknife) - jackknife
    denominator = 6.0 * float(np.sum(np.square(deviations)) ** 1.5)
    acceleration = 0.0 if denominator == 0.0 else float(np.sum(np.power(deviations, 3)) / denominator)

    def adjusted(alpha: float) -> float:
        z_alpha = NORMAL.inv_cdf(alpha)
        denominator_value = 1.0 - acceleration * (z0 + z_alpha)
        if denominator_value == 0.0:
            return alpha
        return min(1.0, max(0.0, NORMAL.cdf(z0 + (z0 + z_alpha) / denominator_value)))

    lower_probability = adjusted(0.025)
    upper_probability = adjusted(0.975)
    return float(np.quantile(bootstrap, lower_probability)), float(np.quantile(bootstrap, upper_probability))


def summarize_values(
    *,
    dataset: str,
    attack: str,
    nominal_malicious_ratio: float,
    proof_mode: str,
    metric: str,
    values: Iterable[float | None],
    resamples: int = 10_000,
    seed: int = 20260901,
) -> SummaryMetrics:
    """汇总一个明确配对单位上的指标序列。"""

    raw = tuple(values)
    usable = [float(value) for value in raw if value is not None]
    if any(not math.isfinite(value) for value in usable):
        raise MetricError("汇总值包含非有限数")
    mean = float(np.mean(usable)) if usable else None
    std = float(np.std(usable, ddof=1)) if len(usable) >= 2 else None
    lower, upper = bca_interval(usable, resamples=resamples, seed=seed) if usable else (None, None)
    return SummaryMetrics(
        dataset=dataset,
        attack=attack,
        nominal_malicious_ratio=nominal_malicious_ratio,
        proof_mode=proof_mode,
        metric=metric,
        trial_count=len(raw),
        uncalculable_count=len(raw) - len(usable),
        mean=mean,
        sample_std=std,
        ci95_lower=lower,
        ci95_upper=upper,
        bootstrap_resamples=resamples,
        bootstrap_seed=seed,
    )


def summarize_trial_metrics(
    records: Sequence[TrialMetrics], *, resamples: int = 10_000, seed: int = 20260901
) -> tuple[SummaryMetrics, ...]:
    """按数据集、攻击、比例和证明模式汇总 trial 记录。"""

    groups: dict[tuple[str, str, float, str], list[TrialMetrics]] = {}
    for record in records:
        key = (record.dataset, record.attack, record.nominal_malicious_ratio, record.proof_mode)
        groups.setdefault(key, []).append(record)
    output: list[SummaryMetrics] = []
    fields = (
        "exact_crse",
        "bucket_crse",
        "crse_ratio",
        "mean_spearman",
        "malicious_weight_share",
        "proof_acceptance_rate",
        "invalid_round_rate",
    )
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda record: record.trial_id)
        for offset, field in enumerate(fields):
            output.append(
                summarize_values(
                    dataset=key[0],
                    attack=key[1],
                    nominal_malicious_ratio=key[2],
                    proof_mode=key[3],
                    metric=field,
                    values=(getattr(record, field) for record in group),
                    resamples=resamples,
                    seed=seed + offset,
                )
            )
    return tuple(output)
