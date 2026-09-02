"""五类半合成投毒攻击和状态隔离的 Fang 优化。"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from .scenario import (
    BaseScenario,
    BoolArray,
    FloatArray,
    ScenarioError,
    _readonly_bool,
    _readonly_float,
)

ATTACKS = ("bias", "drift", "spike", "flip", "fang")


@dataclass(frozen=True)
class AttackParameters:
    """五类主攻击的冻结论文参数。"""

    bias_offset: float = 1.8
    drift_per_round: float = 0.13
    spike_upper_probability: float = 0.85
    fang_step_size: float = 0.65
    fang_maximum_steps: int = 12
    fang_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        real_values = (
            ("bias_offset", self.bias_offset),
            ("drift_per_round", self.drift_per_round),
            ("spike_upper_probability", self.spike_upper_probability),
            ("fang_step_size", self.fang_step_size),
            ("fang_tolerance", self.fang_tolerance),
        )
        for name, value in real_values:
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(float(value)):
                raise ScenarioError(f"{name} 必须是有限实数")
        if self.bias_offset < 0.0 or self.drift_per_round < 0.0:
            raise ScenarioError("bias_offset 和 drift_per_round 必须非负")
        if not 0.0 <= self.spike_upper_probability <= 1.0:
            raise ScenarioError("spike_upper_probability 必须位于 [0, 1]")
        if self.fang_step_size <= 0.0 or self.fang_tolerance < 0.0:
            raise ScenarioError("Fang 步长或容差非法")
        if isinstance(self.fang_maximum_steps, bool) or not isinstance(self.fang_maximum_steps, int):
            raise ScenarioError("fang_maximum_steps 必须是正整数")
        if self.fang_maximum_steps <= 0:
            raise ScenarioError("fang_maximum_steps 必须是正整数")

    def as_manifest_fragment(self) -> dict[str, float | int]:
        """返回会影响攻击报告的完整冻结参数。"""

        return {
            "bias_offset": float(self.bias_offset),
            "drift_per_round": float(self.drift_per_round),
            "spike_upper_probability": float(self.spike_upper_probability),
            "fang_step_size": float(self.fang_step_size),
            "fang_maximum_steps": self.fang_maximum_steps,
            "fang_tolerance": float(self.fang_tolerance),
        }


DEFAULT_ATTACK_PARAMETERS = AttackParameters()


@dataclass(frozen=True)
class AggregationPreview:
    """Fang 优化所需的攻击标签不可见聚合预览。"""

    estimate: FloatArray
    weights: FloatArray
    valid_update: bool

    def __post_init__(self) -> None:
        estimate = _readonly_float(self.estimate, "estimate")
        weights = _readonly_float(self.weights, "weights")
        if estimate.ndim != 1 or weights.ndim != 1 or not np.all(np.isfinite(estimate)):
            raise ScenarioError("AggregationPreview 必须含有限一维 estimate 和 weights")
        if np.any(weights < 0):
            raise ScenarioError("AggregationPreview weights 不可为负")
        object.__setattr__(self, "estimate", estimate)
        object.__setattr__(self, "weights", weights)


@runtime_checkable
class FangEvaluator(Protocol):
    """Fang 所需的最小预览接口，刻意不接收 attack 标签。"""

    def preview(self, reports: FloatArray, present: BoolArray, state: Any) -> AggregationPreview:
        """计算当前报告的更新和最终权重。"""


@dataclass(frozen=True)
class AttackScenario:
    """固定恶意身份与一类攻击产生的完整报告张量。"""

    base: BaseScenario
    attack: str
    malicious_mask: BoolArray
    reports: FloatArray

    def __post_init__(self) -> None:
        if self.attack not in ATTACKS:
            raise ScenarioError(f"不支持攻击：{self.attack}")
        mask = _readonly_bool(self.malicious_mask, "malicious_mask")
        reports = _readonly_float(self.reports, "reports")
        if mask.shape != (self.base.worker_count,) or reports.shape != self.base.honest_reports.shape:
            raise ScenarioError("攻击报告形状不匹配")
        if np.any(~np.isfinite(reports[self.base.present])) or np.any(~np.isnan(reports[~self.base.present])):
            raise ScenarioError("攻击报告必须保留稀疏缺失语义")
        object.__setattr__(self, "malicious_mask", mask)
        object.__setattr__(self, "reports", reports)


def _project(values: FloatArray, base: BaseScenario) -> FloatArray:
    lower = np.asarray([domain[0] for domain in base.standardized_domains])
    upper = np.asarray([domain[1] for domain in base.standardized_domains])
    return np.clip(values, lower, upper).astype(np.float64, copy=False)


def _malicious_noise(base: BaseScenario, attack: str) -> FloatArray:
    rng = np.random.default_rng(base.attack_seed(attack))
    return rng.normal(0.0, base.sigma_m, size=base.honest_reports.shape).astype(np.float64)


def _non_fang_reports(
    base: BaseScenario,
    attack: str,
    mask: BoolArray,
    parameters: AttackParameters,
) -> FloatArray:
    reports = np.array(base.honest_reports, dtype=np.float64, copy=True)
    noise = _malicious_noise(base, attack)
    truth = base.truth[:, None, :]
    malicious = np.broadcast_to(mask[None, :, None], reports.shape)
    if attack == "bias":
        candidate = truth + parameters.bias_offset + noise
    elif attack == "drift":
        time_index = np.arange(base.round_count, dtype=np.float64)[:, None, None]
        candidate = truth + parameters.bias_offset + parameters.drift_per_round * time_index + noise
    elif attack == "spike":
        rng = np.random.default_rng(base.attack_seed(attack))
        upper = np.asarray([domain[1] for domain in base.standardized_domains])
        lower = np.asarray([domain[0] for domain in base.standardized_domains])
        candidate = np.where(rng.random(reports.shape) < parameters.spike_upper_probability, upper, lower)
    elif attack == "flip":
        candidate = truth - base.honest_epsilon + noise
    else:
        raise ScenarioError(f"不支持攻击：{attack}")
    candidate = _project(np.asarray(candidate, dtype=np.float64), base)
    reports[malicious] = candidate[malicious]
    reports[~base.present] = np.nan
    return reports


def _clone_state(state: Any) -> Any:
    clone_method = getattr(state, "clone", None)
    return clone_method() if callable(clone_method) else copy.deepcopy(state)


def optimize_fang_round(
    *,
    reports: npt.ArrayLike,
    present: npt.ArrayLike,
    truth: npt.ArrayLike,
    malicious_mask: npt.ArrayLike,
    standardized_domains: tuple[tuple[float, float], ...],
    evaluator: FangEvaluator,
    state: Any,
    step_size: float = 0.65,
    maximum_steps: int = 12,
    tolerance: float = 1e-6,
) -> FloatArray:
    """在状态副本上固定权重求梯度，返回单轮 Fang 投毒报告。

    正式 PP-CH 状态不在此函数内提交。每次预览都得到独立状态副本，因而
    优化过程不能将历史推进到下一轮。
    """

    report_array = np.array(reports, dtype=np.float64, copy=True)
    present_array = np.asarray(present, dtype=np.bool_)
    truth_array = np.asarray(truth, dtype=np.float64)
    mask = np.asarray(malicious_mask, dtype=np.bool_)
    if (
        report_array.ndim != 2
        or truth_array.shape != (report_array.shape[1],)
        or present_array.shape != (report_array.shape[0],)
    ):
        raise ScenarioError("Fang 单轮输入形状不匹配")
    if mask.shape != present_array.shape or len(standardized_domains) != report_array.shape[1]:
        raise ScenarioError("Fang 恶意掩码或域维度不匹配")
    if step_size <= 0 or maximum_steps <= 0 or tolerance < 0:
        raise ScenarioError("Fang 优化参数非法")
    if not np.all(np.isfinite(report_array[present_array])) or not np.all(np.isfinite(truth_array)):
        raise ScenarioError("Fang 仅接受有限的存在报告和真值")
    lower = np.asarray([domain[0] for domain in standardized_domains])
    upper = np.asarray([domain[1] for domain in standardized_domains])
    active = mask & present_array
    if not np.any(active):
        report_array[~present_array] = np.nan
        return report_array
    for _ in range(maximum_steps):
        preview = evaluator.preview(report_array, present_array, _clone_state(state))
        if preview.weights.shape != present_array.shape or preview.estimate.shape != truth_array.shape:
            raise ScenarioError("Fang evaluator 返回的形状不匹配")
        total_weight = float(np.sum(preview.weights, dtype=np.float64))
        if not preview.valid_update or total_weight <= 0.0:
            break
        gradient = 2.0 * (preview.estimate - truth_array)[None, :] * preview.weights[:, None] / total_weight
        change = step_size * gradient[active]
        candidate = report_array[active] + change
        candidate = np.clip(candidate, lower, upper)
        max_change = float(np.max(np.abs(candidate - report_array[active])))
        report_array[active] = candidate
        if max_change < tolerance:
            break
    report_array[~present_array] = np.nan
    return report_array


def build_attack_scenario(
    base: BaseScenario,
    attack: str,
    malicious_count: int,
    *,
    parameters: AttackParameters = DEFAULT_ATTACK_PARAMETERS,
    fang_evaluator: FangEvaluator | None = None,
    fang_state: Any = None,
) -> AttackScenario:
    """基于同一基础场景生成一个攻击版本，不提交或修改正式状态。"""

    if attack not in ATTACKS:
        raise ScenarioError(f"不支持攻击：{attack}")
    if not isinstance(parameters, AttackParameters):
        raise ScenarioError("parameters 必须是 AttackParameters")
    mask = base.malicious_mask(malicious_count)
    if attack != "fang":
        return AttackScenario(base, attack, mask, _non_fang_reports(base, attack, mask, parameters))
    if fang_evaluator is None:
        raise ScenarioError("Fang 攻击必须提供不含攻击标签的 evaluator")
    reports = np.array(base.honest_reports, dtype=np.float64, copy=True)
    for round_index in range(base.round_count):
        reports[round_index] = optimize_fang_round(
            reports=reports[round_index],
            present=base.present[round_index],
            truth=base.truth[round_index],
            malicious_mask=mask,
            standardized_domains=base.standardized_domains,
            evaluator=fang_evaluator,
            state=fang_state,
            step_size=parameters.fang_step_size,
            maximum_steps=parameters.fang_maximum_steps,
            tolerance=parameters.fang_tolerance,
        )
    return AttackScenario(base, attack, mask, reports)
