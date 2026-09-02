"""论文定义的 PP-CH 两阶段筛选与滑动窗口状态机。"""

from __future__ import annotations

import copy
import math
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

FloatTuple = tuple[float, ...]
BoolTuple = tuple[bool, ...]
History = tuple[FloatTuple, ...]


def _require_real(value: object, name: str) -> float:
    """返回可计算的有限实数，并拒绝布尔值等隐式数值。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} 必须是有限实数")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} 必须是有限实数") from error
    if not math.isfinite(converted):
        raise ValueError(f"{name} 必须是有限实数")
    return converted


def _require_positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} 必须是正整数")
    if value <= 0:
        raise ValueError(f"{name} 必须是正整数")
    return value


@dataclass(frozen=True)
class PPCHConfig:
    """PP-CH 与历史滑窗的论文参数。"""

    window_length: int = 5
    decay: float = 0.8
    alpha: float = 1
    beta: float = 1
    cold_start: float = 0.2
    cook_k: float = 6
    minimum_scale: float = 1.5
    effective_parameter_ratio: float = 0.6
    minimum_leverage: float = 0.01
    maximum_leverage: float = 0.25
    leverage_exponent: float = 1
    epsilon: float = 1e-6
    hampel_z_max: float = 12
    hampel_a: float = 1.5
    hampel_b: float = 3
    hampel_c: float = 4.5

    def __post_init__(self) -> None:
        _require_positive_integer(self.window_length, "window_length")

        numeric_fields = (
            ("decay", self.decay),
            ("alpha", self.alpha),
            ("beta", self.beta),
            ("cold_start", self.cold_start),
            ("cook_k", self.cook_k),
            ("minimum_scale", self.minimum_scale),
            ("effective_parameter_ratio", self.effective_parameter_ratio),
            ("minimum_leverage", self.minimum_leverage),
            ("maximum_leverage", self.maximum_leverage),
            ("leverage_exponent", self.leverage_exponent),
            ("epsilon", self.epsilon),
            ("hampel_z_max", self.hampel_z_max),
            ("hampel_a", self.hampel_a),
            ("hampel_b", self.hampel_b),
            ("hampel_c", self.hampel_c),
        )
        values = {name: _require_real(value, name) for name, value in numeric_fields}

        if not 0.0 < values["decay"] <= 1.0:
            raise ValueError("decay 必须位于 (0, 1]")
        if values["alpha"] <= 0.0 or values["beta"] <= 0.0:
            raise ValueError("alpha 和 beta 必须为正数")
        if not 0.0 <= values["cold_start"] <= 1.0:
            raise ValueError("cold_start 必须位于 [0, 1]")
        if values["cook_k"] <= 0.0:
            raise ValueError("cook_k 必须为正数")
        if values["minimum_scale"] <= 0.0:
            raise ValueError("minimum_scale 必须为正数")
        if values["effective_parameter_ratio"] <= 0.0:
            raise ValueError("effective_parameter_ratio 必须为正数")
        if not (0.0 < values["minimum_leverage"] <= values["maximum_leverage"] < 1.0):
            raise ValueError("杠杆上下界必须满足 0 < minimum <= maximum < 1")
        if values["leverage_exponent"] < 0.0:
            raise ValueError("leverage_exponent 必须为非负数")
        if values["epsilon"] <= 0.0:
            raise ValueError("epsilon 必须为正数")
        if not (0.0 < values["hampel_a"] < values["hampel_b"] < values["hampel_c"] < values["hampel_z_max"]):
            raise ValueError("Hampel 参数必须满足 0 < a < b < c < z_max")


@dataclass(frozen=True)
class PPCHDecision:
    """一次只读评估的不可变结果和诊断量。"""

    final_weights: FloatTuple
    raw_weights: FloatTuple
    sliding_weights: FloatTuple
    hampel_weights: FloatTuple
    cook_distances: FloatTuple
    leverage: FloatTuple
    survivors: BoolTuple
    valid_update: bool
    stage1_center: float | None
    stage1_scale: float | None
    stage2_center: float | None
    stage2_scale: float | None

    def __post_init__(self) -> None:
        vector_names = (
            "final_weights",
            "raw_weights",
            "sliding_weights",
            "hampel_weights",
            "cook_distances",
            "leverage",
            "survivors",
        )
        vectors = tuple(getattr(self, name) for name in vector_names)
        if any(not isinstance(vector, tuple) for vector in vectors):
            raise TypeError("PPCHDecision 的向量字段必须是 tuple")
        lengths = {len(vector) for vector in vectors}
        if len(lengths) != 1:
            raise ValueError("PPCHDecision 的向量维度必须一致")
        if any(type(value) is not bool for value in self.survivors):
            raise TypeError("survivors 必须是布尔 tuple")
        if type(self.valid_update) is not bool:
            raise TypeError("valid_update 必须是布尔值")

        bounded_vectors = (
            ("final_weights", self.final_weights, 0.0, 1.0),
            ("raw_weights", self.raw_weights, 0.0, 1.0),
            ("sliding_weights", self.sliding_weights, 0.0, 1.0),
            ("hampel_weights", self.hampel_weights, 0.0, 1.0),
            ("leverage", self.leverage, 0.0, 1.0),
        )
        for name, vector, lower, upper in bounded_vectors:
            for value in vector:
                numeric = _require_real(value, name)
                if not lower <= numeric <= upper:
                    raise ValueError(f"{name} 的元素必须位于 [{lower}, {upper}]")
        for value in self.cook_distances:
            numeric = _require_real(value, "cook_distances")
            if math.isnan(numeric) or numeric < 0.0:
                raise ValueError("cook_distances 的元素必须是非负实数")

        stage_pairs = (
            ("stage1", self.stage1_center, self.stage1_scale),
            ("stage2", self.stage2_center, self.stage2_scale),
        )
        for name, center, scale in stage_pairs:
            if (center is None) != (scale is None):
                raise ValueError(f"{name} 的中心和尺度必须同时存在或同时为空")
            if center is not None:
                if _require_real(center, f"{name}_center") < 0.0:
                    raise ValueError(f"{name}_center 必须为非负数")
                if _require_real(scale, f"{name}_scale") <= 0.0:
                    raise ValueError(f"{name}_scale 必须为正数")
        if self.stage2_center is not None and self.stage1_center is None:
            raise ValueError("Stage 2 诊断量不能脱离 Stage 1 存在")

        raw_total = math.fsum(self.raw_weights)
        final_total = math.fsum(self.final_weights)
        if self.valid_update:
            if raw_total <= 0.0 or not math.isclose(final_total, 1.0, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("有效决策必须具有正原始权重和归一化最终权重")
        elif final_total != 0.0:
            raise ValueError("无效决策的最终权重必须全为零")


class PPCHState:
    """维护前轮权重和最近 ``L`` 个历史贡献槽。"""

    def __init__(
        self,
        worker_count: int,
        config: PPCHConfig | None = None,
    ) -> None:
        self._worker_count = _require_positive_integer(worker_count, "worker_count")
        if config is None:
            config = PPCHConfig()
        if not isinstance(config, PPCHConfig):
            raise TypeError("config 必须是 PPCHConfig")
        self._config = config
        initial_weight = 1.0 / self._worker_count
        self._previous_weights: FloatTuple = (initial_weight,) * self._worker_count
        self._history: deque[FloatTuple] = deque(maxlen=config.window_length)
        self._round_count = 0

    @property
    def worker_count(self) -> int:
        return self._worker_count

    @property
    def config(self) -> PPCHConfig:
        return self._config

    @property
    def previous_weights(self) -> FloatTuple:
        return self._previous_weights

    @property
    def history(self) -> History:
        return tuple(self._history)

    @property
    def round_count(self) -> int:
        return self._round_count

    def clone(self) -> PPCHState:
        """返回可独立推进的状态深拷贝，供 Fang 搜索分支使用。"""

        return copy.deepcopy(self)

    def preview(
        self,
        scores: Sequence[float],
        present: Sequence[bool],
        verified: Sequence[bool],
    ) -> PPCHDecision:
        """基于当前历史评估一轮，且完全不修改状态。"""

        validated = self._validate_round_inputs(scores, present, verified)
        return self._preview_validated(*validated)

    def commit(
        self,
        scores: Sequence[float],
        present: Sequence[bool],
        verified: Sequence[bool],
        decision: PPCHDecision,
    ) -> None:
        """校验并提交一次预览结果，推进一个历史槽。"""

        validated = self._validate_round_inputs(scores, present, verified)
        if not isinstance(decision, PPCHDecision):
            raise TypeError("decision 必须是 PPCHDecision")
        expected = self._preview_validated(*validated)
        if decision != expected:
            raise ValueError("decision 与当前状态或本轮输入不匹配")
        self._commit_validated(*validated, decision)

    def update(
        self,
        scores: Sequence[float],
        present: Sequence[bool],
        verified: Sequence[bool],
    ) -> PPCHDecision:
        """依次执行预览和提交，并返回本轮不可变决策。"""

        validated = self._validate_round_inputs(scores, present, verified)
        decision = self._preview_validated(*validated)
        self._commit_validated(*validated, decision)
        return decision

    def _validate_round_inputs(
        self,
        scores: Iterable[object],
        present: Iterable[bool],
        verified: Iterable[bool],
    ) -> tuple[FloatTuple, BoolTuple, BoolTuple]:
        score_items = self._materialize(scores, "scores")
        present_items = self._materialize(present, "present")
        verified_items = self._materialize(verified, "verified")

        validated_scores = tuple(self._validate_score(value, index) for index, value in enumerate(score_items))
        validated_present = self._validate_mask(present_items, "present")
        validated_verified = self._validate_mask(verified_items, "verified")
        return validated_scores, validated_present, validated_verified

    def _materialize(self, values: Iterable[object], name: str) -> tuple[object, ...]:
        if isinstance(values, (str, bytes, bytearray)):
            raise TypeError(f"{name} 必须是一维序列")
        try:
            items = tuple(values)
        except TypeError as error:
            raise TypeError(f"{name} 必须是一维序列") from error
        if len(items) != self._worker_count:
            raise ValueError(f"{name} 长度必须为 worker_count={self._worker_count}")
        return items

    @staticmethod
    def _validate_score(value: object, index: int) -> float:
        score = _require_real(value, f"scores[{index}]")
        if score < 0.0:
            raise ValueError(f"scores[{index}] 必须为非负数")
        return score

    @staticmethod
    def _validate_mask(values: tuple[object, ...], name: str) -> BoolTuple:
        if any(type(value) is not bool for value in values):
            raise TypeError(f"{name} 的元素必须是 bool")
        return tuple(bool(value) for value in values)

    def _preview_validated(
        self,
        scores: FloatTuple,
        present: BoolTuple,
        verified: BoolTuple,
    ) -> PPCHDecision:
        sliding = self._sliding_weights()
        accepted = tuple(index for index in range(self._worker_count) if present[index] and verified[index])
        if not accepted:
            return self._empty_decision(sliding)

        config = self._config
        leverage = [0.0] * self._worker_count
        cook_distances = [0.0] * self._worker_count
        survivors = [False] * self._worker_count
        hampel_weights = [0.0] * self._worker_count
        raw_weights = [0.0] * self._worker_count

        previous_total = math.fsum(self._previous_weights[index] for index in accepted)
        influence_denominator = previous_total + len(accepted) * config.epsilon
        for index in accepted:
            share = (self._previous_weights[index] + config.epsilon) / influence_denominator
            leverage[index] = min(
                config.maximum_leverage,
                max(
                    config.minimum_leverage,
                    config.effective_parameter_ratio * share,
                ),
            )

        stage1_center = self._median(scores[index] for index in accepted)
        stage1_scale = self._robust_scale(abs(scores[index] - stage1_center) for index in accepted)
        cook_threshold = config.cook_k / len(accepted)
        for index in accepted:
            right_tail = max(0.0, (scores[index] - stage1_center) / stage1_scale)
            leverage_ratio = leverage[index] / (1.0 - leverage[index])
            cook_distances[index] = self._cook_distance(
                right_tail,
                leverage_ratio,
                config.leverage_exponent,
            )
            survivors[index] = cook_distances[index] <= cook_threshold

        survivor_indices = tuple(index for index in accepted if survivors[index])
        if not survivor_indices:
            return self._decision(
                final_weights=(0.0,) * self._worker_count,
                raw_weights=tuple(raw_weights),
                sliding_weights=sliding,
                hampel_weights=tuple(hampel_weights),
                cook_distances=tuple(cook_distances),
                leverage=tuple(leverage),
                survivors=tuple(survivors),
                valid_update=False,
                stage1_center=stage1_center,
                stage1_scale=stage1_scale,
                stage2_center=None,
                stage2_scale=None,
            )

        stage2_center = self._median(scores[index] for index in survivor_indices)
        stage2_scale = self._robust_scale(abs(scores[index] - stage2_center) for index in survivor_indices)
        for index in survivor_indices:
            z_score = min(
                config.hampel_z_max,
                max(0.0, (scores[index] - stage2_center) / stage2_scale),
            )
            hampel_weights[index] = self._hampel_weight(z_score)
            raw_weights[index] = (sliding[index] ** config.alpha) * (hampel_weights[index] ** config.beta)

        raw_total = math.fsum(raw_weights)
        valid_update = raw_total > 0.0
        if valid_update:
            final_weights = tuple(weight / raw_total for weight in raw_weights)
        else:
            final_weights = (0.0,) * self._worker_count

        return self._decision(
            final_weights=final_weights,
            raw_weights=tuple(raw_weights),
            sliding_weights=sliding,
            hampel_weights=tuple(hampel_weights),
            cook_distances=tuple(cook_distances),
            leverage=tuple(leverage),
            survivors=tuple(survivors),
            valid_update=valid_update,
            stage1_center=stage1_center,
            stage1_scale=stage1_scale,
            stage2_center=stage2_center,
            stage2_scale=stage2_scale,
        )

    def _sliding_weights(self) -> FloatTuple:
        if not self._history:
            return (float(self._config.cold_start),) * self._worker_count

        numerator = [0.0] * self._worker_count
        denominator_terms = []
        # 最新槽的指数为零，越旧的槽按 decay 继续衰减。
        for age, slot in enumerate(reversed(self._history)):
            decay_weight = self._config.decay**age
            denominator_terms.append(decay_weight)
            for index, contribution in enumerate(slot):
                numerator[index] += decay_weight * contribution
        denominator = math.fsum(denominator_terms)
        return tuple(value / denominator for value in numerator)

    def _commit_validated(
        self,
        scores: FloatTuple,
        present: BoolTuple,
        verified: BoolTuple,
        decision: PPCHDecision,
    ) -> None:
        if decision.valid_update:
            # 历史只描述证明有效且到达的报告，不受 Hampel 最终筛选影响。
            contribution = tuple(
                1.0 / (1.0 + scores[index]) if present[index] and verified[index] else 0.0
                for index in range(self._worker_count)
            )
            self._previous_weights = decision.final_weights
        else:
            contribution = (0.0,) * self._worker_count
        self._history.append(contribution)
        self._round_count += 1

    def _empty_decision(self, sliding: FloatTuple) -> PPCHDecision:
        zeros = (0.0,) * self._worker_count
        return self._decision(
            final_weights=zeros,
            raw_weights=zeros,
            sliding_weights=sliding,
            hampel_weights=zeros,
            cook_distances=zeros,
            leverage=zeros,
            survivors=(False,) * self._worker_count,
            valid_update=False,
            stage1_center=None,
            stage1_scale=None,
            stage2_center=None,
            stage2_scale=None,
        )

    @staticmethod
    def _decision(**kwargs: object) -> PPCHDecision:
        return PPCHDecision(**kwargs)  # type: ignore[arg-type]

    @staticmethod
    def _median(values: Iterable[float]) -> float:
        ordered = sorted(values)
        count = len(ordered)
        midpoint = count // 2
        if count % 2:
            return ordered[midpoint]
        lower = ordered[midpoint - 1]
        upper = ordered[midpoint]
        return lower + (upper - lower) / 2.0

    def _robust_scale(self, deviations: Iterable[float]) -> float:
        mad = self._median(deviations)
        return max(1.4826 * mad, self._config.minimum_scale)

    @staticmethod
    def _cook_distance(
        right_tail: float,
        leverage_ratio: float,
        leverage_exponent: float,
    ) -> float:
        leverage_factor = leverage_ratio**leverage_exponent
        try:
            return float(right_tail * right_tail * leverage_factor)
        except OverflowError:
            return math.inf

    def _hampel_weight(self, z_score: float) -> float:
        config = self._config
        if z_score <= config.hampel_a:
            return 1.0
        if z_score <= config.hampel_b:
            return config.hampel_a / z_score
        if z_score <= config.hampel_c:
            return config.hampel_a * (config.hampel_c - z_score) / ((config.hampel_c - config.hampel_b) * z_score)
        return 0.0


__all__ = ["PPCHConfig", "PPCHDecision", "PPCHState"]
