"""攻击不可见的基础场景与稳定随机种子派生。"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


class ScenarioError(ValueError):
    """场景输入未满足固定形状、域或随机性契约。"""


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def stable_seed(namespace: str, *parts: object) -> int:
    """用规范 JSON 与 SHA256 派生跨 Python 版本稳定的 64 位种子。"""

    if not isinstance(namespace, str) or not namespace:
        raise ScenarioError("namespace 必须为非空字符串")
    digest = hashlib.sha256(_canonical_json({"namespace": namespace, "parts": list(parts)})).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


@dataclass(frozen=True)
class TrialSeeds:
    """跨攻击固定的身份和诚实噪声种子，以及攻击无关恶意根种子。"""

    identity_seed: int
    honest_noise_seed: int
    malicious_report_root_seed: int

    def as_manifest_fragment(self) -> dict[str, int]:
        """输出完整派生种子以便制品审计。"""

        return {
            "honest_noise_seed": self.honest_noise_seed,
            "identity_seed": self.identity_seed,
            "malicious_report_root_seed": self.malicious_report_root_seed,
        }


def derive_trial_seeds(dataset: str, fold: int, block: int, trial_id: int) -> TrialSeeds:
    """固定主试验的三个互相隔离随机流。"""

    if not dataset:
        raise ScenarioError("dataset 不能为空")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (fold, block, trial_id)):
        raise ScenarioError("fold、block 和 trial_id 必须为非负整数")
    coordinate = (dataset, fold, block, trial_id)
    return TrialSeeds(
        identity_seed=stable_seed("brptd/identity", *coordinate),
        honest_noise_seed=stable_seed("brptd/honest-noise", *coordinate),
        malicious_report_root_seed=stable_seed("brptd/malicious-report", *coordinate),
    )


def _readonly_float(values: npt.ArrayLike, name: str) -> FloatArray:
    result = np.array(values, dtype=np.float64, copy=True)
    if not result.size:
        raise ScenarioError(f"{name} 不能为空")
    result.setflags(write=False)
    return result


def _readonly_bool(values: npt.ArrayLike, name: str) -> BoolArray:
    result = np.array(values, dtype=np.bool_, copy=True)
    if not result.size:
        raise ScenarioError(f"{name} 不能为空")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class BaseScenario:
    """攻击前固定的潜在真值、诚实报告与身份随机排列。"""

    dataset: str
    worker_ids: tuple[str, ...]
    truth: FloatArray
    present: BoolArray
    standardized_domains: tuple[tuple[float, float], ...]
    sigma_h: FloatArray
    sigma_m: FloatArray
    identity_permutation: tuple[int, ...]
    honest_epsilon: FloatArray
    honest_reports: FloatArray
    seeds: TrialSeeds

    def __post_init__(self) -> None:
        if not self.dataset or not self.worker_ids or len(set(self.worker_ids)) != len(self.worker_ids):
            raise ScenarioError("dataset 与 worker_ids 必须有效")
        truth = _readonly_float(self.truth, "truth")
        present = _readonly_bool(self.present, "present")
        sigma_h = _readonly_float(self.sigma_h, "sigma_h")
        sigma_m = _readonly_float(self.sigma_m, "sigma_m")
        epsilon = _readonly_float(self.honest_epsilon, "honest_epsilon")
        reports = _readonly_float(self.honest_reports, "honest_reports")
        rounds, dimension = truth.shape if truth.ndim == 2 else (0, 0)
        expected_presence = (rounds, len(self.worker_ids))
        expected_reports = (rounds, len(self.worker_ids), dimension)
        if rounds == 0 or dimension == 0 or present.shape != expected_presence:
            raise ScenarioError("truth 和 present 形状不匹配")
        if epsilon.shape != expected_reports or reports.shape != expected_reports:
            raise ScenarioError("噪声和诚实报告形状不匹配")
        if sigma_h.shape != (dimension,) or sigma_m.shape != (dimension,):
            raise ScenarioError("sigma 维度不匹配")
        if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(sigma_h)) or np.any(sigma_h <= 0):
            raise ScenarioError("truth 和 sigma_h 必须有限，且 sigma_h 为正")
        if not np.allclose(sigma_m, 2.0 * sigma_h):
            raise ScenarioError("sigma_m 必须等于 2 * sigma_h")
        if len(self.standardized_domains) != dimension:
            raise ScenarioError("标准化域维度不匹配")
        for index, (lower, upper) in enumerate(self.standardized_domains):
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise ScenarioError(f"standardized_domains[{index}] 非法")
        if tuple(sorted(self.identity_permutation)) != tuple(range(len(self.worker_ids))):
            raise ScenarioError("identity_permutation 必须是 Worker 索引排列")
        if np.any(~np.isfinite(epsilon[present])) or np.any(~np.isfinite(reports[present])):
            raise ScenarioError("存在报告必须为有限值")
        if np.any(~np.isnan(epsilon[~present])) or np.any(~np.isnan(reports[~present])):
            raise ScenarioError("缺席报告和噪声必须保留 NaN")
        object.__setattr__(self, "truth", truth)
        object.__setattr__(self, "present", present)
        object.__setattr__(self, "sigma_h", sigma_h)
        object.__setattr__(self, "sigma_m", sigma_m)
        object.__setattr__(self, "honest_epsilon", epsilon)
        object.__setattr__(self, "honest_reports", reports)

    @property
    def round_count(self) -> int:
        return int(self.truth.shape[0])

    @property
    def worker_count(self) -> int:
        return len(self.worker_ids)

    @property
    def dimension(self) -> int:
        return int(self.truth.shape[1])

    def malicious_mask(self, count: int) -> BoolArray:
        """返回同一身份排列前缀对应的嵌套恶意集合。"""

        if isinstance(count, bool) or not isinstance(count, int) or count < 0 or count > self.worker_count:
            raise ScenarioError("恶意 Worker 数超出范围")
        mask = np.zeros(self.worker_count, dtype=np.bool_)
        mask[list(self.identity_permutation[:count])] = True
        mask.setflags(write=False)
        return mask

    def attack_seed(self, attack: str) -> int:
        """让攻击类型仅影响恶意报告流。"""

        if not attack:
            raise ScenarioError("attack 不能为空")
        return stable_seed("brptd/attack-report", self.seeds.malicious_report_root_seed, attack)

    def as_manifest_fragment(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "domains": [list(domain) for domain in self.standardized_domains],
            "seeds": self.seeds.as_manifest_fragment(),
            "worker_ids": list(self.worker_ids),
        }


def build_base_scenario(
    *,
    dataset: str,
    worker_ids: Sequence[str],
    truth: npt.ArrayLike,
    present: npt.ArrayLike,
    standardized_domains: Sequence[Sequence[float]],
    sigma_h: npt.ArrayLike,
    seeds: TrialSeeds,
) -> BaseScenario:
    """生成共享诚实报告和攻击无关的身份排列。"""

    names = tuple(str(worker_id) for worker_id in worker_ids)
    truth_array = np.asarray(truth, dtype=np.float64)
    present_array = np.asarray(present, dtype=np.bool_)
    sigma = np.asarray(sigma_h, dtype=np.float64)
    if truth_array.ndim != 2 or not truth_array.size or present_array.shape != (truth_array.shape[0], len(names)):
        raise ScenarioError("truth、present 和 worker_ids 形状不匹配")
    if sigma.shape != (truth_array.shape[1],) or np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ScenarioError("sigma_h 必须是正有限向量")
    domain_values: list[tuple[float, float]] = []
    for domain in standardized_domains:
        raw_domain = tuple(float(value) for value in domain)
        if len(raw_domain) != 2 or raw_domain[0] >= raw_domain[1]:
            raise ScenarioError("standardized_domains 非法")
        domain_values.append((raw_domain[0], raw_domain[1]))
    domains = tuple(domain_values)
    if len(domains) != truth_array.shape[1]:
        raise ScenarioError("standardized_domains 维度不匹配")
    if not np.all(np.isfinite(truth_array)):
        raise ScenarioError("truth 必须有限")
    rng_identity = np.random.default_rng(seeds.identity_seed)
    permutation = tuple(int(value) for value in rng_identity.permutation(len(names)))
    rng_honest = np.random.default_rng(seeds.honest_noise_seed)
    epsilon = rng_honest.normal(0.0, sigma, size=(truth_array.shape[0], len(names), truth_array.shape[1]))
    reports = truth_array[:, None, :] + epsilon
    lower = np.asarray([domain[0] for domain in domains])
    upper = np.asarray([domain[1] for domain in domains])
    reports = np.clip(reports, lower, upper)
    epsilon = reports - truth_array[:, None, :]
    reports = np.asarray(reports, dtype=np.float64)
    epsilon = np.asarray(epsilon, dtype=np.float64)
    reports[~present_array] = np.nan
    epsilon[~present_array] = np.nan
    return BaseScenario(
        dataset=dataset,
        worker_ids=names,
        truth=truth_array,
        present=present_array,
        standardized_domains=domains,
        sigma_h=sigma,
        sigma_m=2.0 * sigma,
        identity_permutation=permutation,
        honest_epsilon=epsilon,
        honest_reports=reports,
        seeds=seeds,
    )
