"""只使用训练期数据的面板选择、稳健变换和半合成真值准备。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .models import DataContractError, DatasetSpec, SparsePanel

FloatArray = npt.NDArray[np.float64]


def _indices(indices: Sequence[int], length: int, name: str) -> tuple[int, ...]:
    result = tuple(int(index) for index in indices)
    if not result:
        raise DataContractError(f"{name} 不能为空")
    if any(index < 0 or index >= length for index in result):
        raise DataContractError(f"{name} 含有越界索引")
    if len(set(result)) != len(result):
        raise DataContractError(f"{name} 不可重复")
    return result


def _as_readonly(values: npt.ArrayLike) -> FloatArray:
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _median_absolute_deviation(values: FloatArray, axis: int | None = None) -> FloatArray:
    center = np.nanmedian(values, axis=axis, keepdims=True)
    return np.asarray(np.nanmedian(np.abs(values - center), axis=axis), dtype=np.float64)


@dataclass(frozen=True)
class TrainingTransform:
    """训练区冻结的中心、尺度、域映射及诚实噪声估计。"""

    center: FloatArray
    mad: FloatArray
    scale: FloatArray
    standardized_domains: tuple[tuple[float, float], ...]
    sigma_h: FloatArray
    sigma_m: FloatArray
    selected_worker_ids: tuple[str, ...]
    training_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in ("center", "mad", "scale", "sigma_h", "sigma_m"):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
                raise DataContractError(f"{name} 必须是一维有限数组")
            if name in {"scale", "sigma_h", "sigma_m"} and np.any(values <= 0):
                raise DataContractError(f"{name} 必须严格为正")
            object.__setattr__(self, name, _as_readonly(values))
        if not (self.center.shape == self.mad.shape == self.scale.shape == self.sigma_h.shape == self.sigma_m.shape):
            raise DataContractError("训练变换数组维度不一致")
        if not np.allclose(self.sigma_m, 2.0 * self.sigma_h):
            raise DataContractError("sigma_m 必须严格等于 2 * sigma_h")
        if len(self.standardized_domains) != self.center.size:
            raise DataContractError("standardized_domains 维度不一致")
        for index, domain in enumerate(self.standardized_domains):
            if len(domain) != 2 or not np.isfinite(domain[0]) or not np.isfinite(domain[1]) or domain[0] >= domain[1]:
                raise DataContractError(f"standardized_domains[{index}] 非法")
        if not self.selected_worker_ids or len(set(self.selected_worker_ids)) != len(self.selected_worker_ids):
            raise DataContractError("selected_worker_ids 必须非空且不可重复")
        if not self.training_indices:
            raise DataContractError("training_indices 不能为空")

    def standardize(self, values: npt.ArrayLike) -> FloatArray:
        """在冻结尺度下映射原始值到稳健标准化坐标。"""

        raw = np.asarray(values, dtype=np.float64)
        if raw.shape[-1] != self.center.size:
            raise DataContractError("待标准化数组末维不匹配")
        if not np.all(np.isfinite(raw)):
            raise DataContractError("待标准化数组包含非有限值")
        return np.asarray((raw - self.center) / self.scale, dtype=np.float64)

    def inverse(self, values: npt.ArrayLike) -> FloatArray:
        """把标准化坐标映射回原始空间。"""

        raw = np.asarray(values, dtype=np.float64)
        if raw.shape[-1] != self.center.size:
            raise DataContractError("待逆变换数组末维不匹配")
        if not np.all(np.isfinite(raw)):
            raise DataContractError("待逆变换数组包含非有限值")
        return np.asarray(raw * self.scale + self.center, dtype=np.float64)

    def project_standardized(self, values: npt.ArrayLike) -> FloatArray:
        """裁剪到冻结的公开标准化物理域。"""

        raw = np.asarray(values, dtype=np.float64)
        if raw.shape[-1] != self.center.size:
            raise DataContractError("待投影数组末维不匹配")
        if not np.all(np.isfinite(raw)):
            raise DataContractError("待投影数组包含非有限值")
        lower = np.asarray([domain[0] for domain in self.standardized_domains])
        upper = np.asarray([domain[1] for domain in self.standardized_domains])
        return np.clip(raw, lower, upper).astype(np.float64, copy=False)


def select_training_panel(
    panel: SparsePanel,
    training_indices: Sequence[int],
    worker_count: int,
) -> SparsePanel:
    """按训练区覆盖率选固定面板，并按 Worker 标识打破并列。"""

    train = _indices(training_indices, panel.round_count, "training_indices")
    if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count <= 0:
        raise DataContractError("worker_count 必须为正整数")
    if worker_count > panel.worker_count:
        raise DataContractError("worker_count 超过可用 Worker 数")
    coverage = panel.present[list(train), :].mean(axis=0)
    ordered = sorted(
        range(panel.worker_count),
        key=lambda index: (-float(coverage[index]), panel.worker_ids[index]),
    )
    return panel.restrict_workers(tuple(panel.worker_ids[index] for index in ordered[:worker_count]))


def restrict_to_active_prefix(panel: SparsePanel, *, minimum_active_workers: int) -> SparsePanel:
    """按公开到达掩码截去末尾无法承载固定面板的时间段。

    该函数不读取测量值，也不删除保留前缀中的稀疏轮次。因此，后续的
    训练期面板选择和每块覆盖率校验仍能观察真实缺席报告。它只把数据源
    在最后一个满足原始设备活跃阈值的时间点之后的退场尾部排除出排程。
    """

    if isinstance(minimum_active_workers, bool) or not isinstance(minimum_active_workers, int):
        raise DataContractError("minimum_active_workers 必须为正整数")
    if minimum_active_workers <= 0 or minimum_active_workers > panel.worker_count:
        raise DataContractError("minimum_active_workers 超出 Worker 范围")
    active = np.flatnonzero(np.sum(panel.present, axis=1) >= minimum_active_workers)
    if not active.size:
        raise DataContractError("没有满足最低活跃设备数的时间窗口")
    return panel.take_rounds(tuple(range(int(active[-1]) + 1)))


def fit_training_transform(
    panel: SparsePanel,
    spec: DatasetSpec,
    training_indices: Sequence[int],
) -> TrainingTransform:
    """仅用训练区的存在报告拟合稳健标准化与噪声尺度。"""

    if panel.dataset_id != spec.dataset_id:
        raise DataContractError("panel 与 spec 的 dataset_id 不一致")
    if panel.features != spec.features:
        raise DataContractError("panel 与 spec 的特征顺序不一致")
    train = _indices(training_indices, panel.round_count, "training_indices")
    train_values = panel.values[list(train), :, :]
    train_present = panel.present[list(train), :]
    if not np.any(train_present):
        raise DataContractError("训练区不存在任何有效报告")

    flattened = np.where(train_present[..., None], train_values, np.nan).reshape(-1, panel.dimension)
    center = np.nanmedian(flattened, axis=0)
    mad = _median_absolute_deviation(flattened, axis=0)
    resolutions = np.asarray(spec.resolutions, dtype=np.float64)
    scale = np.maximum(1.4826 * mad, resolutions)
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(mad)) or not np.all(scale > 0):
        raise DataContractError("训练数据无法产生有限稳健尺度")

    per_round_center = np.nanmedian(np.where(train_present[..., None], train_values, np.nan), axis=1)
    deviations = train_values - per_round_center[:, None, :]
    deviations = np.where(train_present[..., None], deviations, np.nan).reshape(-1, panel.dimension)
    sigma_raw = _median_absolute_deviation(deviations, axis=0)
    sigma_h = np.maximum(1.4826 * sigma_raw / scale, resolutions / scale)
    if not np.all(np.isfinite(sigma_h)) or np.any(sigma_h <= 0):
        raise DataContractError("训练数据无法产生有限诚实噪声尺度")

    standardized_domains = tuple(
        ((lower - center[index]) / scale[index], (upper - center[index]) / scale[index])
        for index, (lower, upper) in enumerate(spec.domains)
    )
    return TrainingTransform(
        center=center,
        mad=mad,
        scale=scale,
        standardized_domains=standardized_domains,
        sigma_h=sigma_h,
        sigma_m=2.0 * sigma_h,
        selected_worker_ids=panel.worker_ids,
        training_indices=train,
    )


def build_clean_truth(panel: SparsePanel, transform: TrainingTransform) -> FloatArray:
    """以每轮有效原始报告中位数构造潜在真值，并映射到冻结坐标。"""

    if panel.worker_ids != transform.selected_worker_ids:
        raise DataContractError("panel Worker 顺序必须与训练变换一致")
    raw_truth = np.nanmedian(np.where(panel.present[..., None], panel.values, np.nan), axis=1)
    if not np.all(np.isfinite(raw_truth)):
        raise DataContractError("存在没有有效报告的轮次，无法构造干净真值")
    return transform.standardize(raw_truth)
