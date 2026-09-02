"""不可变稀疏面板和数据集公开物理契约。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


class DataContractError(ValueError):
    """输入数据不满足实验公开契约。"""


@dataclass(frozen=True)
class DatasetSpec:
    """数据集的固定特征、物理域和原始空间分辨率。"""

    dataset_id: str
    features: tuple[str, ...]
    domains: tuple[tuple[float, float], ...]
    resolutions: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise DataContractError("dataset_id 不能为空")
        if not self.features:
            raise DataContractError("features 不能为空")
        if len(self.features) != len(self.domains) or len(self.features) != len(self.resolutions):
            raise DataContractError("features、domains 和 resolutions 的维度必须一致")
        if len(set(self.features)) != len(self.features):
            raise DataContractError("features 不可重复")
        for index, (lower, upper) in enumerate(self.domains):
            if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
                raise DataContractError(f"domains[{index}] 必须是有限递增区间")
        for index, resolution in enumerate(self.resolutions):
            if not np.isfinite(resolution) or resolution <= 0:
                raise DataContractError(f"resolutions[{index}] 必须为正有限数")

    @property
    def dimension(self) -> int:
        """返回特征维度。"""

        return len(self.features)


@dataclass(frozen=True)
class SparsePanel:
    """按时间、Worker 和特征排列的稀疏合法观测面板。

    `values[t, n, :]` 仅在 `present[t, n]` 为真时是有限观测；缺席位置
    必须逐维保留 NaN。这一不变量杜绝了使用当轮中位数伪造缺失报告。
    """

    dataset_id: str
    timestamps: tuple[datetime, ...]
    worker_ids: tuple[str, ...]
    features: tuple[str, ...]
    values: FloatArray
    present: BoolArray

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise DataContractError("dataset_id 不能为空")
        if not self.timestamps:
            raise DataContractError("timestamps 不能为空")
        if tuple(sorted(self.timestamps)) != self.timestamps:
            raise DataContractError("timestamps 必须严格按物理时间排序")
        if len(set(self.timestamps)) != len(self.timestamps):
            raise DataContractError("timestamps 不可重复")
        if not self.worker_ids or len(set(self.worker_ids)) != len(self.worker_ids):
            raise DataContractError("worker_ids 必须非空且不可重复")
        if not self.features or len(set(self.features)) != len(self.features):
            raise DataContractError("features 必须非空且不可重复")

        raw_values = np.asarray(self.values, dtype=np.float64)
        raw_present = np.asarray(self.present, dtype=np.bool_)
        expected_values_shape = (len(self.timestamps), len(self.worker_ids), len(self.features))
        expected_present_shape = (len(self.timestamps), len(self.worker_ids))
        if raw_values.shape != expected_values_shape:
            raise DataContractError(f"values 形状必须为 {expected_values_shape}，实际为 {raw_values.shape}")
        if raw_present.shape != expected_present_shape:
            raise DataContractError(f"present 形状必须为 {expected_present_shape}，实际为 {raw_present.shape}")

        occupied = raw_values[raw_present]
        missing = raw_values[~raw_present]
        if occupied.size and not np.all(np.isfinite(occupied)):
            raise DataContractError("存在的报告必须全部为有限值")
        if missing.size and not np.all(np.isnan(missing)):
            raise DataContractError("缺席报告必须逐维保存为 NaN")

        # 冻结底层数组，防止拟合后被评估期数据或调用方意外改写。
        values = np.array(raw_values, dtype=np.float64, copy=True)
        present = np.array(raw_present, dtype=np.bool_, copy=True)
        values.setflags(write=False)
        present.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "present", present)

    @property
    def round_count(self) -> int:
        """返回时间轮次数。"""

        return len(self.timestamps)

    @property
    def worker_count(self) -> int:
        """返回固定 Worker 数。"""

        return len(self.worker_ids)

    @property
    def dimension(self) -> int:
        """返回观测维度。"""

        return len(self.features)

    def restrict_workers(self, worker_ids: Sequence[str]) -> SparsePanel:
        """按给定顺序返回固定 Worker 子面板。"""

        requested = tuple(str(worker_id) for worker_id in worker_ids)
        if not requested or len(set(requested)) != len(requested):
            raise DataContractError("请求的 worker_ids 必须非空且不可重复")
        lookup = {worker_id: index for index, worker_id in enumerate(self.worker_ids)}
        try:
            indices = [lookup[worker_id] for worker_id in requested]
        except KeyError as error:
            raise DataContractError(f"未知 Worker：{error.args[0]}") from error
        return SparsePanel(
            dataset_id=self.dataset_id,
            timestamps=self.timestamps,
            worker_ids=requested,
            features=self.features,
            values=self.values[:, indices, :],
            present=self.present[:, indices],
        )

    def take_rounds(self, indices: Sequence[int]) -> SparsePanel:
        """返回按给定时间索引选择的子面板。"""

        selected = tuple(int(index) for index in indices)
        if not selected:
            raise DataContractError("round indices 不能为空")
        if any(index < 0 or index >= self.round_count for index in selected):
            raise DataContractError("round index 超出范围")
        if tuple(sorted(selected)) != selected or len(set(selected)) != len(selected):
            raise DataContractError("round indices 必须严格递增且不可重复")
        return SparsePanel(
            dataset_id=self.dataset_id,
            timestamps=tuple(self.timestamps[index] for index in selected),
            worker_ids=self.worker_ids,
            features=self.features,
            values=self.values[list(selected), :, :],
            present=self.present[list(selected), :],
        )

    def as_manifest_fragment(self) -> dict[str, Any]:
        """返回可直接写入实验清单的无二义性面板摘要。"""

        return {
            "dataset": self.dataset_id,
            "feature_count": self.dimension,
            "features": list(self.features),
            "round_count": self.round_count,
            "time_end": self.timestamps[-1].isoformat(),
            "time_start": self.timestamps[0].isoformat(),
            "worker_count": self.worker_count,
            "worker_ids": list(self.worker_ids),
        }
