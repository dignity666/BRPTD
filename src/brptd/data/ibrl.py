"""IBRL 官方文本的严格解析与五分钟中位数聚合。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import numpy as np

from .models import DataContractError, DatasetSpec, SparsePanel

IBRL_FEATURES = ("temperature", "humidity", "light", "voltage")
IBRL_SPEC = DatasetSpec(
    dataset_id="ibrl",
    features=IBRL_FEATURES,
    domains=((-40.0, 85.0), (0.0, 100.0), (0.0, 2000.0), (1.5, 3.6)),
    resolutions=(0.01, 0.01, 0.01, 0.001),
)


def _floor_five_minutes(timestamp: datetime) -> datetime:
    return timestamp.replace(minute=(timestamp.minute // 5) * 5, second=0, microsecond=0)


def _valid_vector(values: tuple[float, ...]) -> bool:
    return all(
        np.isfinite(value) and lower <= value <= upper
        for value, (lower, upper) in zip(values, IBRL_SPEC.domains, strict=True)
    )


def _parse_line(line: str, line_number: int) -> tuple[datetime, str, tuple[float, ...]] | None:
    fields = line.split()
    if len(fields) < 8:
        return None
    try:
        timestamp = datetime.fromisoformat(f"{fields[0]} {fields[1]}")
        worker_numeric = int(fields[3])
        values = tuple(float(value) for value in fields[4:8])
    except (TypeError, ValueError):
        return None
    if worker_numeric < 1 or worker_numeric > 54 or not _valid_vector(values):
        return None
    return _floor_five_minutes(timestamp), str(worker_numeric), values


def _build_panel(
    records: Iterable[tuple[datetime, str, tuple[float, ...]]],
) -> SparsePanel:
    grouped: defaultdict[tuple[datetime, str], list[tuple[float, ...]]] = defaultdict(list)
    for timestamp, worker_id, values in records:
        grouped[(timestamp, worker_id)].append(values)
    if not grouped:
        raise DataContractError("IBRL 输入没有任何通过物理域校验的记录")
    timestamps = tuple(sorted({timestamp for timestamp, _ in grouped}))
    worker_ids = tuple(str(worker_id) for worker_id in range(1, 55))
    time_index = {timestamp: index for index, timestamp in enumerate(timestamps)}
    worker_index = {worker_id: index for index, worker_id in enumerate(worker_ids)}
    array_values = np.full((len(timestamps), len(worker_ids), len(IBRL_FEATURES)), np.nan, dtype=np.float64)
    present = np.zeros((len(timestamps), len(worker_ids)), dtype=np.bool_)
    for (timestamp, worker_id), entries in grouped.items():
        index_t = time_index[timestamp]
        index_n = worker_index[worker_id]
        array_values[index_t, index_n, :] = np.median(np.asarray(entries, dtype=np.float64), axis=0)
        present[index_t, index_n] = True
    return SparsePanel(
        dataset_id=IBRL_SPEC.dataset_id,
        timestamps=timestamps,
        worker_ids=worker_ids,
        features=IBRL_FEATURES,
        values=array_values,
        present=present,
    )


def load_ibrl(path: str | Path) -> SparsePanel:
    """解析 `data.txt` 并返回保留 54 个 mote 的稀疏五分钟面板。

    Top 50 面板选择故意不在此函数进行，因为它必须按每个时间折的训练区
    单独执行，参见 :func:`select_training_panel`。
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    records: list[tuple[datetime, str, tuple[float, ...]]] = []
    with source.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            parsed = _parse_line(line, line_number)
            if parsed is not None:
                records.append(parsed)
    return _build_panel(records)
