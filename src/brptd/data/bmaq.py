"""BMAQ 十二站官方 CSV 的严格小时对齐解析。"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import numpy as np

from .models import DataContractError, DatasetSpec, SparsePanel

BMAQ_FEATURES = (
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "WSPM",
)
BMAQ_STATIONS = (
    "Aotizhongxin",
    "Changping",
    "Dingling",
    "Dongsi",
    "Guanyuan",
    "Gucheng",
    "Huairou",
    "Nongzhanguan",
    "Shunyi",
    "Tiantan",
    "Wanliu",
    "Wanshouxigong",
)
BMAQ_SPEC = DatasetSpec(
    dataset_id="bmaq",
    features=BMAQ_FEATURES,
    domains=(
        (0.0, 1000.0),
        (0.0, 1200.0),
        (0.0, 500.0),
        (0.0, 300.0),
        (0.0, 10000.0),
        (0.0, 1200.0),
        (-40.0, 50.0),
        (900.0, 1100.0),
        (-50.0, 30.0),
        (0.0, 20.0),
    ),
    resolutions=(1.0, 1.0, 1.0, 1.0, 100.0, 1.0, 0.1, 0.1, 0.1, 0.1),
)


def _station_from_path(path: Path) -> str:
    stem = path.stem
    for station in BMAQ_STATIONS:
        if station in stem:
            return station
    raise DataContractError(f"无法从文件名识别 BMAQ 站点：{path.name}")


def _parse_row(row: dict[str, str], station: str) -> tuple[datetime, str, tuple[float, ...]] | None:
    try:
        timestamp = datetime(int(row["year"]), int(row["month"]), int(row["day"]), int(row["hour"]))
        values = tuple(float(row[feature]) for feature in BMAQ_FEATURES)
    except (KeyError, TypeError, ValueError):
        return None
    valid = all(
        np.isfinite(value) and lower <= value <= upper
        for value, (lower, upper) in zip(values, BMAQ_SPEC.domains, strict=True)
    )
    # 任何必需字段无效时，该站点该小时缺席，绝不以同轮统计量填补。
    return (timestamp, station, values) if valid else None


def _read_csv(path: Path) -> Iterable[tuple[datetime, str, tuple[float, ...]]]:
    station = _station_from_path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = _parse_row(row, station)
            if parsed is not None:
                yield parsed


def load_bmaq(paths: str | Path | Iterable[str | Path]) -> SparsePanel:
    """加载十二站 CSV 并按官方小时字段对齐为稀疏面板。"""

    if isinstance(paths, (str, Path)):
        candidate = Path(paths)
        source_paths = tuple(sorted(candidate.glob("*.csv"))) if candidate.is_dir() else (candidate,)
    else:
        source_paths = tuple(sorted((Path(path) for path in paths), key=lambda path: str(path)))
    if not source_paths:
        raise DataContractError("BMAQ 输入目录不含 CSV")
    recognized: dict[str, Path] = {}
    for path in source_paths:
        station = _station_from_path(path)
        previous = recognized.get(station)
        if previous is not None:
            raise DataContractError(f"BMAQ 存在重复站点 CSV：{station} ({previous.name}, {path.name})")
        recognized[station] = path
    missing = set(BMAQ_STATIONS) - set(recognized)
    if missing:
        raise DataContractError(f"BMAQ 缺少官方站点 CSV：{sorted(missing)}")
    records = [record for path in (recognized[station] for station in BMAQ_STATIONS) for record in _read_csv(path)]
    if not records:
        raise DataContractError("BMAQ 输入没有任何通过物理域校验的记录")
    timestamps = tuple(sorted({timestamp for timestamp, _, _ in records}))
    values = np.full((len(timestamps), len(BMAQ_STATIONS), len(BMAQ_FEATURES)), np.nan, dtype=np.float64)
    present = np.zeros((len(timestamps), len(BMAQ_STATIONS)), dtype=np.bool_)
    time_index = {timestamp: index for index, timestamp in enumerate(timestamps)}
    station_index = {station: index for index, station in enumerate(BMAQ_STATIONS)}
    for timestamp, station, vector in records:
        index_t = time_index[timestamp]
        index_n = station_index[station]
        # 官方记录每站每小时唯一。出现重复时拒绝，防止非声明的聚合语义。
        if present[index_t, index_n]:
            raise DataContractError(f"BMAQ 出现重复站点小时：{station} {timestamp.isoformat()}")
        values[index_t, index_n, :] = vector
        present[index_t, index_n] = True
    return SparsePanel(
        dataset_id=BMAQ_SPEC.dataset_id,
        timestamps=timestamps,
        worker_ids=BMAQ_STATIONS,
        features=BMAQ_FEATURES,
        values=values,
        present=present,
    )
