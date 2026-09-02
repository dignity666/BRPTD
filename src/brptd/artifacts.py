"""实验 JSON、CSV 与图表元数据的原子发布工具。"""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import (
    ROUND_METRIC_FIELDS,
    SUMMARY_FIELDS,
    TRIAL_METRIC_FIELDS,
    RoundMetrics,
    SummaryMetrics,
    TrialMetrics,
)


class ArtifactError(ValueError):
    """制品数据不能无歧义发布。"""


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(vars(value))
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ArtifactError("JSON 制品禁止 NaN 或 Infinity")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise ArtifactError(f"JSON 制品不支持类型：{type(value).__name__}")


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    """以稳定键序和禁用 NaN 的 JSON 原子写入一次运行清单。"""

    target = Path(path)
    converted = _json_value(dict(manifest))
    payload = json.dumps(converted, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    _atomic_write(target, payload)
    return target


def _csv_scalar(value: Any) -> str:
    converted = _json_value(value)
    if converted is None:
        return ""
    if isinstance(converted, bool):
        return "true" if converted else "false"
    if isinstance(converted, (list, dict)):
        return json.dumps(converted, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if isinstance(converted, float):
        return format(converted, ".17g")
    return str(converted)


def write_csv_records(path: str | Path, fields: Sequence[str], records: Iterable[Mapping[str, Any]]) -> Path:
    """以固定列顺序将记录集先序列化到内存，再原子发布。"""

    rows = []
    for record in records:
        missing = set(fields) - set(record)
        extra = set(record) - set(fields)
        if missing or extra:
            raise ArtifactError(f"CSV 字段不匹配，缺失={sorted(missing)}，额外={sorted(extra)}")
        rows.append({field: _csv_scalar(record[field]) for field in fields})
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    target = Path(path)
    _atomic_write(target, buffer.getvalue())
    return target


def write_round_metrics(path: str | Path, records: Iterable[RoundMetrics]) -> Path:
    """发布逐轮指标 CSV。"""

    return write_csv_records(path, ROUND_METRIC_FIELDS, (record.as_record() for record in records))


def write_trial_metrics(path: str | Path, records: Iterable[TrialMetrics]) -> Path:
    """发布逐试验块指标 CSV。"""

    return write_csv_records(path, TRIAL_METRIC_FIELDS, (record.as_record() for record in records))


def write_summary(path: str | Path, records: Iterable[SummaryMetrics]) -> Path:
    """发布统计摘要 CSV。"""

    return write_csv_records(path, SUMMARY_FIELDS, (record.as_record() for record in records))
