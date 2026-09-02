"""无泄漏扩展时间折和固定 15 轮试验块。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .models import DataContractError, SparsePanel


class FoldConstructionError(DataContractError):
    """公开数据不足以满足论文冻结的折与块契约。"""


@dataclass(frozen=True)
class Block:
    """一个固定长度、通过覆盖率约束的 outer 时间块。"""

    block_id: int
    indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.block_id < 0 or not self.indices:
            raise FoldConstructionError("block_id 和 indices 必须有效")
        if tuple(range(self.indices[0], self.indices[0] + len(self.indices))) != self.indices:
            raise FoldConstructionError("块必须由连续递增时间索引组成")


@dataclass(frozen=True)
class TimeFold:
    """训练区、隔离区和 outer 区严格互斥的扩展时间折。"""

    fold_id: int
    training_indices: tuple[int, ...]
    embargo_indices: tuple[int, ...]
    outer_indices: tuple[int, ...]
    blocks: tuple[Block, ...]

    def __post_init__(self) -> None:
        if self.fold_id < 0 or not self.training_indices or not self.outer_indices:
            raise FoldConstructionError("时间折缺少训练区或 outer 区")
        sets = [set(self.training_indices), set(self.embargo_indices), set(self.outer_indices)]
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise FoldConstructionError("训练区、隔离区和 outer 区必须互斥")
        if len(self.blocks) != 5:
            raise FoldConstructionError("每折必须恰好包含 5 个块")
        covered: set[int] = set()
        for block in self.blocks:
            if not set(block.indices).issubset(sets[2]) or covered & set(block.indices):
                raise FoldConstructionError("块必须互不重叠且完全位于 outer 区")
            covered.update(block.indices)


@dataclass(frozen=True)
class FoldBoundary:
    """尚未施加固定面板覆盖率筛选的时间边界。"""

    fold_id: int
    training_indices: tuple[int, ...]
    embargo_indices: tuple[int, ...]
    outer_indices: tuple[int, ...]


def build_fold_boundaries(
    round_count: int,
    *,
    outer_starts: Sequence[float] = (0.40, 0.55, 0.70, 0.85),
    outer_fraction: float = 0.15,
    embargo_rounds: int = 15,
) -> tuple[FoldBoundary, ...]:
    """仅依据时间长度构造四折边界，供训练期固定面板选择使用。"""

    if isinstance(round_count, bool) or not isinstance(round_count, int) or round_count <= 0:
        raise FoldConstructionError("round_count 必须为正整数")
    if not 0 < outer_fraction <= 1 or embargo_rounds < 0:
        raise FoldConstructionError("outer_fraction 或 embargo_rounds 非法")
    outer_length = math.floor(round_count * outer_fraction)
    if outer_length <= 0:
        raise FoldConstructionError("outer 区长度为空")
    result: list[FoldBoundary] = []
    for fold_id, start_fraction in enumerate(outer_starts):
        if not 0 < float(start_fraction) < 1:
            raise FoldConstructionError("outer_starts 必须位于 (0, 1)")
        outer_start = math.floor(round_count * float(start_fraction))
        outer_end = min(round_count, outer_start + outer_length)
        embargo_start = max(0, outer_start - embargo_rounds)
        training = tuple(range(embargo_start))
        if not training or not outer_start < outer_end:
            raise FoldConstructionError(f"fold {fold_id} 的训练区或 outer 区为空")
        result.append(
            FoldBoundary(
                fold_id=fold_id,
                training_indices=training,
                embargo_indices=tuple(range(embargo_start, outer_start)),
                outer_indices=tuple(range(outer_start, outer_end)),
            )
        )
    return tuple(result)


def select_fold_blocks(
    panel: SparsePanel,
    boundaries: Sequence[FoldBoundary],
    *,
    block_length: int = 15,
    blocks_per_fold: int = 5,
    minimum_coverage: float = 0.80,
) -> tuple[TimeFold, ...]:
    """在已经按训练区冻结的面板上贪心选择五个合格 outer 块。"""

    if block_length <= 0 or blocks_per_fold != 5 or not 0 < minimum_coverage <= 1:
        raise FoldConstructionError("块选择参数不满足论文契约")
    coverage = panel.present.mean(axis=1)
    folds: list[TimeFold] = []
    for boundary in boundaries:
        if len(boundary.outer_indices) < block_length * blocks_per_fold:
            raise FoldConstructionError(f"fold {boundary.fold_id} 的 outer 区长度不足")
        outer_start = boundary.outer_indices[0]
        outer_end = boundary.outer_indices[-1] + 1
        blocks: list[Block] = []
        cursor = outer_start
        while cursor + block_length <= outer_end and len(blocks) < blocks_per_fold:
            candidate = tuple(range(cursor, cursor + block_length))
            if bool(np.all(coverage[list(candidate)] >= minimum_coverage)):
                blocks.append(Block(len(blocks), candidate))
                cursor += block_length
            else:
                cursor += 1
        if len(blocks) != blocks_per_fold:
            raise FoldConstructionError(f"fold {boundary.fold_id} 无法从 outer 区构造五个满足覆盖率的块")
        folds.append(
            TimeFold(
                fold_id=boundary.fold_id,
                training_indices=boundary.training_indices,
                embargo_indices=boundary.embargo_indices,
                outer_indices=boundary.outer_indices,
                blocks=tuple(blocks),
            )
        )
    return tuple(folds)


def build_time_folds(
    panel: SparsePanel,
    *,
    outer_starts: Sequence[float] = (0.40, 0.55, 0.70, 0.85),
    outer_fraction: float = 0.15,
    embargo_rounds: int = 15,
    block_length: int = 15,
    blocks_per_fold: int = 5,
    minimum_coverage: float = 0.80,
) -> tuple[TimeFold, ...]:
    """按时间顺序构造四折，并贪心选择每折最早的五个合格块。

    从 outer 起点开始扫描。一个候选 15 轮区间若每轮都有至少 80% 的固定
    Worker 到达即入选；入选后跳过整个区间，未通过的候选仅右移一轮。这使
    选择完全由 outer 可观察性决定，且不会查看攻击或评估输出。
    """

    if not 0 < outer_fraction <= 1 or not 0 < minimum_coverage <= 1:
        raise FoldConstructionError("outer_fraction 和 minimum_coverage 必须在 (0, 1] 内")
    if embargo_rounds < 0 or block_length <= 0 or blocks_per_fold <= 0:
        raise FoldConstructionError("embargo_rounds、block_length 和 blocks_per_fold 必须合法")
    if blocks_per_fold != 5:
        raise FoldConstructionError("论文契约固定每折 5 个块")
    boundaries = build_fold_boundaries(
        panel.round_count,
        outer_starts=outer_starts,
        outer_fraction=outer_fraction,
        embargo_rounds=embargo_rounds,
    )
    return select_fold_blocks(
        panel,
        boundaries,
        block_length=block_length,
        blocks_per_fold=blocks_per_fold,
        minimum_coverage=minimum_coverage,
    )
