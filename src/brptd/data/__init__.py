"""真实数据获取、稀疏面板和训练期预处理。

该子包把原始观测与实验场景分开：数据层只保留经物理域校验的观测、
时间折与训练期变换，不生成攻击报告，也不执行 PP-CH 更新。
"""

from .bmaq import BMAQ_FEATURES, BMAQ_SPEC, load_bmaq
from .fetch import DataFetchError, fetch_dataset, load_manifest
from .folds import (
    Block,
    FoldBoundary,
    FoldConstructionError,
    TimeFold,
    build_fold_boundaries,
    build_time_folds,
    select_fold_blocks,
)
from .ibrl import IBRL_FEATURES, IBRL_SPEC, load_ibrl
from .models import DataContractError, DatasetSpec, SparsePanel
from .preprocess import (
    TrainingTransform,
    build_clean_truth,
    fit_training_transform,
    restrict_to_active_prefix,
    select_training_panel,
)

__all__ = [
    "BMAQ_FEATURES",
    "BMAQ_SPEC",
    "IBRL_FEATURES",
    "IBRL_SPEC",
    "Block",
    "DataContractError",
    "DataFetchError",
    "DatasetSpec",
    "FoldBoundary",
    "FoldConstructionError",
    "SparsePanel",
    "TimeFold",
    "TrainingTransform",
    "build_clean_truth",
    "build_fold_boundaries",
    "build_time_folds",
    "fetch_dataset",
    "fit_training_transform",
    "load_bmaq",
    "load_ibrl",
    "load_manifest",
    "restrict_to_active_prefix",
    "select_fold_blocks",
    "select_training_panel",
]
