"""数据模型、训练变换和时间折边界的负向契约测试。"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

import numpy as np

from brptd.data import (
    DataContractError,
    DatasetSpec,
    FoldBoundary,
    FoldConstructionError,
    SparsePanel,
    TrainingTransform,
    build_fold_boundaries,
    build_time_folds,
    select_fold_blocks,
)


def _panel() -> SparsePanel:
    timestamps = tuple(datetime(2020, 1, 1) + timedelta(minutes=index) for index in range(100))
    values = np.ones((100, 3, 2), dtype=np.float64)
    present = np.ones((100, 3), dtype=np.bool_)
    return SparsePanel("tiny", timestamps, ("a", "b", "c"), ("x", "y"), values, present)


class DatasetSpecTests(unittest.TestCase):
    def test_invalid_spec_fields_are_rejected(self) -> None:
        cases = (
            ("", ("x",), ((0.0, 1.0),), (0.1,)),
            ("x", (), (), ()),
            ("x", ("x",), (), ()),
            ("x", ("x", "x"), ((0.0, 1.0), (0.0, 1.0)), (0.1, 0.1)),
            ("x", ("x",), ((1.0, 1.0),), (0.1,)),
            ("x", ("x",), ((float("nan"), 1.0),), (0.1,)),
            ("x", ("x",), ((0.0, 1.0),), (0.0,)),
            ("x", ("x",), ((0.0, 1.0),), (float("inf"),)),
        )
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises(DataContractError):
                DatasetSpec(*arguments)
        self.assertEqual(1, DatasetSpec("x", ("x",), ((0.0, 1.0),), (0.1,)).dimension)


class SparsePanelTests(unittest.TestCase):
    def test_invalid_panel_shapes_and_order_are_rejected(self) -> None:
        panel = _panel()
        cases = (
            ("", panel.timestamps, panel.worker_ids, panel.features, panel.values, panel.present),
            ("tiny", (), panel.worker_ids, panel.features, panel.values, panel.present),
            ("tiny", tuple(reversed(panel.timestamps)), panel.worker_ids, panel.features, panel.values, panel.present),
            (
                "tiny",
                (*panel.timestamps, panel.timestamps[-1]),
                panel.worker_ids,
                panel.features,
                panel.values,
                panel.present,
            ),
            ("tiny", panel.timestamps, ("a", "a", "c"), panel.features, panel.values, panel.present),
            ("tiny", panel.timestamps, panel.worker_ids, ("x", "x"), panel.values, panel.present),
            ("tiny", panel.timestamps, panel.worker_ids, panel.features, np.ones((2, 3, 2)), panel.present),
            ("tiny", panel.timestamps, panel.worker_ids, panel.features, panel.values, np.ones((100, 2))),
        )
        for arguments in cases:
            with self.subTest(arguments=arguments[:4]), self.assertRaises(DataContractError):
                SparsePanel(*arguments)
        values = np.array(panel.values, copy=True)
        values[0, 0, 0] = np.nan
        with self.assertRaises(DataContractError):
            SparsePanel("tiny", panel.timestamps, panel.worker_ids, panel.features, values, panel.present)
        present = np.array(panel.present, copy=True)
        present[0, 0] = False
        values = np.array(panel.values, copy=True)
        with self.assertRaises(DataContractError):
            SparsePanel("tiny", panel.timestamps, panel.worker_ids, panel.features, values, present)

    def test_panel_views_validate_indices_and_are_frozen(self) -> None:
        panel = _panel()
        self.assertEqual((100, 2, 2), panel.restrict_workers(("c", "a")).values.shape)
        self.assertEqual((2, 3, 2), panel.take_rounds((1, 2)).values.shape)
        self.assertEqual(3, panel.worker_count)
        self.assertEqual(2, panel.dimension)
        self.assertIn("worker_ids", panel.as_manifest_fragment())
        for workers in ((), ("a", "a"), ("unknown",)):
            with self.subTest(workers=workers), self.assertRaises(DataContractError):
                panel.restrict_workers(workers)
        for indices in ((), (-1,), (100,), (2, 1), (1, 1)):
            with self.subTest(indices=indices), self.assertRaises(DataContractError):
                panel.take_rounds(indices)


class TransformAndFoldTests(unittest.TestCase):
    def test_transform_validation_and_fold_boundary_errors(self) -> None:
        common = dict(
            center=np.array([0.0, 0.0]),
            mad=np.array([1.0, 1.0]),
            scale=np.array([1.0, 1.0]),
            standardized_domains=((-1.0, 1.0), (-1.0, 1.0)),
            sigma_h=np.array([1.0, 1.0]),
            sigma_m=np.array([2.0, 2.0]),
            selected_worker_ids=("a",),
            training_indices=(0,),
        )
        for key, value in (
            ("center", np.array([np.nan, 0.0])),
            ("scale", np.array([0.0, 1.0])),
            ("sigma_h", np.array([0.0, 1.0])),
            ("sigma_m", np.array([3.0, 2.0])),
            ("standardized_domains", ((-1.0, 1.0),)),
            ("standardized_domains", ((1.0, -1.0), (-1.0, 1.0))),
            ("selected_worker_ids", ()),
            ("training_indices", ()),
        ):
            arguments = dict(common)
            arguments[key] = value
            with self.subTest(key=key), self.assertRaises(DataContractError):
                TrainingTransform(**arguments)
        transform = TrainingTransform(**common)
        for method in (transform.standardize, transform.inverse, transform.project_standardized):
            with self.subTest(method=method), self.assertRaises(DataContractError):
                method(np.array([1.0]))
            with self.subTest(method=method), self.assertRaises(DataContractError):
                method(np.array([np.nan, 0.0]))
        np.testing.assert_allclose(transform.project_standardized(np.array([3.0, -3.0])), (1.0, -1.0))

        with self.assertRaises(FoldConstructionError):
            build_fold_boundaries(0)
        with self.assertRaises(FoldConstructionError):
            build_fold_boundaries(100, outer_fraction=0.0)
        with self.assertRaises(FoldConstructionError):
            build_fold_boundaries(100, embargo_rounds=-1)
        with self.assertRaises(FoldConstructionError):
            build_time_folds(_panel(), blocks_per_fold=4)
        boundary = FoldBoundary(0, (0,), (), tuple(range(40, 100)))
        with self.assertRaises(FoldConstructionError):
            select_fold_blocks(_panel(), (boundary,), block_length=0)


if __name__ == "__main__":
    unittest.main()
