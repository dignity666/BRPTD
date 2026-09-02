"""真实数据稀疏语义、训练隔离和时间折的回归测试。"""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from brptd.data import (
    BMAQ_SPEC,
    DataContractError,
    DataFetchError,
    DatasetSpec,
    FoldConstructionError,
    SparsePanel,
    build_clean_truth,
    build_time_folds,
    fetch_dataset,
    fit_training_transform,
    load_bmaq,
    load_ibrl,
    restrict_to_active_prefix,
    select_training_panel,
)


def _panel(rounds: int = 100, workers: int = 4) -> SparsePanel:
    timestamps = tuple(datetime(2020, 1, 1) + timedelta(minutes=5 * index) for index in range(rounds))
    values = np.empty((rounds, workers, 2), dtype=np.float64)
    for round_index in range(rounds):
        for worker_index in range(workers):
            values[round_index, worker_index] = (20.0 + worker_index, 40.0 + round_index % 3)
    present = np.ones((rounds, workers), dtype=np.bool_)
    return SparsePanel(
        dataset_id="tiny",
        timestamps=timestamps,
        worker_ids=tuple(f"w{index}" for index in range(workers)),
        features=("x", "y"),
        values=values,
        present=present,
    )


TINY_SPEC = DatasetSpec("tiny", ("x", "y"), ((0.0, 100.0), (0.0, 100.0)), (0.1, 0.1))


class SparsePanelAndPreprocessTests(unittest.TestCase):
    def test_missing_values_must_remain_nan(self) -> None:
        panel = _panel(20, 3)
        values = np.array(panel.values, copy=True)
        present = np.array(panel.present, copy=True)
        present[1, 2] = False
        with self.assertRaisesRegex(ValueError, "NaN"):
            SparsePanel("tiny", panel.timestamps, panel.worker_ids, panel.features, values, present)
        values[1, 2] = np.nan
        sparse = SparsePanel("tiny", panel.timestamps, panel.worker_ids, panel.features, values, present)
        self.assertTrue(np.all(np.isnan(sparse.values[1, 2])))

    def test_training_panel_and_transform_ignore_outer_mutation(self) -> None:
        panel = _panel(100, 4)
        train = tuple(range(40))
        selected = select_training_panel(panel, train, 3)
        transform = fit_training_transform(selected, TINY_SPEC, train)
        values = np.array(panel.values, copy=True)
        values[70:, :, :] = 99.0
        mutated = SparsePanel("tiny", panel.timestamps, panel.worker_ids, panel.features, values, panel.present)
        second = fit_training_transform(select_training_panel(mutated, train, 3), TINY_SPEC, train)
        np.testing.assert_array_equal(transform.center, second.center)
        np.testing.assert_array_equal(transform.scale, second.scale)
        np.testing.assert_array_equal(transform.sigma_h, second.sigma_h)

    def test_truth_uses_only_present_workers_without_imputation(self) -> None:
        panel = _panel(20, 3)
        values = np.array(panel.values, copy=True)
        present = np.array(panel.present, copy=True)
        present[3, 2] = False
        values[3, 2] = np.nan
        sparse = SparsePanel("tiny", panel.timestamps, panel.worker_ids, panel.features, values, present)
        transform = fit_training_transform(sparse, TINY_SPEC, tuple(range(10)))
        truth = build_clean_truth(sparse, transform)
        raw = transform.inverse(truth)
        np.testing.assert_allclose(raw[3], np.median(values[3, :2], axis=0))

    def test_active_prefix_uses_only_arrival_mask_and_preserves_sparse_rows(self) -> None:
        panel = _panel(120, 4)
        values = np.array(panel.values, copy=True)
        present = np.array(panel.present, copy=True)
        # 中间的稀疏轮次必须被原样保留，只有末尾设备退场区会被截去。
        present[10, 3] = False
        values[10, 3] = np.nan
        present[100:, 2:] = False
        values[100:, 2:] = np.nan
        sparse = SparsePanel("tiny", panel.timestamps, panel.worker_ids, panel.features, values, present)

        active = restrict_to_active_prefix(sparse, minimum_active_workers=3)

        self.assertEqual(100, active.round_count)
        self.assertEqual(sparse.timestamps[:100], active.timestamps)
        self.assertFalse(active.present[10, 3])
        self.assertTrue(np.all(np.isnan(active.values[10, 3])))
        with self.assertRaisesRegex(DataContractError, "没有满足"):
            restrict_to_active_prefix(active.take_rounds((10,)), minimum_active_workers=4)


class FoldTests(unittest.TestCase):
    def test_four_folds_have_twenty_nonoverlapping_blocks(self) -> None:
        panel = _panel(500, 5)
        folds = build_time_folds(panel)
        self.assertEqual(4, len(folds))
        self.assertEqual(20, sum(len(fold.blocks) for fold in folds))
        for fold in folds:
            self.assertEqual(5, len(fold.blocks))
            self.assertFalse(set(fold.training_indices) & set(fold.outer_indices))
            self.assertFalse(set(fold.embargo_indices) & set(fold.outer_indices))

    def test_invalid_coverage_or_short_outer_area_fails_closed(self) -> None:
        with self.assertRaises(FoldConstructionError):
            build_time_folds(_panel(80, 4))
        panel = _panel(500, 5)
        present = np.array(panel.present, copy=True)
        present[200:400] = False
        values = np.array(panel.values, copy=True)
        values[~present] = np.nan
        sparse = SparsePanel("tiny", panel.timestamps, panel.worker_ids, panel.features, values, present)
        with self.assertRaises(FoldConstructionError):
            build_time_folds(sparse)


class ParserTests(unittest.TestCase):
    def test_ibrl_filters_domains_and_aggregates_five_minutes(self) -> None:
        content = "\n".join(
            [
                "2004-02-28 00:01:00.000 1 1 20.0 40.0 100.0 2.5",
                "2004-02-28 00:04:00.000 1 1 22.0 42.0 120.0 2.7",
                "2004-02-28 00:06:00.000 1 2 20.0 40.0 2100.0 2.5",
                "bad row",
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "data.txt"
            source.write_text(content, encoding="utf-8")
            panel = load_ibrl(source)
        self.assertEqual(1, panel.round_count)
        self.assertTrue(panel.present[0, 0])
        np.testing.assert_allclose(panel.values[0, 0], (21.0, 41.0, 110.0, 2.6))
        self.assertFalse(panel.present[0, 1])
        self.assertTrue(np.all(np.isnan(panel.values[0, 1])))

    def test_bmaq_invalid_required_field_marks_station_hour_missing(self) -> None:
        header = "No,year,month,day,hour,PM2.5,PM10,SO2,NO2,CO,O3,TEMP,PRES,DEWP,WSPM,station\n"
        good = "1,2013,3,1,0,10,20,3,4,500,6,7,1000,8,1,Aotizhongxin\n"
        invalid = "1,2013,3,1,0,,20,3,4,500,6,7,1000,8,1,Changping\n"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for station in (
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
            ):
                row = invalid if station == "Changping" else good.replace("Aotizhongxin", station)
                (root / f"PRSA_Data_{station}_20130301-20170228.csv").write_text(header + row, encoding="utf-8")
            panel = load_bmaq(root)
        self.assertEqual(1, panel.round_count)
        self.assertTrue(panel.present[0, 0])
        self.assertFalse(panel.present[0, 1])
        self.assertTrue(np.all(np.isnan(panel.values[0, 1])))
        self.assertEqual(BMAQ_SPEC.features, panel.features)

    def test_bmaq_duplicate_station_file_fails_closed(self) -> None:
        """同一站点的两个 CSV 不可由路径字典静默覆盖。"""

        header = "No,year,month,day,hour,PM2.5,PM10,SO2,NO2,CO,O3,TEMP,PRES,DEWP,WSPM,station\n"
        row = "1,2013,3,1,0,10,20,3,4,500,6,7,1000,8,1,{station}\n"
        stations = (
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
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for station in stations:
                (root / f"PRSA_Data_{station}_20130301-20170228.csv").write_text(
                    header + row.format(station=station), encoding="utf-8"
                )
            (root / "PRSA_Data_Aotizhongxin_duplicate.csv").write_text(
                header + row.format(station="Aotizhongxin"), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "重复站点"):
                load_bmaq(root)


class FetchTests(unittest.TestCase):
    def test_unlocked_archive_hash_rejects_before_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_dir = Path(directory) / "manifests"
            manifest_dir.mkdir()
            (manifest_dir / "ibrl.json").write_text(
                '{"dataset":{"id":"ibrl"},"archive":{"sha256":null},"source":{"download_url":"https://example.invalid/data"}}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DataFetchError, "未锁定"):
                fetch_dataset("ibrl", manifest_directory=manifest_dir, data_root=Path(directory) / "raw")


if __name__ == "__main__":
    unittest.main()
