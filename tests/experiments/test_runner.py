"""核心实验运行器的可恢复记录与证明模式契约测试。"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from brptd.data import SparsePanel
from brptd.experiments import execute_attack_scenario
from brptd.experiments.runner import DEFAULT_EXPERIMENT_CONFIG, ExperimentConfig, _dataset_trials, _publish_run
from brptd.metrics import TrialMetrics
from brptd.simulation import BaseScenario, build_base_scenario, derive_trial_seeds


def _scenario() -> BaseScenario:
    truth = np.column_stack((np.linspace(0.0, 1.4, 15), np.linspace(1.0, 2.4, 15)))
    return build_base_scenario(
        dataset="ibrl",
        worker_ids=("w0", "w1", "w2", "w3", "w4"),
        truth=truth,
        present=np.ones((15, 5), dtype=bool),
        standardized_domains=((-10.0, 10.0), (-10.0, 10.0)),
        sigma_h=np.array([0.1, 0.1]),
        seeds=derive_trial_seeds("ibrl", 0, 0, 202600),
    )


class ExperimentRunnerTests(unittest.TestCase):
    def test_publish_run_writes_only_data_artifacts(self) -> None:
        record = TrialMetrics(
            dataset="ibrl",
            attack="bias",
            trial_id=202600,
            fold=0,
            block=0,
            nominal_malicious_ratio=0.3,
            actual_malicious_ratio=0.3,
            proof_mode="contract",
            exact_crse=1.0,
            bucket_crse=1.1,
            crse_ratio=1.1,
            mean_spearman=1.0,
            malicious_weight_share=0.1,
            proof_acceptance_rate=1.0,
            invalid_round_rate=0.0,
            uncalculable_spearman_count=0,
            uncalculable_ratio_count=0,
        )
        with TemporaryDirectory() as temporary_directory:
            _publish_run(
                output_directory=Path(temporary_directory),
                config=ExperimentConfig(bootstrap_resamples=10),
                round_records=(),
                trial_records=(record,),
                proof_failures=(),
                scenario_manifest=(),
            )
            self.assertEqual(
                {"manifest.json", "round_metrics.csv", "summary.csv", "trial_metrics.csv"},
                {path.name for path in Path(temporary_directory).iterdir()},
            )

    def test_contract_mode_creates_fifteen_round_records_and_one_trial_record(self) -> None:
        result = execute_attack_scenario(
            base=_scenario(),
            attack="bias",
            malicious_count=2,
            trial_id=202600,
            fold=0,
            block=0,
            proof_mode="contract",
            bin_count=8,
        )
        self.assertEqual(15, len(result.round_metrics))
        self.assertEqual(1, len(result.trial_metrics))
        self.assertEqual("contract", result.trial_metrics[0].proof_mode)
        self.assertTrue(all(record.proof_acceptance_rate == 1.0 for record in result.round_metrics))

    def test_fang_runs_without_attack_label_in_aggregation_api(self) -> None:
        result = execute_attack_scenario(
            base=_scenario(),
            attack="fang",
            malicious_count=2,
            trial_id=202600,
            fold=0,
            block=0,
            proof_mode="contract",
            bin_count=16,
        )
        self.assertEqual(15, len(result.round_metrics))
        self.assertEqual(1, len(result.trial_metrics))

    def test_ibrl_active_prefix_is_recorded_before_fold_construction(self) -> None:
        rounds, workers, dimensions = 600, 50, 4
        timestamps = tuple(datetime(2024, 1, 1) + timedelta(minutes=5 * index) for index in range(rounds))
        values = np.zeros((rounds, workers, dimensions), dtype=np.float64)
        present = np.ones((rounds, workers), dtype=np.bool_)
        # 原始末段只有 39 台到达，必须在时间折前作为退场尾部截去。
        present[500:, 39:] = False
        values[500:, 39:] = np.nan
        raw = SparsePanel(
            "ibrl",
            timestamps,
            tuple(str(index) for index in range(1, workers + 1)),
            ("temperature", "humidity", "light", "voltage"),
            values,
            present,
        )
        with (
            patch("brptd.experiments.runner._load_panel", return_value=(raw, ())),
            patch("brptd.experiments.runner._source_data_hashes", return_value={}),
        ):
            scenarios = list(_dataset_trials("ibrl", Path("/unused"), DEFAULT_EXPERIMENT_CONFIG))

        self.assertEqual(20, len(scenarios))
        schedule = scenarios[0][4]["availability_schedule"]
        self.assertEqual("raw-arrival-active-prefix", schedule["policy"])
        self.assertEqual(600, schedule["source_round_count"])
        self.assertEqual(500, schedule["retained_round_count"])
        self.assertEqual(40, schedule["minimum_active_workers"])


if __name__ == "__main__":
    unittest.main()
