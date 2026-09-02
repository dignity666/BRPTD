"""指标、BCa 统计和无歧义空值口径的回归测试。"""

from __future__ import annotations

import unittest

import numpy as np

from brptd.metrics import (
    MetricError,
    RoundMetrics,
    TrialMetrics,
    average_ranks,
    bca_interval,
    crse_ratio,
    invalid_round_rate,
    malicious_weight_share,
    proof_acceptance_rate,
    standardized_crse,
    summarize_trial_metrics,
    worker_spearman,
)


class MetricFunctionTests(unittest.TestCase):
    def test_crse_and_ratio_use_standardized_values(self) -> None:
        self.assertAlmostEqual(5.0, standardized_crse([[3.0, 4.0]], [[0.0, 0.0]]))
        self.assertEqual(2.0, crse_ratio(10.0, 5.0))
        self.assertIsNone(crse_ratio(10.0, 0.0))
        with self.assertRaises(MetricError):
            standardized_crse([np.nan], [0.0])

    def test_average_ties_and_uncalculable_spearman(self) -> None:
        self.assertEqual((1.5, 1.5, 3.0), average_ranks([1.0, 1.0, 2.0]))
        self.assertAlmostEqual(1.0, worker_spearman([1.0, 1.0, 3.0], [7.0, 7.0, 9.0]) or 0.0)
        self.assertIsNone(worker_spearman([1.0, 1.0], [2.0, 3.0]))
        self.assertIsNone(worker_spearman([1.0], [2.0]))

    def test_weight_share_rejects_invalid_and_handles_zero(self) -> None:
        self.assertAlmostEqual(0.75, malicious_weight_share([0.25, 0.75], [False, True]) or 0.0)
        self.assertIsNone(malicious_weight_share([0.0, 0.0], [False, True]))
        with self.assertRaises(MetricError):
            malicious_weight_share([-1.0, 1.0], [True, False])

    def test_weight_share_does_not_exceed_one_from_floating_sum_order(self) -> None:
        weights = (
            0.9998641705695277,
            9.167865429416925e-24,
            1.3853341875023794e-31,
            5.290186881139163e-31,
            2.99400506788184e-12,
            3.533559841485175e-31,
            0.0001358294274783646,
            7.013605416150803e-33,
            8.902102907424978e-27,
            3.323922033939741e-39,
            4.766487897216341e-34,
            9.190906294119145e-20,
        )
        malicious = (True, False, False, True, True, False, True, False, False, True, False, True)
        self.assertEqual(1.0, malicious_weight_share(weights, malicious))

    def test_metric_inputs_reject_bad_shapes_values_and_empty_sequences(self) -> None:
        with self.assertRaises(MetricError):
            standardized_crse([1.0, 2.0], [1.0])
        with self.assertRaises(MetricError):
            standardized_crse([], [])
        with self.assertRaises(MetricError):
            crse_ratio(-1.0, 1.0)
        with self.assertRaises(MetricError):
            crse_ratio(float("nan"), 1.0)
        with self.assertRaises(MetricError):
            average_ranks([[1.0, 2.0]])
        with self.assertRaises(MetricError):
            worker_spearman([1.0], [1.0, 2.0])
        with self.assertRaises(MetricError):
            malicious_weight_share([1.0], [True, False])
        with self.assertRaises(MetricError):
            proof_acceptance_rate([])
        with self.assertRaises(MetricError):
            invalid_round_rate([])


class SummaryTests(unittest.TestCase):
    def _trial(self, trial_id: int, ratio: float | None) -> TrialMetrics:
        return TrialMetrics(
            dataset="ibrl",
            attack="bias",
            trial_id=trial_id,
            fold=trial_id // 5,
            block=trial_id % 5,
            nominal_malicious_ratio=0.3,
            actual_malicious_ratio=0.3,
            proof_mode="contract",
            exact_crse=1.0 + trial_id,
            bucket_crse=2.0 + trial_id,
            crse_ratio=ratio,
            mean_spearman=0.5,
            malicious_weight_share=0.3,
            proof_acceptance_rate=1.0,
            invalid_round_rate=0.0,
            uncalculable_spearman_count=0,
            uncalculable_ratio_count=0 if ratio is not None else 1,
        )

    def test_bca_is_deterministic_and_summary_counts_nulls(self) -> None:
        self.assertEqual(
            bca_interval([1.0, 2.0, 3.0], resamples=100, seed=17), bca_interval([1.0, 2.0, 3.0], resamples=100, seed=17)
        )
        summary = summarize_trial_metrics((self._trial(0, 1.0), self._trial(1, None)), resamples=100, seed=17)
        ratio = next(record for record in summary if record.metric == "crse_ratio")
        self.assertEqual(2, ratio.trial_count)
        self.assertEqual(1, ratio.uncalculable_count)
        self.assertEqual(1.0, ratio.mean)
        self.assertIsNone(ratio.sample_std)


class RecordTests(unittest.TestCase):
    def test_round_record_rejects_mismatched_rank_order(self) -> None:
        with self.assertRaisesRegex(MetricError, "长度"):
            RoundMetrics(
                "ibrl",
                "bias",
                202600,
                0,
                0,
                0,
                0.3,
                0.3,
                "contract",
                1.0,
                1.0,
                1.0,
                (1.0,),
                (1.0, 2.0),
                1.0,
                0.3,
                1.0,
                True,
            )

    def test_record_validation_rejects_out_of_range_fields(self) -> None:
        base = dict(
            dataset="ibrl",
            attack="bias",
            trial_id=202600,
            fold=0,
            block=0,
            round_index=0,
            nominal_malicious_ratio=0.3,
            actual_malicious_ratio=0.3,
            proof_mode="contract",
            exact_crse=1.0,
            bucket_crse=1.0,
            crse_ratio=1.0,
            exact_worker_ranks=(1.0, 2.0),
            proxy_worker_ranks=(1.0, 2.0),
            spearman=1.0,
            malicious_weight_share=0.3,
            proof_acceptance_rate=1.0,
            valid_update=True,
        )
        for key, value in (
            ("dataset", ""),
            ("trial_id", -1),
            ("round_index", -1),
            ("nominal_malicious_ratio", 2.0),
            ("actual_malicious_ratio", -0.1),
            ("crse_ratio", -1.0),
            ("spearman", 2.0),
            ("malicious_weight_share", 2.0),
            ("proof_acceptance_rate", 2.0),
            ("exact_worker_ranks", (float("nan"), 2.0)),
        ):
            arguments = dict(base)
            arguments[key] = value
            with self.subTest(key=key), self.assertRaises(MetricError):
                RoundMetrics(**arguments)


if __name__ == "__main__":
    unittest.main()
