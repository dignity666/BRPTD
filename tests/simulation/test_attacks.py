"""五类攻击公式、随机流隔离和 Fang 状态隔离回归测试。"""

from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np

from brptd.simulation import (
    AggregationPreview,
    AttackParameters,
    build_attack_scenario,
    build_base_scenario,
    derive_trial_seeds,
    optimize_fang_round,
    stable_seed,
)


def _base() -> object:
    truth = np.array([[0.0, 1.0], [0.5, 1.5], [1.0, 2.0]], dtype=np.float64)
    present = np.ones((3, 4), dtype=np.bool_)
    present[1, 3] = False
    return build_base_scenario(
        dataset="ibrl",
        worker_ids=("w0", "w1", "w2", "w3"),
        truth=truth,
        present=present,
        standardized_domains=((-20.0, 20.0), (-20.0, 20.0)),
        sigma_h=np.array([0.1, 0.2]),
        seeds=derive_trial_seeds("ibrl", 0, 0, 202600),
    )


class AttackFormulaTests(unittest.TestCase):
    def test_identity_and_honest_noise_are_shared_across_attack_types(self) -> None:
        base = _base()
        bias = build_attack_scenario(base, "bias", 2)
        drift = build_attack_scenario(base, "drift", 2)
        np.testing.assert_array_equal(bias.malicious_mask, drift.malicious_mask)
        honest = ~bias.malicious_mask
        np.testing.assert_array_equal(bias.reports[:, honest], base.honest_reports[:, honest])
        np.testing.assert_array_equal(drift.reports[:, honest], base.honest_reports[:, honest])
        np.testing.assert_array_equal(base.malicious_mask(1) | base.malicious_mask(2), base.malicious_mask(2))

    def test_bias_drift_spike_and_flip_follow_fixed_formulas(self) -> None:
        base = _base()
        mask = base.malicious_mask(1)
        index = int(np.flatnonzero(mask)[0])
        bias = build_attack_scenario(base, "bias", 1)
        noise = np.random.default_rng(base.attack_seed("bias")).normal(
            0.0, base.sigma_m, size=base.honest_reports.shape
        )
        np.testing.assert_allclose(bias.reports[:, index], base.truth + 1.8 + noise[:, index])
        drift = build_attack_scenario(base, "drift", 1)
        drift_noise = np.random.default_rng(base.attack_seed("drift")).normal(
            0.0, base.sigma_m, size=base.honest_reports.shape
        )
        np.testing.assert_allclose(drift.reports[0, index], base.truth[0] + 1.8 + drift_noise[0, index])
        np.testing.assert_allclose(drift.reports[2, index], base.truth[2] + 1.8 + 0.26 + drift_noise[2, index])
        spike = build_attack_scenario(base, "spike", 1)
        self.assertTrue(np.all(np.isin(spike.reports[:, index], (-20.0, 20.0))))
        flip = build_attack_scenario(base, "flip", 1)
        flip_noise = np.random.default_rng(base.attack_seed("flip")).normal(
            0.0, base.sigma_m, size=base.honest_reports.shape
        )
        np.testing.assert_allclose(
            flip.reports[0, index], base.truth[0] - base.honest_epsilon[0, index] + flip_noise[0, index]
        )

    def test_declared_attack_parameters_control_each_attack_random_stream(self) -> None:
        base = _base()
        mask = base.malicious_mask(1)
        index = int(np.flatnonzero(mask)[0])
        parameters = AttackParameters(
            bias_offset=2.5,
            drift_per_round=0.4,
            spike_upper_probability=1.0,
            fang_step_size=0.2,
            fang_maximum_steps=3,
            fang_tolerance=1e-5,
        )
        bias = build_attack_scenario(base, "bias", 1, parameters=parameters)
        noise = np.random.default_rng(base.attack_seed("bias")).normal(
            0.0, base.sigma_m, size=base.honest_reports.shape
        )
        np.testing.assert_allclose(bias.reports[:, index], base.truth + 2.5 + noise[:, index])
        drift = build_attack_scenario(base, "drift", 1, parameters=parameters)
        drift_noise = np.random.default_rng(base.attack_seed("drift")).normal(
            0.0, base.sigma_m, size=base.honest_reports.shape
        )
        np.testing.assert_allclose(drift.reports[2, index], base.truth[2] + 2.5 + 0.8 + drift_noise[2, index])
        spike = build_attack_scenario(base, "spike", 1, parameters=parameters)
        np.testing.assert_allclose(spike.reports[:, index], np.full((3, 2), 20.0))

    def test_missing_reports_remain_missing_after_every_attack(self) -> None:
        base = _base()
        for attack in ("bias", "drift", "spike", "flip"):
            scenario = build_attack_scenario(base, attack, 4)
            self.assertTrue(np.all(np.isnan(scenario.reports[1, 3])))

    def test_same_inputs_produce_byte_identical_reports(self) -> None:
        first = build_attack_scenario(_base(), "bias", 2)
        second = build_attack_scenario(_base(), "bias", 2)
        self.assertEqual(first.reports.tobytes(), second.reports.tobytes())

    def test_seed_derivation_and_scenario_contract_reject_invalid_inputs(self) -> None:
        self.assertEqual(stable_seed("unit", 1), stable_seed("unit", 1))
        self.assertNotEqual(stable_seed("unit", 1), stable_seed("unit", 2))
        with self.assertRaises(ValueError):
            stable_seed("", 1)
        with self.assertRaises(ValueError):
            derive_trial_seeds("", 0, 0, 202600)
        with self.assertRaises(ValueError):
            derive_trial_seeds("ibrl", -1, 0, 202600)
        base = _base()
        for count in (-1, 5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                base.malicious_mask(count)
        with self.assertRaises(ValueError):
            base.attack_seed("")


@dataclass
class _State:
    commits: int = 0

    def clone(self) -> _State:
        return _State(self.commits)


class _Evaluator:
    def __init__(self) -> None:
        self.seen_argument_count = 0

    def preview(self, reports: np.ndarray, present: np.ndarray, state: _State) -> AggregationPreview:
        self.seen_argument_count += 1
        state.commits += 1
        available = reports[present]
        estimate = np.mean(available, axis=0)
        weights = np.where(present, 1.0, 0.0)
        return AggregationPreview(estimate=estimate, weights=weights, valid_update=True)


class FangTests(unittest.TestCase):
    def test_fang_optimizes_on_clones_and_never_advances_formal_state(self) -> None:
        base = _base()
        state = _State()
        evaluator = _Evaluator()
        attack = build_attack_scenario(base, "fang", 2, fang_evaluator=evaluator, fang_state=state)
        self.assertEqual(0, state.commits)
        self.assertGreater(evaluator.seen_argument_count, 0)
        self.assertTrue(np.all(np.isnan(attack.reports[1, 3])))
        self.assertFalse(
            np.array_equal(attack.reports[:, attack.malicious_mask], base.honest_reports[:, attack.malicious_mask])
        )

    def test_fang_handles_zero_weights_without_mutating_input(self) -> None:
        class ZeroEvaluator:
            def preview(self, reports: np.ndarray, present: np.ndarray, state: object) -> AggregationPreview:
                return AggregationPreview(np.array([0.0, 0.0]), np.zeros(4), False)

        reports = np.zeros((4, 2))
        result = optimize_fang_round(
            reports=reports,
            present=np.ones(4, dtype=bool),
            truth=np.array([0.0, 0.0]),
            malicious_mask=np.array([True, False, False, False]),
            standardized_domains=((-1.0, 1.0), (-1.0, 1.0)),
            evaluator=ZeroEvaluator(),
            state=object(),
        )
        np.testing.assert_array_equal(reports, result)


if __name__ == "__main__":
    unittest.main()
