"""PP-CH 两阶段筛选与滑动窗口状态机的回归测试。"""

from __future__ import annotations

import inspect
import math
import unittest
from dataclasses import FrozenInstanceError, fields

from brptd.robustness import PPCHConfig, PPCHDecision, PPCHState


class PPCHConfigTests(unittest.TestCase):
    def test_paper_defaults_are_exposed_by_frozen_config(self) -> None:
        config = PPCHConfig()

        self.assertEqual(config.window_length, 5)
        self.assertEqual(config.decay, 0.8)
        self.assertEqual(config.alpha, 1)
        self.assertEqual(config.beta, 1)
        self.assertEqual(config.cold_start, 0.2)
        self.assertEqual(config.cook_k, 6)
        self.assertEqual(config.minimum_scale, 1.5)
        self.assertEqual(config.effective_parameter_ratio, 0.6)
        self.assertEqual(config.minimum_leverage, 0.01)
        self.assertEqual(config.maximum_leverage, 0.25)
        self.assertEqual(config.leverage_exponent, 1)
        self.assertEqual(config.epsilon, 1e-6)
        self.assertEqual(config.hampel_z_max, 12)
        self.assertEqual(config.hampel_a, 1.5)
        self.assertEqual(config.hampel_b, 3)
        self.assertEqual(config.hampel_c, 4.5)
        self.assertNotIn("weight_floor", {field.name for field in fields(config)})

        with self.assertRaises(FrozenInstanceError):
            config.decay = 0.5  # type: ignore[misc]

    def test_invalid_configurations_are_rejected(self) -> None:
        invalid_cases = (
            {"window_length": 0},
            {"window_length": True},
            {"decay": 0},
            {"decay": 1.01},
            {"alpha": 0},
            {"beta": -1},
            {"cold_start": -0.01},
            {"cold_start": 1.01},
            {"cook_k": 0},
            {"minimum_scale": 0},
            {"effective_parameter_ratio": 0},
            {"minimum_leverage": 0},
            {"maximum_leverage": 1},
            {"minimum_leverage": 0.3, "maximum_leverage": 0.2},
            {"leverage_exponent": -0.01},
            {"epsilon": 0},
            {"hampel_a": 0},
            {"hampel_a": 3, "hampel_b": 3},
            {"hampel_b": 5, "hampel_c": 4.5},
            {"hampel_c": 12, "hampel_z_max": 12},
            {"decay": math.inf},
            {"epsilon": math.nan},
        )

        for kwargs in invalid_cases:
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                PPCHConfig(**kwargs)


class PPCHDecisionRuleTests(unittest.TestCase):
    def test_first_round_previous_weights_are_uniform(self) -> None:
        state = PPCHState(4)

        decision = state.preview(
            scores=[0.0, 0.0, 0.0, 0.0],
            present=[True, True, True, True],
            verified=[True, True, True, True],
        )

        self.assertEqual(state.previous_weights, (0.25, 0.25, 0.25, 0.25))
        self.assertEqual(decision.leverage, (0.15, 0.15, 0.15, 0.15))
        self.assertEqual(decision.final_weights, (0.25, 0.25, 0.25, 0.25))

    def test_cook_filter_is_centered_and_penalizes_only_the_right_tail(self) -> None:
        state = PPCHState(4)
        masks = [True] * 4

        base = state.preview([0.0, 0.0, 0.0, 10.0], masks, masks)
        shifted = state.preview([100.0, 100.0, 100.0, 110.0], masks, masks)

        expected_outlier_distance = (10.0 / 1.5) ** 2 * (0.15 / 0.85)
        self.assertEqual(base.survivors, (True, True, True, False))
        self.assertEqual(shifted.survivors, base.survivors)
        self.assertEqual(base.cook_distances[:3], (0.0, 0.0, 0.0))
        self.assertAlmostEqual(base.cook_distances[3], expected_outlier_distance)
        self.assertEqual(shifted.cook_distances, base.cook_distances)
        self.assertEqual(base.stage1_center, 0.0)
        self.assertEqual(shifted.stage1_center, 100.0)
        self.assertEqual(base.stage1_scale, 1.5)

    def test_hampel_uses_all_four_paper_segments_without_a_weight_floor(self) -> None:
        config = PPCHConfig(cook_k=1_000_000, minimum_scale=1.0)
        state = PPCHState(8, config=config)
        masks = [True] * 8

        decision = state.preview(
            scores=[0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 4.0, 5.0],
            present=masks,
            verified=masks,
        )

        expected_third_segment = 1.5 * (4.5 - 4.0) / ((4.5 - 3.0) * 4.0)
        self.assertEqual(decision.hampel_weights[:5], (1.0,) * 5)
        self.assertAlmostEqual(decision.hampel_weights[5], 1.5 / 2.0)
        self.assertAlmostEqual(decision.hampel_weights[6], expected_third_segment)
        self.assertEqual(decision.hampel_weights[7], 0.0)
        self.assertEqual(decision.raw_weights[7], 0.0)
        self.assertEqual(decision.final_weights[7], 0.0)
        self.assertEqual(decision.stage2_center, 0.0)
        self.assertEqual(decision.stage2_scale, 1.0)

    def test_only_present_and_verified_workers_enter_both_stages(self) -> None:
        state = PPCHState(4)

        decision = state.preview(
            scores=[0.0, 1.0, 2.0, 3.0],
            present=[True, True, False, True],
            verified=[True, False, True, True],
        )

        self.assertEqual(decision.survivors, (True, False, False, True))
        self.assertEqual(decision.leverage[1:3], (0.0, 0.0))
        self.assertEqual(decision.cook_distances[1:3], (0.0, 0.0))
        self.assertEqual(decision.hampel_weights[1:3], (0.0, 0.0))
        self.assertEqual(decision.raw_weights[1:3], (0.0, 0.0))
        self.assertEqual(decision.final_weights[1:3], (0.0, 0.0))

    def test_decision_is_frozen_and_contains_only_immutable_sequences(self) -> None:
        decision = PPCHState(2).preview([0.0, 0.0], [True, True], [True, True])

        tuple_fields = (
            "final_weights",
            "raw_weights",
            "sliding_weights",
            "hampel_weights",
            "cook_distances",
            "leverage",
            "survivors",
        )
        for name in tuple_fields:
            self.assertIsInstance(getattr(decision, name), tuple)
        with self.assertRaises(FrozenInstanceError):
            decision.valid_update = False  # type: ignore[misc]


class PPCHSlidingWindowTests(unittest.TestCase):
    def test_current_round_does_not_enter_its_own_sliding_score(self) -> None:
        state = PPCHState(1)

        current = state.update([9.0], [True], [True])
        following = state.preview([0.0], [True], [True])

        self.assertEqual(current.sliding_weights, (0.2,))
        self.assertAlmostEqual(following.sliding_weights[0], 0.1)

    def test_missing_report_contributes_zero_and_remains_in_denominator(self) -> None:
        state = PPCHState(2, PPCHConfig(decay=1.0))
        state.update([0.0, 0.0], [True, False], [True, False])

        after_missing = state.preview([0.0, 0.0], [True, True], [True, True])
        self.assertEqual(after_missing.sliding_weights, (1.0, 0.0))

        state.update([0.0, 0.0], [True, True], [True, True])
        two_slots = state.preview([0.0, 0.0], [True, True], [True, True])
        self.assertEqual(two_slots.sliding_weights, (1.0, 0.5))

    def test_cold_start_applies_only_when_there_are_no_history_slots(self) -> None:
        state = PPCHState(2)

        cold = state.preview([0.0, 0.0], [False, False], [False, False])
        state.commit([0.0, 0.0], [False, False], [False, False], decision=cold)
        after_zero_slot = state.preview([0.0, 0.0], [False, False], [False, False])

        self.assertEqual(cold.sliding_weights, (0.2, 0.2))
        self.assertEqual(after_zero_slot.sliding_weights, (0.0, 0.0))

    def test_window_length_and_decay_use_only_the_latest_history_slots(self) -> None:
        state = PPCHState(
            1,
            PPCHConfig(window_length=2, decay=0.5),
        )
        state.update([0.0], [True], [True])  # rho = 1
        state.update([1.0], [True], [True])  # rho = 1/2
        state.update([3.0], [True], [True])  # rho = 1/4

        decision = state.preview([0.0], [True], [True])

        expected = (0.5 * 0.5 + 0.25) / (0.5 + 1.0)
        self.assertAlmostEqual(decision.sliding_weights[0], expected)
        self.assertEqual(len(state.history), 2)


class PPCHStateTransitionTests(unittest.TestCase):
    def test_preview_and_clone_do_not_mutate_the_source_state(self) -> None:
        state = PPCHState(2)
        state.update([0.0, 2.0], [True, True], [True, True])
        before = (state.round_count, state.previous_weights, state.history)

        first_preview = state.preview([1.0, 3.0], [True, True], [True, True])
        second_preview = state.preview([1.0, 3.0], [True, True], [True, True])
        clone = state.clone()
        clone.update([0.0, 0.0], [True, True], [True, True])

        self.assertEqual(first_preview, second_preview)
        self.assertEqual((state.round_count, state.previous_weights, state.history), before)
        self.assertNotEqual(clone.round_count, state.round_count)
        self.assertNotEqual(clone.history, state.history)

    def test_normal_commit_updates_previous_weights_and_records_all_valid_reports(self) -> None:
        state = PPCHState(4)
        scores = [0.0, 0.0, 0.0, 10.0]
        masks = [True] * 4

        decision = state.preview(scores, masks, masks)
        result = state.commit(scores, masks, masks, decision)

        self.assertIsNone(result)
        self.assertTrue(decision.valid_update)
        self.assertAlmostEqual(sum(decision.final_weights), 1.0)
        self.assertEqual(state.previous_weights, decision.final_weights)
        self.assertEqual(state.round_count, 1)
        self.assertEqual(state.history[0][:3], (1.0, 1.0, 1.0))
        self.assertAlmostEqual(state.history[0][3], 1.0 / 11.0)
        self.assertFalse(decision.survivors[3])

    def test_empty_stage_one_keeps_previous_weights_and_commits_zero_history(self) -> None:
        state = PPCHState(3)
        before = state.previous_weights

        decision = state.update(
            [0.0, 1.0, 2.0],
            [False, False, False],
            [False, False, False],
        )

        self.assertFalse(decision.valid_update)
        self.assertEqual(decision.survivors, (False, False, False))
        self.assertEqual(decision.final_weights, (0.0, 0.0, 0.0))
        self.assertEqual(state.previous_weights, before)
        self.assertEqual(state.history, ((0.0, 0.0, 0.0),))
        self.assertIsNone(decision.stage1_center)
        self.assertIsNone(decision.stage2_center)

    def test_zero_raw_sum_keeps_previous_weights_and_commits_zero_history(self) -> None:
        state = PPCHState(2)
        empty = state.preview([0.0, 0.0], [False, False], [False, False])
        state.commit([0.0, 0.0], [False, False], [False, False], empty)
        before = state.previous_weights

        decision = state.update([0.0, 0.0], [True, True], [True, True])

        self.assertEqual(decision.sliding_weights, (0.0, 0.0))
        self.assertEqual(decision.hampel_weights, (1.0, 1.0))
        self.assertEqual(decision.raw_weights, (0.0, 0.0))
        self.assertFalse(decision.valid_update)
        self.assertEqual(state.previous_weights, before)
        self.assertEqual(state.history[-1], (0.0, 0.0))

    def test_commit_rejects_a_decision_from_different_inputs(self) -> None:
        state = PPCHState(2)
        decision = state.preview([0.0, 0.0], [True, True], [True, True])

        with self.assertRaises(ValueError):
            state.commit([0.0, 3.0], [True, True], [True, True], decision)


class PPCHValidationTests(unittest.TestCase):
    def test_worker_count_and_config_type_are_strictly_validated(self) -> None:
        for worker_count in (0, -1, True, 1.5):
            with self.subTest(worker_count=worker_count), self.assertRaises((TypeError, ValueError)):
                PPCHState(worker_count)  # type: ignore[arg-type]

        with self.assertRaises(TypeError):
            PPCHState(2, config={})  # type: ignore[arg-type]

    def test_dimensions_scores_and_boolean_masks_are_strictly_validated(self) -> None:
        state = PPCHState(2)
        valid = ([0.0, 1.0], [True, True], [True, True])
        invalid_cases = (
            ([0.0], valid[1], valid[2]),
            (valid[0], [True], valid[2]),
            (valid[0], valid[1], [True]),
            ([-1.0, 0.0], valid[1], valid[2]),
            ([math.inf, 0.0], valid[1], valid[2]),
            ([math.nan, 0.0], valid[1], valid[2]),
            ([True, 0.0], valid[1], valid[2]),
            (valid[0], [1, True], valid[2]),
            (valid[0], valid[1], [True, 0]),
            ("01", valid[1], valid[2]),
        )

        for scores, present, verified in invalid_cases:
            with (
                self.subTest(scores=scores, present=present, verified=verified),
                self.assertRaises((TypeError, ValueError)),
            ):
                state.preview(scores, present, verified)  # type: ignore[arg-type]

    def test_attack_labels_do_not_exist_in_the_public_api(self) -> None:
        public_methods = (PPCHState.preview, PPCHState.commit, PPCHState.update)
        for method in public_methods:
            parameter_names = set(inspect.signature(method).parameters)
            self.assertFalse(
                any("attack" in name.lower() for name in parameter_names),
                msg=f"{method.__name__} 暴露了攻击标签参数",
            )

        config_fields = {field.name.lower() for field in fields(PPCHConfig)}
        decision_fields = {field.name.lower() for field in fields(PPCHDecision)}
        self.assertFalse(any("attack" in name for name in config_fields))
        self.assertFalse(any("attack" in name for name in decision_fields))


if __name__ == "__main__":
    unittest.main()
