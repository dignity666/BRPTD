"""标准化整数残差分量与精确残差的回归测试。"""

from __future__ import annotations

import random
import unittest
from dataclasses import FrozenInstanceError
from decimal import Decimal

from brptd.numeric import (
    ResidualBinConfig,
    ResidualComponent,
    exact_residual,
    residual_components,
)


class ResidualComponentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ResidualBinConfig.from_standardized_domains(
            ((0, 10), (-5, 5)),
            bin_count=16,
            etas=(1, Decimal("0.25")),
        )

    def test_components_bind_signed_difference_and_round_bound(self) -> None:
        measurement = (9_000_000, -4_000_000)
        truth = (2_000_000, 1_000_000)

        first, second = residual_components(measurement, truth, self.config)

        self.assertEqual(first.signed, 7_000_000_000_000)
        self.assertEqual((first.positive, first.negative), (first.signed, 0))
        self.assertEqual(first.bound, 8_000_000_000_000)
        self.assertEqual(second.signed, -1_250_000_000_000)
        self.assertEqual((second.positive, second.negative), (0, 1_250_000_000_000))
        self.assertEqual(second.bound, 1_500_000_000_000)

    def test_exact_residual_sums_canonical_absolute_components(self) -> None:
        measurement = (9_000_000, -4_000_000)
        truth = (2_000_000, 1_000_000)

        result = exact_residual(measurement, truth, self.config)

        self.assertEqual(result, 8_250_000_000_000)

    def test_component_is_frozen_and_rejects_noncanonical_state(self) -> None:
        component = ResidualComponent(
            signed=4,
            positive=4,
            negative=0,
            bound=5,
        )
        with self.assertRaises(FrozenInstanceError):
            component.bound = 6  # type: ignore[misc]

        invalid_components = (
            {"signed": 1, "positive": 2, "negative": 1, "bound": 3},
            {"signed": 2, "positive": 1, "negative": 0, "bound": 3},
            {"signed": 4, "positive": 4, "negative": 0, "bound": 3},
            {"signed": 0, "positive": -1, "negative": -1, "bound": 1},
        )
        for arguments in invalid_components:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                ResidualComponent(**arguments)

    def test_residual_inputs_require_exact_dimensions_and_integers(self) -> None:
        bad_pairs = (
            ((1,), (1, 2)),
            ((1, 2), (1,)),
            ((1, 2, 3), (1, 2)),
            ((1.0, 2), (1, 2)),
            ((True, 2), (1, 2)),
        )
        for measurement, truth in bad_pairs:
            with self.subTest(measurement=measurement, truth=truth):
                with self.assertRaises(ValueError):
                    residual_components(measurement, truth, self.config)

    def test_residual_inputs_must_stay_inside_public_domains(self) -> None:
        cases = (
            ((10_000_001, 0), (0, 0), "measurement"),
            ((0, 0), (-1, 0), "truth"),
            ((0, -5_000_001), (0, 0), "measurement"),
            ((0, 0), (0, 5_000_001), "truth"),
        )
        for measurement, truth, message in cases:
            with self.subTest(measurement=measurement, truth=truth):
                with self.assertRaisesRegex(ValueError, message):
                    exact_residual(measurement, truth, self.config)


class ResidualPropertyTests(unittest.TestCase):
    def test_deterministic_random_components_preserve_all_relations(self) -> None:
        config = ResidualBinConfig.from_standardized_domains(
            ((-2, 4), (10, 20), (Decimal("-0.5"), Decimal("0.5"))),
            bin_count=257,
            etas=(1, Decimal("0.25"), 2),
        )
        generator = random.Random(20260901)

        for _ in range(400):
            measurement = tuple(generator.randint(lower, upper) for lower, upper in config.domains)
            truth = tuple(generator.randint(lower, upper) for lower, upper in config.domains)
            components = residual_components(measurement, truth, config)

            for index, component in enumerate(components):
                expected = config.etas[index] * (measurement[index] - truth[index])
                self.assertEqual(component.signed, expected)
                self.assertEqual(component.positive - component.negative, expected)
                self.assertEqual(component.positive * component.negative, 0)
                self.assertLessEqual(component.positive + component.negative, component.bound)
            self.assertEqual(
                exact_residual(measurement, truth, config),
                sum(item.positive + item.negative for item in components),
            )


if __name__ == "__main__":
    unittest.main()
