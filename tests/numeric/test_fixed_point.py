"""定点编码与规范正负分解的回归测试。"""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError
from decimal import Decimal

from brptd.numeric import (
    FixedPointConfig,
    canonical_split,
    decode,
    encode,
    round_half_away_from_zero,
)


class FixedPointConfigTests(unittest.TestCase):
    def test_defaults_match_protocol_scales_and_ristretto_order(self) -> None:
        config = FixedPointConfig()

        self.assertEqual(config.measurement_scale, 1_000_000)
        self.assertEqual(config.coefficient_scale, 1_000_000)
        self.assertEqual(config.normalization_scale, 10**12)
        self.assertEqual(config.range_bits, 64)
        self.assertEqual(
            config.scalar_order,
            2**252 + 27742317777372353535851937790883648493,
        )

    def test_config_is_frozen(self) -> None:
        config = FixedPointConfig()

        with self.assertRaises(FrozenInstanceError):
            config.measurement_scale = 10  # type: ignore[misc]

    def test_config_rejects_invalid_scales_and_range_relation(self) -> None:
        invalid_arguments = (
            {"measurement_scale": 0},
            {"measurement_scale": True},
            {"coefficient_scale": -1},
            {"range_bits": 0},
            {"range_bits": 252},
            {"range_bits": 3, "scalar_order": 14},
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                FixedPointConfig(**arguments)


class DecimalRoundingTests(unittest.TestCase):
    def test_rounds_half_values_away_from_zero(self) -> None:
        cases = (
            (Decimal("0"), 0),
            (Decimal("-0"), 0),
            (Decimal("0.49"), 0),
            (Decimal("-0.49"), 0),
            (Decimal("0.5"), 1),
            (Decimal("-0.5"), -1),
            (Decimal("1.5"), 2),
            (Decimal("-1.5"), -2),
        )

        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(round_half_away_from_zero(value), expected)

    def test_encode_and_decode_preserve_protocol_precision(self) -> None:
        self.assertEqual(encode(Decimal("0.0000005")), 1)
        self.assertEqual(encode(Decimal("-0.0000005")), -1)
        self.assertEqual(encode(Decimal("1.2345674")), 1_234_567)
        self.assertEqual(decode(1_500_000), Decimal("1.5"))
        self.assertEqual(decode(-1), Decimal("-0.000001"))

    def test_numeric_helpers_reject_nonfinite_or_ambiguous_values(self) -> None:
        for value in (Decimal("NaN"), Decimal("Infinity"), float("-inf"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                encode(value)

        for value in (True, 1.5, "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                decode(value)  # type: ignore[arg-type]

        with self.assertRaises(ValueError):
            encode(1, scale=0)
        with self.assertRaises(ValueError):
            decode(1, scale=True)


class CanonicalSplitTests(unittest.TestCase):
    def test_split_is_unique_and_mutually_exclusive(self) -> None:
        for difference in (-10, -1, 0, 1, 10):
            with self.subTest(difference=difference):
                positive, negative = canonical_split(difference)
                self.assertEqual(positive - negative, difference)
                self.assertEqual(positive * negative, 0)
                self.assertEqual(positive + negative, abs(difference))

    def test_split_rejects_nonintegers_and_bool(self) -> None:
        for value in (True, 1.0, Decimal("1"), "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                canonical_split(value)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
