"""公开残差上界与固定宽度桶的回归测试。"""

from __future__ import annotations

import random
import unittest
from dataclasses import FrozenInstanceError
from decimal import Decimal

from brptd.numeric import (
    FixedPointConfig,
    ResidualBinConfig,
    bucket_interval,
    compute_residual_bin,
)


class ResidualBinConfigTests(unittest.TestCase):
    def test_standardized_domains_encode_all_public_parameters(self) -> None:
        config = ResidualBinConfig.from_standardized_domains(
            ((Decimal("-1.5"), Decimal("2.5")), (0, 10)),
            bin_count=8,
            etas=(Decimal("0.5"), 2),
        )

        self.assertEqual(
            config.domains,
            ((-1_500_000, 2_500_000), (0, 10_000_000)),
        )
        self.assertEqual(config.etas, (500_000, 2_000_000))
        self.assertEqual(config.err_max, 22_000_000_000_000)
        self.assertEqual(config.delta_bin, 2_750_000_000_001)
        self.assertEqual(config.bin_count, 8)
        self.assertEqual(config.normalization_scale, 10**12)

    def test_default_eta_is_one_for_every_coordinate(self) -> None:
        config = ResidualBinConfig.from_standardized_domains(((0, 1), (2, 3)))

        self.assertEqual(config.etas, (1_000_000, 1_000_000))

    def test_config_is_frozen(self) -> None:
        config = ResidualBinConfig.from_standardized_domains(((0, 1),))

        with self.assertRaises(FrozenInstanceError):
            config.delta_bin = 1  # type: ignore[misc]

    def test_config_rejects_bad_domains_etas_and_bin_count(self) -> None:
        bad_domain_sets = (
            (),
            ((0,),),
            ((0, 1, 2),),
            ((1, 0),),
            ((1, 1),),
            ((0, Decimal("NaN")),),
            ((0, Decimal("Infinity")),),
            ((0, Decimal("0.0000004")),),
        )
        for domains in bad_domain_sets:
            with self.subTest(domains=domains), self.assertRaises(ValueError):
                ResidualBinConfig.from_standardized_domains(domains)

        bad_eta_sets = ((1,), (1, 2, 3), (0, 1), (-1, 1), (Decimal("NaN"), 1))
        for etas in bad_eta_sets:
            with self.subTest(etas=etas), self.assertRaises(ValueError):
                ResidualBinConfig.from_standardized_domains(((0, 1), (0, 1)), etas=etas)

        for bin_count in (True, 0, -1, 1.5):
            with self.subTest(bin_count=bin_count), self.assertRaises(ValueError):
                ResidualBinConfig.from_standardized_domains(
                    ((0, 1),),
                    bin_count=bin_count,  # type: ignore[arg-type]
                )

    def test_config_detects_u64_product_and_sum_overflow(self) -> None:
        with self.assertRaisesRegex(OverflowError, "u64"):
            ResidualBinConfig.from_standardized_domains(((0, 20_000_000),))

        with self.assertRaisesRegex(OverflowError, "u64"):
            ResidualBinConfig.from_standardized_domains(((0, 10_000_000), (0, 10_000_000)))

    def test_config_detects_range_proof_overflow(self) -> None:
        small_range = FixedPointConfig(range_bits=32)

        with self.assertRaisesRegex(OverflowError, "range_bits"):
            ResidualBinConfig.from_standardized_domains(((0, 1),), fixed_point=small_range)


class BucketTests(unittest.TestCase):
    def test_first_and_truncated_last_bucket(self) -> None:
        config = ResidualBinConfig.from_standardized_domains(
            ((0, Decimal("0.000010")),),
            bin_count=3,
            etas=(Decimal("0.000001"),),
        )
        self.assertEqual(config.err_max, 10)
        self.assertEqual(config.delta_bin, 4)

        self.assertEqual(bucket_interval(0, config), (0, 3))
        self.assertEqual(bucket_interval(2, config), (8, 10))

        first = compute_residual_bin(0, config)
        last = compute_residual_bin(10, config)
        self.assertEqual((first.label, first.lo, first.hi), (0, 0, 3))
        self.assertEqual((last.label, last.lo, last.hi), (2, 8, 10))
        self.assertEqual(first.proxy, Decimal(3) / Decimal(10**12))
        self.assertEqual(last.proxy, Decimal(10) / Decimal(10**12))
        self.assertFalse(hasattr(last, "residual"))

    def test_unit_width_bucket_has_exact_endpoint_proxy(self) -> None:
        config = ResidualBinConfig.from_standardized_domains(
            ((0, Decimal("0.000003")),),
            bin_count=4,
            etas=(Decimal("0.000001"),),
        )

        self.assertEqual(config.delta_bin, 1)
        result = compute_residual_bin(2, config)
        self.assertEqual((result.label, result.lo, result.hi), (2, 2, 2))
        self.assertEqual(result.proxy, Decimal(2) / Decimal(10**12))

    def test_bucket_functions_reject_out_of_domain_values(self) -> None:
        config = ResidualBinConfig.from_standardized_domains(
            ((0, Decimal("0.000010")),),
            bin_count=3,
            etas=(Decimal("0.000001"),),
        )
        for residual in (True, -1, 11, 1.5):
            with self.subTest(residual=residual), self.assertRaises(ValueError):
                compute_residual_bin(residual, config)  # type: ignore[arg-type]

        for label in (True, -1, 3, 1.5):
            with self.subTest(label=label), self.assertRaises(ValueError):
                bucket_interval(label, config)  # type: ignore[arg-type]

    def test_random_residual_always_lies_in_its_bucket(self) -> None:
        config = ResidualBinConfig.from_standardized_domains(
            ((-10, 20), (100, 150)),
            bin_count=8192,
            etas=(Decimal("0.000001"), Decimal("0.000002")),
        )
        generator = random.Random(20260901)

        for _ in range(400):
            residual = generator.randint(0, config.err_max)
            result = compute_residual_bin(residual, config)

            self.assertEqual(result.label, residual // config.delta_bin)
            self.assertEqual(result.lo, result.label * config.delta_bin)
            self.assertEqual(
                result.hi,
                min((result.label + 1) * config.delta_bin - 1, config.err_max),
            )
            self.assertLessEqual(result.lo, residual)
            self.assertLessEqual(residual, result.hi)


if __name__ == "__main__":
    unittest.main()
