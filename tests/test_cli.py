"""CLI 配置和证明自检命令的边界回归测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from brptd.cli import _load_attack_config, _load_bucket_config, _proof_validate
from brptd.proofs import ProofVerificationError, ResidualBinProof, VerifiedResidualBin


class ExperimentConfigLoadingTests(unittest.TestCase):
    def test_versioned_toml_values_reach_experiment_config(self) -> None:
        root = Path(__file__).resolve().parents[1]
        attack_config = _load_attack_config(root / "configs" / "paper_attacks.toml", None)
        bucket_config = _load_bucket_config(root / "configs" / "bucket_sensitivity.toml", None)

        self.assertEqual((0.40, 0.55, 0.70, 0.85), attack_config.outer_starts)
        self.assertEqual(0.15, attack_config.outer_fraction)
        self.assertEqual(15, attack_config.embargo_rounds)
        self.assertEqual(5, attack_config.blocks_per_fold)
        self.assertEqual(1.8, attack_config.attack_parameters.bias_offset)
        self.assertEqual(0.13, attack_config.attack_parameters.drift_per_round)
        self.assertEqual(tuple(2**power for power in range(3, 14)), bucket_config.bucket_counts)
        self.assertEqual(0.85, bucket_config.attack_parameters.spike_upper_probability)


class ProofValidateTests(unittest.TestCase):
    def test_validate_records_positive_and_tampered_negative_outcomes(self) -> None:
        verified = VerifiedResidualBin(
            label=2,
            lower=16,
            upper=23,
            proxy=23,
            ciphertext_digest="a" * 64,
            context_digest="b" * 64,
        )
        proof = ResidualBinProof(b"proof-payload")
        keypair = SimpleNamespace(public_key=b"p" * 32)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "proof_validate.json"
            with (
                patch("brptd.proofs.testing.generate_test_keypair", return_value=keypair),
                patch("brptd.proofs.testing.encrypt_measurements_deterministic", return_value=object()),
                patch("brptd.proofs.testing.prove_residual_bin_deterministic", return_value=proof),
                patch("brptd.proofs.build_proof_context", return_value=b"context"),
                patch(
                    "brptd.proofs.verify_residual_bin",
                    side_effect=[verified, ProofVerificationError("tampered")],
                ),
            ):
                self.assertEqual(0, _proof_validate(target))
            manifest = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual("verified", manifest["positive_status"])
        self.assertEqual("rejected", manifest["tamper_status"])
        self.assertEqual(2, manifest["label"])


if __name__ == "__main__":
    unittest.main()
