"""残差分桶证明 Python 包装层的契约测试。"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import inspect
import json
import sys
import unittest
from pathlib import Path
from types import ModuleType


class _BackendStub(ModuleType):
    """记录包装层调用，并模拟 Rust 扩展的最小行为。"""

    def __init__(self) -> None:
        super().__init__("brptd_ristretto_backend")
        self.calls: list[tuple[object, ...]] = []
        self._batches: dict[bytes, bytes] = {}
        self._reports: dict[bytes, tuple[object, ...]] = {}

    def generate_demo_keypair(self, seed: int | None):
        self.calls.append(("generate_demo_keypair", seed))
        key_byte = 7 if seed is None else seed % 251
        return bytes([key_byte]) * 32, bytes([key_byte + 1]) * 32

    def encrypt_measurements(
        self,
        measurements: list[int],
        public_key: bytes,
        seed: int | None,
    ):
        self.calls.append(("encrypt_measurements", tuple(measurements), public_key, seed))
        state = b"state:" + bytes(value % 251 for value in measurements)
        ciphertexts = hashlib.sha512(state + public_key).digest() * len(measurements)
        self._batches[state] = ciphertexts
        return state, ciphertexts

    def prove_residual_bin(
        self,
        state: bytes,
        public_key: bytes,
        truths: list[int],
        etas: list[int],
        domains: list[int],
        delta_bin: int,
        context: bytes,
        seed: int | None,
    ):
        self.calls.append(("prove_residual_bin", seed))
        ciphertexts = self._batches[state]
        payload = b"proof" + hashlib.sha256(state + context).digest() + ciphertexts
        self._reports[payload] = (
            public_key,
            tuple(truths),
            tuple(etas),
            tuple(domains),
            delta_bin,
            context,
            ciphertexts,
        )
        return payload

    def verify_residual_bin(
        self,
        report: bytes,
        public_key: bytes,
        truths: list[int],
        etas: list[int],
        domains: list[int],
        delta_bin: int,
        context: bytes,
    ):
        expected = self._reports.get(report)
        actual = (
            public_key,
            tuple(truths),
            tuple(etas),
            tuple(domains),
            delta_bin,
            context,
        )
        if expected is None or expected[:6] != actual:
            return False, "statement mismatch", 0, 0, 0, 0
        return True, "ok", 3, 24, 31, 31

    def inspect_residual_bin_report(self, report: bytes):
        expected = self._reports[report]
        ciphertexts = expected[6]
        return [
            ("ciphertext_offset", len(report) - len(ciphertexts)),
            ("ciphertext_bytes", len(ciphertexts)),
        ]


BACKEND = _BackendStub()
sys.modules[BACKEND.__name__] = BACKEND


class ResidualBinWrapperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.proofs = importlib.import_module("brptd.proofs")
        cls.testing = importlib.import_module("brptd.proofs.testing")

    def setUp(self) -> None:
        BACKEND.calls.clear()
        BACKEND._batches.clear()
        BACKEND._reports.clear()
        self.measurements = (26, 65, 72)
        self.truths = (25, 80, 60)
        self.etas = (4, 1, 1)
        self.domains = ((24, 27), (60, 85), (55, 75))
        self.public_key = bytes(range(32))
        self.context = self.proofs.build_proof_context(
            task="truth-discovery",
            dataset="sensors-2026",
            worker="worker-7",
            round_id=4,
            truths=self.truths,
            etas=self.etas,
            measurement_domains=self.domains,
            delta_bin=8,
            public_key=self.public_key,
        )

    def _proof(self):
        batch = self.proofs.encrypt_measurements(
            self.measurements,
            self.public_key,
        )
        proof = self.proofs.prove_residual_bin(
            batch,
            self.truths,
            self.etas,
            self.domains,
            8,
            self.public_key,
            self.context,
        )
        return batch, proof

    def test_context_is_canonical_utf8_json_and_binds_required_digests(self):
        decoded = json.loads(self.context.decode("utf-8"))

        self.assertEqual(
            self.context,
            json.dumps(
                decoded,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
        )
        self.assertEqual(decoded["protocol"], "BRPTD/residual-bin")
        self.assertEqual(decoded["version"], 1)
        self.assertEqual(decoded["key_epoch"], 0)
        self.assertEqual(decoded["task"], "truth-discovery")
        self.assertEqual(decoded["dataset"], "sensors-2026")
        self.assertEqual(decoded["worker"], "worker-7")
        self.assertEqual(decoded["round"], 4)
        self.assertEqual(
            decoded["truth_digest"],
            hashlib.sha256(b"[25,80,60]").hexdigest(),
        )
        self.assertEqual(
            decoded["key_digest"],
            hashlib.sha256(self.public_key).hexdigest(),
        )
        self.assertRegex(decoded["parameters_digest"], r"^[0-9a-f]{64}$")

    def test_public_api_uses_frozen_byte_wrappers_and_has_no_seed(self):
        self.assertNotIn("seed", inspect.signature(self.proofs.encrypt_measurements).parameters)
        self.assertNotIn("seed", inspect.signature(self.proofs.prove_residual_bin).parameters)

        batch, proof = self._proof()

        self.assertIsInstance(batch, self.proofs.EncryptedBatch)
        self.assertIsInstance(proof, self.proofs.ResidualBinProof)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            batch.state = b"changed"
        with self.assertRaises(dataclasses.FrozenInstanceError):
            proof.payload = b"changed"
        self.assertEqual(BACKEND.calls[0][-1], None)
        self.assertEqual(BACKEND.calls[1][-1], None)

    def test_verify_returns_exact_six_field_frozen_result_with_digests(self):
        batch, proof = self._proof()

        result = self.proofs.verify_residual_bin(
            proof,
            self.truths,
            self.etas,
            self.domains,
            8,
            self.public_key,
            self.context,
        )

        self.assertEqual(
            [field.name for field in dataclasses.fields(result)],
            [
                "label",
                "lower",
                "upper",
                "proxy",
                "ciphertext_digest",
                "context_digest",
            ],
        )
        self.assertEqual((result.label, result.lower, result.upper, result.proxy), (3, 24, 31, 31))
        self.assertEqual(
            result.ciphertext_digest,
            hashlib.sha256(batch.ciphertexts).hexdigest(),
        )
        self.assertEqual(
            result.context_digest,
            hashlib.sha256(self.context).hexdigest(),
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.label = 9

    def test_context_and_ciphertext_tampering_raise_typed_error(self):
        _batch, proof = self._proof()

        with self.assertRaisesRegex(
            self.proofs.ProofVerificationError,
            "statement mismatch",
        ):
            self.proofs.verify_residual_bin(
                proof,
                self.truths,
                self.etas,
                self.domains,
                8,
                self.public_key,
                self.context + b" ",
            )

        tampered = bytearray(proof.payload)
        tampered[-1] ^= 1
        with self.assertRaises(self.proofs.ProofVerificationError):
            self.proofs.verify_residual_bin(
                self.proofs.ResidualBinProof(bytes(tampered)),
                self.truths,
                self.etas,
                self.domains,
                8,
                self.public_key,
                self.context,
            )

    def test_statement_digest_is_deterministic_and_sensitive_to_public_inputs(self):
        batch, _proof = self._proof()
        arguments = {
            "context": self.context,
            "ciphertexts": batch.ciphertexts,
            "truths": self.truths,
            "etas": self.etas,
            "measurement_domains": self.domains,
            "delta_bin": 8,
            "public_key": self.public_key,
            "label": 3,
        }

        digest = self.proofs.digest_statement(**arguments)

        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertEqual(digest, self.proofs.digest_statement(**arguments))
        self.assertNotEqual(
            digest,
            self.proofs.digest_statement(**{**arguments, "label": 4}),
        )

    def test_seeded_randomness_is_only_exposed_by_testing_helpers(self):
        key_pair = self.testing.generate_test_keypair(seed=37)
        context = self.proofs.build_proof_context(
            task="truth-discovery",
            dataset="sensors-2026",
            worker="worker-7",
            round_id=4,
            truths=self.truths,
            etas=self.etas,
            measurement_domains=self.domains,
            delta_bin=8,
            public_key=key_pair.public_key,
        )
        batch = self.testing.encrypt_measurements_deterministic(
            self.measurements,
            key_pair.public_key,
            seed=41,
        )
        self.testing.prove_residual_bin_deterministic(
            batch,
            self.truths,
            self.etas,
            self.domains,
            8,
            key_pair.public_key,
            context,
            seed=43,
        )

        self.assertEqual(BACKEND.calls[0], ("generate_demo_keypair", 37))
        self.assertEqual(BACKEND.calls[1][-1], 41)
        self.assertEqual(BACKEND.calls[2][-1], 43)

    def test_package_does_not_import_deleted_uppercase_brptd_package(self):
        source_root = Path(self.proofs.__file__).parent
        source = "\n".join(path.read_text(encoding="utf-8") for path in sorted(source_root.glob("*.py")))

        self.assertNotIn("from BRPTD", source)
        self.assertNotIn("import BRPTD", source)


if __name__ == "__main__":
    unittest.main()
