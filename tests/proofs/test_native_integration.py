"""已构建 Ristretto255 扩展的 Python 端到端回归。

该测试在独立解释器中运行，避免 ``test_residual_bin.py`` 的后端 stub 污染真实
扩展导入。未构建扩展时跳过，构建验收命令会强制执行本测试。
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path


@unittest.skipUnless(
    importlib.util.find_spec("brptd_ristretto_backend") is not None,
    "需要先构建 native/ristretto_backend PyO3 扩展",
)
class NativeProofIntegrationTests(unittest.TestCase):
    def test_native_positive_and_public_input_tampering(self) -> None:
        root = Path(__file__).resolve().parents[2]
        script = r"""
from brptd.proofs import ProofVerificationError, ResidualBinProof, build_proof_context, verify_residual_bin
from brptd.proofs.testing import (
    encrypt_measurements_deterministic,
    generate_test_keypair,
    prove_residual_bin_deterministic,
)

keypair = generate_test_keypair(seed=20260910)
measurements = (26, 65, 72)
truths = (25, 80, 60)
etas = (4, 1, 1)
domains = ((24, 27), (60, 85), (55, 75))
context = build_proof_context(
    task="native-integration",
    dataset="fixture",
    worker="worker-0",
    round_id=0,
    truths=truths,
    etas=etas,
    measurement_domains=domains,
    delta_bin=8,
    public_key=keypair.public_key,
)
batch = encrypt_measurements_deterministic(measurements, keypair.public_key, seed=20260911)
proof = prove_residual_bin_deterministic(
    batch, truths, etas, domains, 8, keypair.public_key, context, seed=20260912
)
result = verify_residual_bin(proof, truths, etas, domains, 8, keypair.public_key, context)
assert (result.label, result.lower, result.upper, result.proxy) == (3, 24, 31, 31)

for changed in (
    dict(truths=(26, 80, 60)),
    dict(etas=(5, 1, 1)),
    dict(measurement_domains=((23, 27), (60, 85), (55, 75))),
    dict(context=context + b" "),
    dict(public_key=bytes([keypair.public_key[0] ^ 1]) + keypair.public_key[1:]),
):
    arguments = {
        "truths": truths,
        "etas": etas,
        "measurement_domains": domains,
        "delta_bin": 8,
        "public_key": keypair.public_key,
        "context": context,
    }
    arguments.update(changed)
    try:
        verify_residual_bin(proof, **arguments)
    except ProofVerificationError:
        pass
    else:
        raise AssertionError("modified public input was accepted")

tampered = bytearray(proof.payload)
tampered[-1] ^= 1
try:
    verify_residual_bin(ResidualBinProof(bytes(tampered)), truths, etas, domains, 8, keypair.public_key, context)
except ProofVerificationError:
    pass
else:
    raise AssertionError("modified proof bytes were accepted")
"""
        environment = dict(os.environ)
        environment["PYTHONPATH"] = f"{root / '.deps'}:{root / 'src'}"
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)


if __name__ == "__main__":
    unittest.main()
