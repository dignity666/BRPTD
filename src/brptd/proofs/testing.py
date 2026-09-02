"""仅供测试使用的确定性 Ristretto 后端辅助接口。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

from .residual_bin import (
    EncryptedBatch,
    ResidualBinProof,
    _encrypt_measurements_with_seed,
    _load_backend,
    _prove_residual_bin_with_seed,
)


@dataclass(frozen=True)
class TestKeyPair:
    """测试专用的确定性 Ristretto255 密钥对。"""

    public_key: bytes
    secret_key: bytes


def _seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("seed must be an unsigned 64-bit integer")
    normalized = int(value)
    if normalized < 0 or normalized > (1 << 64) - 1:
        raise ValueError("seed must be an unsigned 64-bit integer")
    return normalized


def generate_test_keypair(*, seed: int) -> TestKeyPair:
    """生成可复现测试密钥，禁止用于生产密钥管理。"""

    public_key, secret_key = _load_backend().generate_demo_keypair(_seed(seed))
    return TestKeyPair(bytes(public_key), bytes(secret_key))


def encrypt_measurements_deterministic(
    measurements: Sequence[int],
    public_key: bytes,
    *,
    seed: int,
) -> EncryptedBatch:
    """使用显式 seed 加密，仅用于可重复测试。"""

    return _encrypt_measurements_with_seed(measurements, public_key, _seed(seed))


def prove_residual_bin_deterministic(
    encrypted_batch: EncryptedBatch,
    truths: Sequence[int],
    etas: Sequence[int],
    measurement_domains: Sequence[Sequence[int]],
    delta_bin: int,
    public_key: bytes,
    context: bytes,
    *,
    seed: int,
) -> ResidualBinProof:
    """使用显式 seed 生成证明，仅用于可重复测试。"""

    return _prove_residual_bin_with_seed(
        encrypted_batch,
        truths,
        etas,
        measurement_domains,
        delta_bin,
        public_key,
        context,
        _seed(seed),
    )


__all__ = [
    "TestKeyPair",
    "encrypt_measurements_deterministic",
    "generate_test_keypair",
    "prove_residual_bin_deterministic",
]
