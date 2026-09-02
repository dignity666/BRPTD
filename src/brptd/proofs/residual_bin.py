"""Ristretto255 EC ElGamal 残差分桶证明包装层。

密码学运算和证明报文解析均由 ``brptd_ristretto_backend`` 完成。本模块
只负责公开输入规范化、上下文绑定、失败类型转换和审计摘要计算。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from numbers import Integral
from typing import Any, NoReturn

_CONTEXT_PROTOCOL = "BRPTD/residual-bin"
_CONTEXT_VERSION = 1
_CIPHERTEXT_BYTES_PER_DIMENSION = 64
_DIGEST_HEX_LENGTH = 64


class ProofVerificationError(RuntimeError):
    """证明或其绑定公开语句验证失败。"""

    def __init__(self, reason: str) -> None:
        self.reason = str(reason)
        super().__init__(self.reason)


@dataclass(frozen=True)
class EncryptedBatch:
    """本地加密状态及对应的公开 EC ElGamal 密文。"""

    state: bytes
    ciphertexts: bytes
    dimension_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.state, bytes) or not self.state:
            raise ValueError("state must be nonempty bytes")
        if not isinstance(self.dimension_count, int) or isinstance(self.dimension_count, bool):
            raise TypeError("dimension_count must be an integer")
        if self.dimension_count <= 0:
            raise ValueError("dimension_count must be positive")
        expected = self.dimension_count * _CIPHERTEXT_BYTES_PER_DIMENSION
        if not isinstance(self.ciphertexts, bytes) or len(self.ciphertexts) != expected:
            raise ValueError("ciphertexts must contain two 32-byte Ristretto points per dimension")


@dataclass(frozen=True)
class ResidualBinProof:
    """可传输的残差分桶证明报文。"""

    payload: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.payload, bytes) or not self.payload:
            raise ValueError("payload must be nonempty bytes")


@dataclass(frozen=True)
class VerifiedResidualBin:
    """验证后的公开桶结果，不包含明文测量或开启见证。"""

    label: int
    lower: int
    upper: int
    proxy: int
    ciphertext_digest: str
    context_digest: str


def _load_backend() -> Any:
    try:
        return import_module("brptd_ristretto_backend")
    except ImportError as exc:
        raise ImportError(
            "Ristretto backend is unavailable; build native/ristretto_backend "
            "with maturin before using proof operations"
        ) from exc


def _integer_vector(name: str, values: Sequence[int]) -> tuple[int, ...]:
    try:
        raw_values = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of integers") from exc
    if not raw_values:
        raise ValueError(f"{name} must be nonempty")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in raw_values):
        raise ValueError(f"{name} must contain integers")
    return tuple(int(value) for value in raw_values)


def _positive_vector(name: str, values: Sequence[int]) -> tuple[int, ...]:
    result = _integer_vector(name, values)
    if any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain positive integers")
    return result


def _domains(
    measurement_domains: Sequence[Sequence[int]],
) -> tuple[tuple[int, int], ...]:
    try:
        raw_domains = tuple(tuple(domain) for domain in measurement_domains)
    except TypeError as exc:
        raise TypeError("measurement_domains must contain endpoint pairs") from exc
    if not raw_domains:
        raise ValueError("measurement_domains must be nonempty")
    result: list[tuple[int, int]] = []
    for index, domain in enumerate(raw_domains):
        if len(domain) != 2:
            raise ValueError(f"measurement domain at coordinate {index} must have two endpoints")
        lower, upper = domain
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in domain):
            raise ValueError(f"measurement domain at coordinate {index} must contain integers")
        normalized = (int(lower), int(upper))
        if normalized[0] >= normalized[1]:
            raise ValueError(f"measurement domain at coordinate {index} must be strictly increasing")
        result.append(normalized)
    return tuple(result)


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return normalized


def _nonnegative_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a nonnegative integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return normalized


def _identifier(name: str, value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _round_identifier(value: object) -> int | str:
    if isinstance(value, bool):
        raise ValueError("round_id must be a nonnegative integer or nonempty string")
    if isinstance(value, Integral):
        return _nonnegative_integer("round_id", value)
    if isinstance(value, str) and value:
        return value
    raise ValueError("round_id must be a nonnegative integer or nonempty string")


def _public_key(value: bytes) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError("public_key must be bytes-like")
    normalized = bytes(value)
    if len(normalized) != 32:
        raise ValueError("public_key must be a 32-byte compressed Ristretto point")
    return normalized


def _bytes(name: str, value: bytes, *, nonempty: bool = True) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{name} must be bytes-like")
    normalized = bytes(value)
    if nonempty and not normalized:
        raise ValueError(f"{name} must be nonempty")
    return normalized


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _normalize_statement_inputs(
    truths: Sequence[int],
    etas: Sequence[int],
    measurement_domains: Sequence[Sequence[int]],
    delta_bin: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[int, int], ...], int]:
    truth_values = _integer_vector("truths", truths)
    eta_values = _positive_vector("etas", etas)
    domain_values = _domains(measurement_domains)
    if len(truth_values) != len(eta_values) or len(truth_values) != len(domain_values):
        raise ValueError("truths, etas, and measurement_domains must have equal length")
    return (
        truth_values,
        eta_values,
        domain_values,
        _positive_integer("delta_bin", delta_bin),
    )


def _truth_digest(truths: tuple[int, ...]) -> str:
    return _sha256(_canonical_json(list(truths)))


def _parameters_digest(
    etas: tuple[int, ...],
    domains: tuple[tuple[int, int], ...],
    delta_bin: int,
) -> str:
    return _sha256(
        _canonical_json(
            {
                "delta_bin": delta_bin,
                "etas": list(etas),
                "measurement_domains": [list(domain) for domain in domains],
            }
        )
    )


def build_proof_context(
    *,
    task: str,
    dataset: str,
    worker: str,
    round_id: int | str,
    truths: Sequence[int],
    etas: Sequence[int],
    measurement_domains: Sequence[Sequence[int]],
    delta_bin: int,
    public_key: bytes,
    protocol: str = _CONTEXT_PROTOCOL,
    version: int = _CONTEXT_VERSION,
    key_epoch: int = 0,
) -> bytes:
    """构造排序键、紧凑分隔且采用 UTF-8 编码的证明上下文。"""

    truth_values, eta_values, domains, bin_width = _normalize_statement_inputs(
        truths,
        etas,
        measurement_domains,
        delta_bin,
    )
    key = _public_key(public_key)
    return _canonical_json(
        {
            "dataset": _identifier("dataset", dataset),
            "key_digest": _sha256(key),
            "key_epoch": _nonnegative_integer("key_epoch", key_epoch),
            "parameters_digest": _parameters_digest(eta_values, domains, bin_width),
            "protocol": _identifier("protocol", protocol),
            "round": _round_identifier(round_id),
            "task": _identifier("task", task),
            "truth_digest": _truth_digest(truth_values),
            "version": _positive_integer("version", version),
            "worker": _identifier("worker", worker),
        }
    )


def _validate_context_binding(
    context: bytes,
    truths: tuple[int, ...],
    etas: tuple[int, ...],
    domains: tuple[tuple[int, int], ...],
    delta_bin: int,
    public_key: bytes,
) -> bytes:
    encoded = _bytes("context", context)
    try:
        decoded = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("context must be valid UTF-8 JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError("context must be a JSON object")
    if _canonical_json(decoded) != encoded:
        raise ValueError("context must use canonical JSON encoding")
    expected = {
        "truth_digest": _truth_digest(truths),
        "parameters_digest": _parameters_digest(etas, domains, delta_bin),
        "key_digest": _sha256(public_key),
    }
    for key, value in expected.items():
        if decoded.get(key) != value:
            raise ValueError(f"context {key} does not match public inputs")
    for key in ("task", "dataset", "worker", "round", "protocol", "version", "key_epoch"):
        if key not in decoded:
            raise ValueError(f"context is missing required field {key}")
    return encoded


def digest_statement(
    *,
    context: bytes,
    ciphertexts: bytes,
    truths: Sequence[int],
    etas: Sequence[int],
    measurement_domains: Sequence[Sequence[int]],
    delta_bin: int,
    public_key: bytes,
    label: int,
) -> str:
    """计算验证方可独立重建的公开语句 SHA-256 摘要。"""

    truth_values, eta_values, domains, bin_width = _normalize_statement_inputs(
        truths,
        etas,
        measurement_domains,
        delta_bin,
    )
    key = _public_key(public_key)
    encoded_context = _validate_context_binding(
        context,
        truth_values,
        eta_values,
        domains,
        bin_width,
        key,
    )
    ciphertext_bytes = _bytes("ciphertexts", ciphertexts)
    return _sha256(
        _canonical_json(
            {
                "ciphertext_digest": _sha256(ciphertext_bytes),
                "context_digest": _sha256(encoded_context),
                "key_digest": _sha256(key),
                "label": _nonnegative_integer("label", label),
                "parameters_digest": _parameters_digest(eta_values, domains, bin_width),
                "protocol": "BRPTD/residual-bin/statement",
                "truth_digest": _truth_digest(truth_values),
                "version": 1,
            }
        )
    )


def _flatten_domains(domains: tuple[tuple[int, int], ...]) -> list[int]:
    return [endpoint for domain in domains for endpoint in domain]


def _encrypt_measurements_with_seed(
    measurements: Sequence[int],
    public_key: bytes,
    seed: int | None,
) -> EncryptedBatch:
    values = _integer_vector("measurements", measurements)
    key = _public_key(public_key)
    state, ciphertexts = _load_backend().encrypt_measurements(list(values), key, seed)
    return EncryptedBatch(bytes(state), bytes(ciphertexts), len(values))


def encrypt_measurements(
    measurements: Sequence[int],
    public_key: bytes,
) -> EncryptedBatch:
    """使用后端操作系统随机源加密测量。"""

    return _encrypt_measurements_with_seed(measurements, public_key, None)


def _prove_residual_bin_with_seed(
    encrypted_batch: EncryptedBatch,
    truths: Sequence[int],
    etas: Sequence[int],
    measurement_domains: Sequence[Sequence[int]],
    delta_bin: int,
    public_key: bytes,
    context: bytes,
    seed: int | None,
) -> ResidualBinProof:
    if not isinstance(encrypted_batch, EncryptedBatch):
        raise TypeError("encrypted_batch must be an EncryptedBatch")
    truth_values, eta_values, domains, bin_width = _normalize_statement_inputs(
        truths,
        etas,
        measurement_domains,
        delta_bin,
    )
    if encrypted_batch.dimension_count != len(truth_values):
        raise ValueError("encrypted batch and public vectors have different dimensions")
    key = _public_key(public_key)
    encoded_context = _validate_context_binding(
        context,
        truth_values,
        eta_values,
        domains,
        bin_width,
        key,
    )
    payload = _load_backend().prove_residual_bin(
        encrypted_batch.state,
        key,
        list(truth_values),
        list(eta_values),
        _flatten_domains(domains),
        bin_width,
        encoded_context,
        seed,
    )
    proof = ResidualBinProof(bytes(payload))
    if _extract_ciphertexts(proof) != encrypted_batch.ciphertexts:
        raise RuntimeError("proof ciphertexts do not match the encrypted batch")
    return proof


def prove_residual_bin(
    encrypted_batch: EncryptedBatch,
    truths: Sequence[int],
    etas: Sequence[int],
    measurement_domains: Sequence[Sequence[int]],
    delta_bin: int,
    public_key: bytes,
    context: bytes,
) -> ResidualBinProof:
    """使用后端操作系统随机源生成测量域和残差桶范围证明。"""

    return _prove_residual_bin_with_seed(
        encrypted_batch,
        truths,
        etas,
        measurement_domains,
        delta_bin,
        public_key,
        context,
        None,
    )


def _verification_failure(reason: object) -> NoReturn:
    raise ProofVerificationError(str(reason))


def _extract_ciphertexts(proof: ResidualBinProof) -> bytes:
    metadata = {str(name): int(value) for name, value in _load_backend().inspect_residual_bin_report(proof.payload)}
    try:
        offset = metadata["ciphertext_offset"]
        length = metadata["ciphertext_bytes"]
    except KeyError as exc:
        raise ValueError("backend report metadata omits ciphertext bounds") from exc
    end = offset + length
    if offset < 0 or length <= 0 or end > len(proof.payload):
        raise ValueError("backend report metadata contains invalid ciphertext bounds")
    return proof.payload[offset:end]


def verify_residual_bin(
    proof: ResidualBinProof,
    truths: Sequence[int],
    etas: Sequence[int],
    measurement_domains: Sequence[Sequence[int]],
    delta_bin: int,
    public_key: bytes,
    context: bytes,
) -> VerifiedResidualBin:
    """验证证明并返回最小公开桶结果及审计摘要。"""

    try:
        if not isinstance(proof, ResidualBinProof):
            raise TypeError("proof must be a ResidualBinProof")
        truth_values, eta_values, domains, bin_width = _normalize_statement_inputs(
            truths,
            etas,
            measurement_domains,
            delta_bin,
        )
        key = _public_key(public_key)
        encoded_context = _validate_context_binding(
            context,
            truth_values,
            eta_values,
            domains,
            bin_width,
            key,
        )
        response = _load_backend().verify_residual_bin(
            proof.payload,
            key,
            list(truth_values),
            list(eta_values),
            _flatten_domains(domains),
            bin_width,
            encoded_context,
        )
        if not isinstance(response, tuple) or len(response) != 6:
            raise ValueError("backend returned an invalid verification response")
        ok, reason, label, lower, upper, proxy = response
        if not ok:
            _verification_failure(reason)
        ciphertexts = _extract_ciphertexts(proof)
        result = VerifiedResidualBin(
            int(label),
            int(lower),
            int(upper),
            int(proxy),
            _sha256(ciphertexts),
            _sha256(encoded_context),
        )
        if result.lower > result.upper or not result.lower <= result.proxy <= result.upper:
            raise ValueError("backend returned inconsistent residual-bin bounds")
        return result
    except ProofVerificationError:
        raise
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        raise ProofVerificationError(f"statement mismatch: {exc}") from exc
