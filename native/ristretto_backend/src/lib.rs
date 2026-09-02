//! BRPTD 残差分桶证明后端。
//!
//! 群运算、Bulletproof、转录处理和报文解析集中在 Rust 中完成。Python 侧只接收
//! 规范字节串和验证元数据。报文格式保持显式定义，便于核对通信量。

use std::convert::TryInto;

use bulletproofs::{BulletproofGens, PedersenGens, RangeProof};
use curve25519_dalek::constants::RISTRETTO_BASEPOINT_POINT;
use curve25519_dalek::ristretto::{CompressedRistretto, RistrettoPoint};
use curve25519_dalek::scalar::Scalar;
use curve25519_dalek::traits::Identity;
use merlin::Transcript;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rand::RngCore;
use rand_chacha::ChaCha20Rng;
use rand_core::{OsRng, SeedableRng};
use sha2::{Digest, Sha256, Sha512};
use thiserror::Error;

const POINT_BYTES: usize = 32;
const SCALAR_BYTES: usize = 32;
const PACKET_MAGIC: &[u8; 8] = b"BRPTDBP1";
const STATE_MAGIC: &[u8; 8] = b"BRPTENC1";
const TRANSCRIPT_DOMAIN: &[u8] = b"BRPTD/residual-bin/v1";
const MESSAGE_BASE_DOMAIN: &[u8] = b"BRPTD/Ristretto/message-base/v1";
const BLIND_BASE_DOMAIN: &[u8] = b"BRPTD/Ristretto/blind-base/v1";

#[derive(Debug, Error)]
enum BackendError {
    #[error("invalid input: {0}")]
    Input(String),
    #[error("malformed encoding: {0}")]
    Encoding(String),
    #[error("proof verification failed: {0}")]
    Verification(String),
    #[error("Bulletproof error: {0}")]
    Bulletproof(String),
}

type BackendResult<T> = Result<T, BackendError>;

impl From<BackendError> for PyErr {
    fn from(error: BackendError) -> Self {
        match error {
            BackendError::Input(message) | BackendError::Encoding(message) => {
                PyValueError::new_err(message)
            }
            BackendError::Verification(message) | BackendError::Bulletproof(message) => {
                PyRuntimeError::new_err(message)
            }
        }
    }
}

#[derive(Clone)]
struct Ciphertext {
    u: RistrettoPoint,
    v: RistrettoPoint,
}

#[derive(Clone)]
struct ResidualCommitments {
    plus: RistrettoPoint,
    minus: RistrettoPoint,
}

#[derive(Clone)]
struct Opening {
    value: i64,
    mask: Scalar,
}

struct BatchState {
    openings: Vec<Opening>,
    ciphertexts: Vec<Ciphertext>,
}

struct ParsedReport {
    dimension_count: usize,
    bin_label: u64,
    ciphertexts: Vec<Ciphertext>,
    commitments: Vec<ResidualCommitments>,
    composed: ComposedProof,
    domain_range_bytes: Vec<u8>,
    residual_range_bytes: Vec<u8>,
    header_bytes: usize,
    ciphertext_offset: usize,
    commitment_offset: usize,
    composed_offset: usize,
}

struct PublicInputs {
    truths: Vec<i64>,
    etas: Vec<u64>,
    domains: Vec<(i64, i64)>,
    residual_bounds: Vec<u64>,
    round_residual_bound: u64,
    residual_max: u64,
    bin_label: u64,
    interval_lower: u64,
    interval_upper: u64,
    delta_bin: u64,
}

#[derive(Clone)]
struct RelationResponse {
    x: Scalar,
    mask: Scalar,
    delta_mask: Scalar,
}

#[derive(Clone)]
struct OrResponse {
    challenge_plus: Scalar,
    response_plus: Scalar,
    response_minus: Scalar,
}

#[derive(Clone)]
struct ComposedProof {
    challenge: Scalar,
    relations: Vec<RelationResponse>,
    zero_sides: Vec<OrResponse>,
}

type RangeFamily = (Vec<u64>, Vec<Scalar>, Vec<RistrettoPoint>, usize);

fn message_base() -> RistrettoPoint {
    RistrettoPoint::hash_from_bytes::<Sha512>(MESSAGE_BASE_DOMAIN)
}

fn blind_base() -> RistrettoPoint {
    RistrettoPoint::hash_from_bytes::<Sha512>(BLIND_BASE_DOMAIN)
}

fn encode_point(point: &RistrettoPoint) -> [u8; POINT_BYTES] {
    point.compress().to_bytes()
}

fn decode_point(bytes: &[u8]) -> BackendResult<RistrettoPoint> {
    let encoded: [u8; POINT_BYTES] = bytes
        .try_into()
        .map_err(|_| BackendError::Encoding("Ristretto point must be 32 bytes".into()))?;
    CompressedRistretto(encoded)
        .decompress()
        .ok_or_else(|| BackendError::Encoding("invalid Ristretto point encoding".into()))
}

fn encode_scalar(scalar: &Scalar) -> [u8; SCALAR_BYTES] {
    scalar.to_bytes()
}

fn decode_scalar(bytes: &[u8]) -> BackendResult<Scalar> {
    let encoded: [u8; SCALAR_BYTES] = bytes
        .try_into()
        .map_err(|_| BackendError::Encoding("scalar must be 32 bytes".into()))?;
    Option::<Scalar>::from(Scalar::from_canonical_bytes(encoded))
        .ok_or_else(|| BackendError::Encoding("scalar encoding is not canonical".into()))
}

fn scalar_from_i64(value: i64) -> Scalar {
    if value >= 0 {
        Scalar::from(value as u64)
    } else {
        -Scalar::from(value.unsigned_abs())
    }
}

fn derive_rng(seed: Option<u64>) -> BackendResult<ChaCha20Rng> {
    let mut material = [0u8; 32];
    match seed {
        Some(seed) => {
            let mut hasher = Sha256::new();
            hasher.update(b"BRPTD/test-rng/v1");
            hasher.update(seed.to_le_bytes());
            material.copy_from_slice(&hasher.finalize());
        }
        None => OsRng.fill_bytes(&mut material),
    }
    Ok(ChaCha20Rng::from_seed(material))
}

fn append_len_prefixed(hasher: &mut Sha512, label: &[u8], value: &[u8]) {
    hasher.update((label.len() as u64).to_le_bytes());
    hasher.update(label);
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value);
}

fn append_i64(hasher: &mut Sha512, label: &[u8], value: i64) {
    append_len_prefixed(hasher, label, &value.to_le_bytes());
}

fn append_u64(hasher: &mut Sha512, label: &[u8], value: u64) {
    append_len_prefixed(hasher, label, &value.to_le_bytes());
}

fn append_point(hasher: &mut Sha512, label: &[u8], point: &RistrettoPoint) {
    append_len_prefixed(hasher, label, &encode_point(point));
}

fn statement_digest(
    public_key: &RistrettoPoint,
    context: &[u8],
    inputs: &PublicInputs,
    ciphertexts: &[Ciphertext],
    commitments: &[ResidualCommitments],
) -> [u8; 64] {
    let mut hasher = Sha512::new();
    append_len_prefixed(&mut hasher, b"domain", TRANSCRIPT_DOMAIN);
    append_len_prefixed(&mut hasher, b"context", context);
    append_point(&mut hasher, b"pk", public_key);
    append_u64(&mut hasher, b"dimension-count", inputs.truths.len() as u64);
    append_u64(&mut hasher, b"delta-bin", inputs.delta_bin);
    append_u64(&mut hasher, b"bin-label", inputs.bin_label);
    append_u64(&mut hasher, b"residual-max", inputs.residual_max);
    append_u64(&mut hasher, b"interval-lower", inputs.interval_lower);
    append_u64(&mut hasher, b"interval-upper", inputs.interval_upper);
    for ((truth, eta), (lower, upper)) in inputs
        .truths
        .iter()
        .zip(inputs.etas.iter())
        .zip(inputs.domains.iter())
    {
        append_i64(&mut hasher, b"truth", *truth);
        append_u64(&mut hasher, b"eta", *eta);
        append_i64(&mut hasher, b"domain-lower", *lower);
        append_i64(&mut hasher, b"domain-upper", *upper);
    }
    for ciphertext in ciphertexts {
        append_point(&mut hasher, b"cipher-u", &ciphertext.u);
        append_point(&mut hasher, b"cipher-v", &ciphertext.v);
    }
    for commitment in commitments {
        append_point(&mut hasher, b"residual-plus", &commitment.plus);
        append_point(&mut hasher, b"residual-minus", &commitment.minus);
    }
    hasher.finalize().into()
}

fn composed_challenge(
    digest: &[u8; 64],
    relation_announcements: &[(RistrettoPoint, RistrettoPoint, RistrettoPoint)],
    or_announcements: &[(RistrettoPoint, RistrettoPoint)],
) -> Scalar {
    let mut hasher = Sha512::new();
    append_len_prefixed(&mut hasher, b"statement-digest", digest);
    for (first, second, third) in relation_announcements {
        append_point(&mut hasher, b"relation-r1", first);
        append_point(&mut hasher, b"relation-r2", second);
        append_point(&mut hasher, b"relation-r3", third);
    }
    for (plus, minus) in or_announcements {
        append_point(&mut hasher, b"or-plus", plus);
        append_point(&mut hasher, b"or-minus", minus);
    }
    let output: [u8; 64] = hasher.finalize().into();
    Scalar::from_bytes_mod_order_wide(&output)
}

fn transcript_with_statement(label: &'static [u8], digest: &[u8; 64]) -> Transcript {
    let mut transcript = Transcript::new(label);
    transcript.append_message(b"statement-digest", digest);
    transcript
}

fn checked_dimension_count(
    truths: &[i64],
    etas: &[u64],
    domains_flat: &[i64],
) -> BackendResult<usize> {
    if truths.is_empty() {
        return Err(BackendError::Input("vectors must be nonempty".into()));
    }
    if truths.len() != etas.len() {
        return Err(BackendError::Input(
            "truths and etas must have equal length".into(),
        ));
    }
    let expected_domain_values = truths
        .len()
        .checked_mul(2)
        .ok_or_else(|| BackendError::Input("domain dimension count overflow".into()))?;
    if domains_flat.len() != expected_domain_values {
        return Err(BackendError::Input(
            "domains must contain two endpoints per coordinate".into(),
        ));
    }
    Ok(truths.len())
}

fn parse_domains(domains_flat: &[i64], count: usize) -> BackendResult<Vec<(i64, i64)>> {
    let mut domains = Vec::with_capacity(count);
    for index in 0..count {
        let lower = domains_flat[index * 2];
        let upper = domains_flat[index * 2 + 1];
        if lower >= upper {
            return Err(BackendError::Input(format!(
                "measurement domain at coordinate {index} must be strictly increasing"
            )));
        }
        domains.push((lower, upper));
    }
    Ok(domains)
}

fn derive_public_inputs(
    truths: Vec<i64>,
    etas: Vec<u64>,
    domains_flat: Vec<i64>,
    delta_bin: u64,
    bin_label: u64,
) -> BackendResult<PublicInputs> {
    let count = checked_dimension_count(&truths, &etas, &domains_flat)?;
    if delta_bin == 0 {
        return Err(BackendError::Input("delta_bin must be positive".into()));
    }
    let domains = parse_domains(&domains_flat, count)?;
    let mut residual_bounds = Vec::with_capacity(count);
    let mut round_residual_bound = 0u64;
    let mut residual_max = 0u64;
    for (index, ((truth, eta), (lower, upper))) in truths
        .iter()
        .zip(etas.iter())
        .zip(domains.iter())
        .enumerate()
    {
        if *eta == 0 {
            return Err(BackendError::Input(format!(
                "eta at coordinate {index} must be positive"
            )));
        }
        let lower_diff = (*lower as i128) - (*truth as i128);
        let upper_diff = (*upper as i128) - (*truth as i128);
        let eta = *eta as u128;
        let bound_lower = lower_diff.unsigned_abs().checked_mul(eta).ok_or_else(|| {
            BackendError::Input(format!("residual bound overflow at coordinate {index}"))
        })?;
        let bound_upper = upper_diff.unsigned_abs().checked_mul(eta).ok_or_else(|| {
            BackendError::Input(format!("residual bound overflow at coordinate {index}"))
        })?;
        let bound = bound_lower.max(bound_upper);
        let bound = u64::try_from(bound).map_err(|_| {
            BackendError::Input(format!("residual bound exceeds u64 at coordinate {index}"))
        })?;
        residual_bounds.push(bound);
        round_residual_bound = round_residual_bound
            .checked_add(bound)
            .ok_or_else(|| BackendError::Input("round residual bound overflow".into()))?;

        let domain_width = ((*upper as i128) - (*lower as i128)) as u128;
        let global_term = domain_width.checked_mul(eta).ok_or_else(|| {
            BackendError::Input(format!(
                "global residual bound overflow at coordinate {index}"
            ))
        })?;
        let global_term = u64::try_from(global_term).map_err(|_| {
            BackendError::Input(format!(
                "global residual bound exceeds u64 at coordinate {index}"
            ))
        })?;
        residual_max = residual_max
            .checked_add(global_term)
            .ok_or_else(|| BackendError::Input("global residual maximum overflow".into()))?;
    }
    let maximum_label = residual_max / delta_bin;
    if bin_label > maximum_label {
        return Err(BackendError::Input(
            "bin label is outside residual domain".into(),
        ));
    }
    let interval_lower = bin_label
        .checked_mul(delta_bin)
        .ok_or_else(|| BackendError::Input("bin lower endpoint overflow".into()))?;
    let interval_upper =
        ((bin_label as u128 + 1) * delta_bin as u128 - 1).min(residual_max as u128);
    let interval_upper = u64::try_from(interval_upper)
        .map_err(|_| BackendError::Input("bin upper endpoint overflow".into()))?;
    Ok(PublicInputs {
        truths,
        etas,
        domains,
        residual_bounds,
        round_residual_bound,
        residual_max,
        bin_label,
        interval_lower,
        interval_upper,
        delta_bin,
    })
}

fn derive_bin(err: u64, delta_bin: u64, residual_max: u64) -> BackendResult<u64> {
    if delta_bin == 0 || err > residual_max {
        return Err(BackendError::Input("invalid residual or bin width".into()));
    }
    Ok(err / delta_bin)
}

fn next_power_of_two(value: usize) -> BackendResult<usize> {
    if value == 0 {
        return Err(BackendError::Input(
            "aggregation count must be positive".into(),
        ));
    }
    value
        .checked_next_power_of_two()
        .ok_or_else(|| BackendError::Input("aggregation count is too large".into()))
}

fn range_bits(maximum: u64) -> BackendResult<usize> {
    let required = if maximum == 0 {
        1
    } else {
        (u64::BITS - maximum.leading_zeros()) as usize
    };
    for candidate in [8usize, 16, 32, 64] {
        if required <= candidate {
            return Ok(candidate);
        }
    }
    Err(BackendError::Input("range value exceeds 64 bits".into()))
}

fn validate_range_value(value: u128, label: &str) -> BackendResult<u64> {
    let value =
        u64::try_from(value).map_err(|_| BackendError::Input(format!("{label} exceeds u64")))?;
    Ok(value)
}

fn append_point_bytes(output: &mut Vec<u8>, point: &RistrettoPoint) {
    output.extend_from_slice(&encode_point(point));
}

fn append_scalar_bytes(output: &mut Vec<u8>, scalar: &Scalar) {
    output.extend_from_slice(&encode_scalar(scalar));
}

fn derive_domain_commitments(
    message: &RistrettoPoint,
    ciphertexts: &[Ciphertext],
    inputs: &PublicInputs,
) -> Vec<RistrettoPoint> {
    let mut commitments = Vec::with_capacity(ciphertexts.len() * 2);
    for (index, ciphertext) in ciphertexts.iter().enumerate() {
        let (lower, upper) = inputs.domains[index];
        commitments.push(ciphertext.v - scalar_from_i64(lower) * message);
        commitments.push(scalar_from_i64(upper) * message - ciphertext.v);
    }
    commitments
}

fn derive_residual_commitments(
    message: &RistrettoPoint,
    commitments: &[ResidualCommitments],
    inputs: &PublicInputs,
) -> Vec<RistrettoPoint> {
    let mut result = Vec::with_capacity(commitments.len() * 4 + 2);
    for (index, commitment) in commitments.iter().enumerate() {
        let bound = Scalar::from(inputs.residual_bounds[index]);
        result.push(commitment.plus);
        result.push(bound * message - commitment.plus);
        result.push(commitment.minus);
        result.push(bound * message - commitment.minus);
    }
    result
}

fn padded_points(mut points: Vec<RistrettoPoint>, count: usize) -> Vec<RistrettoPoint> {
    points.resize(count, RistrettoPoint::identity());
    points
}

fn padded_values(mut values: Vec<u64>, count: usize) -> Vec<u64> {
    values.resize(count, 0);
    values
}

fn padded_masks(mut masks: Vec<Scalar>, count: usize) -> Vec<Scalar> {
    masks.resize(count, Scalar::from(0u64));
    masks
}

fn verify_commitment_vector(
    expected: &[RistrettoPoint],
    actual: &[CompressedRistretto],
) -> BackendResult<()> {
    if expected.len() != actual.len() {
        return Err(BackendError::Bulletproof(
            "Bulletproof commitment count mismatch".into(),
        ));
    }
    for (expected, actual) in expected.iter().zip(actual.iter()) {
        if expected.compress() != *actual {
            return Err(BackendError::Bulletproof(
                "Bulletproof returned an unexpected commitment".into(),
            ));
        }
    }
    Ok(())
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, length: usize) -> BackendResult<&'a [u8]> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or_else(|| BackendError::Encoding("length overflow".into()))?;
        if end > self.bytes.len() {
            return Err(BackendError::Encoding("truncated byte string".into()));
        }
        let value = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(value)
    }

    fn fixed<const N: usize>(&mut self) -> BackendResult<[u8; N]> {
        self.take(N)?
            .try_into()
            .map_err(|_| BackendError::Encoding(format!("expected {N} bytes")))
    }

    fn u32(&mut self) -> BackendResult<u32> {
        Ok(u32::from_le_bytes(self.fixed()?))
    }

    fn u64(&mut self) -> BackendResult<u64> {
        Ok(u64::from_le_bytes(self.fixed()?))
    }

    fn i64(&mut self) -> BackendResult<i64> {
        Ok(i64::from_le_bytes(self.fixed()?))
    }

    fn point(&mut self) -> BackendResult<RistrettoPoint> {
        decode_point(self.take(POINT_BYTES)?)
    }

    fn scalar(&mut self) -> BackendResult<Scalar> {
        decode_scalar(self.take(SCALAR_BYTES)?)
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }

    fn at_end(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

fn encode_state(state: &BatchState) -> Vec<u8> {
    let mut output = Vec::new();
    output.extend_from_slice(STATE_MAGIC);
    output.extend_from_slice(&(state.openings.len() as u32).to_le_bytes());
    for opening in &state.openings {
        output.extend_from_slice(&opening.value.to_le_bytes());
        append_scalar_bytes(&mut output, &opening.mask);
    }
    for ciphertext in &state.ciphertexts {
        append_point_bytes(&mut output, &ciphertext.u);
        append_point_bytes(&mut output, &ciphertext.v);
    }
    output
}

fn parse_state(bytes: &[u8]) -> BackendResult<BatchState> {
    let mut reader = Reader::new(bytes);
    if reader.take(STATE_MAGIC.len())? != STATE_MAGIC {
        return Err(BackendError::Encoding(
            "invalid encryption state magic".into(),
        ));
    }
    let count = reader.u32()? as usize;
    if count == 0 {
        return Err(BackendError::Encoding("encryption state is empty".into()));
    }
    let opening_bytes = count
        .checked_mul(8 + SCALAR_BYTES)
        .ok_or_else(|| BackendError::Encoding("opening length overflow".into()))?;
    let ciphertext_bytes = count
        .checked_mul(2 * POINT_BYTES)
        .ok_or_else(|| BackendError::Encoding("ciphertext length overflow".into()))?;
    let required_bytes = opening_bytes
        .checked_add(ciphertext_bytes)
        .ok_or_else(|| BackendError::Encoding("encryption state length overflow".into()))?;
    if reader.remaining() < required_bytes {
        return Err(BackendError::Encoding("truncated encryption state".into()));
    }
    let mut openings = Vec::with_capacity(count);
    for _ in 0..count {
        openings.push(Opening {
            value: reader.i64()?,
            mask: reader.scalar()?,
        });
    }
    let mut ciphertexts = Vec::with_capacity(count);
    for _ in 0..count {
        ciphertexts.push(Ciphertext {
            u: reader.point()?,
            v: reader.point()?,
        });
    }
    if !reader.at_end() {
        return Err(BackendError::Encoding(
            "encryption state contains trailing bytes".into(),
        ));
    }
    Ok(BatchState {
        openings,
        ciphertexts,
    })
}

fn parse_public_key(bytes: &[u8]) -> BackendResult<RistrettoPoint> {
    let point = decode_point(bytes)?;
    if point == RistrettoPoint::identity() {
        return Err(BackendError::Input("public key cannot be identity".into()));
    }
    Ok(point)
}

fn encode_ciphertext_vector(ciphertexts: &[Ciphertext]) -> Vec<u8> {
    let mut output = Vec::with_capacity(ciphertexts.len() * 2 * POINT_BYTES);
    for ciphertext in ciphertexts {
        append_point_bytes(&mut output, &ciphertext.u);
        append_point_bytes(&mut output, &ciphertext.v);
    }
    output
}

fn encode_secret_scalar(scalar: &Scalar) -> Vec<u8> {
    encode_scalar(scalar).to_vec()
}

#[pyfunction]
fn generate_demo_keypair(seed: Option<u64>) -> PyResult<(Vec<u8>, Vec<u8>)> {
    let mut rng = derive_rng(seed)?;
    let mut secret = Scalar::random(&mut rng);
    while secret == Scalar::from(0u64) {
        secret = Scalar::random(&mut rng);
    }
    let public = secret * RISTRETTO_BASEPOINT_POINT;
    Ok((
        encode_point(&public).to_vec(),
        encode_secret_scalar(&secret),
    ))
}

#[pyfunction]
fn encrypt_measurements(
    measurements: Vec<i64>,
    public_key: Vec<u8>,
    seed: Option<u64>,
) -> PyResult<(Vec<u8>, Vec<u8>)> {
    if measurements.is_empty() {
        return Err(BackendError::Input("measurements must be nonempty".into()).into());
    }
    if measurements.len() > u32::MAX as usize {
        return Err(BackendError::Input("measurement count exceeds u32".into()).into());
    }
    let public_key = parse_public_key(&public_key)?;
    let message = message_base();
    let mut rng = derive_rng(seed)?;
    let mut openings = Vec::with_capacity(measurements.len());
    let mut ciphertexts = Vec::with_capacity(measurements.len());
    for value in measurements {
        let mask = Scalar::random(&mut rng);
        let u = mask * RISTRETTO_BASEPOINT_POINT;
        let v = scalar_from_i64(value) * message + mask * public_key;
        openings.push(Opening { value, mask });
        ciphertexts.push(Ciphertext { u, v });
    }
    let state = BatchState {
        openings,
        ciphertexts: ciphertexts.clone(),
    };
    Ok((encode_state(&state), encode_ciphertext_vector(&ciphertexts)))
}

fn pedersen_gens(message: RistrettoPoint, blind: RistrettoPoint) -> PedersenGens {
    PedersenGens {
        B: message,
        B_blinding: blind,
    }
}

#[allow(clippy::too_many_arguments)]
fn prove_range_family(
    values: Vec<u64>,
    masks: Vec<Scalar>,
    expected_commitments: Vec<RistrettoPoint>,
    bits: usize,
    aggregate_count: usize,
    message_base: RistrettoPoint,
    blind_base: RistrettoPoint,
    statement_digest: &[u8; 64],
    transcript_label: &'static [u8],
    rng: &mut ChaCha20Rng,
) -> BackendResult<Vec<u8>> {
    if values.len() != masks.len() || values.len() != expected_commitments.len() {
        return Err(BackendError::Input(
            "range values, masks and commitments must have equal length".into(),
        ));
    }
    let values = padded_values(values, aggregate_count);
    let masks = padded_masks(masks, aggregate_count);
    let expected_commitments = padded_points(expected_commitments, aggregate_count);
    let pc_gens = pedersen_gens(message_base, blind_base);
    let bp_gens = BulletproofGens::new(bits, aggregate_count);
    let mut transcript = transcript_with_statement(transcript_label, statement_digest);
    let (proof, commitments) = RangeProof::prove_multiple_with_rng(
        &bp_gens,
        &pc_gens,
        &mut transcript,
        &values,
        &masks,
        bits,
        rng,
    )
    .map_err(|error| BackendError::Bulletproof(error.to_string()))?;
    verify_commitment_vector(&expected_commitments, &commitments)?;
    Ok(proof.to_bytes())
}

#[allow(clippy::too_many_arguments)]
fn verify_range_family(
    proof_bytes: &[u8],
    expected_commitments: Vec<RistrettoPoint>,
    bits: usize,
    aggregate_count: usize,
    message_base: RistrettoPoint,
    blind_base: RistrettoPoint,
    statement_digest: &[u8; 64],
    transcript_label: &'static [u8],
) -> BackendResult<()> {
    let expected_commitments = padded_points(expected_commitments, aggregate_count);
    let commitments: Vec<CompressedRistretto> = expected_commitments
        .iter()
        .map(|point| point.compress())
        .collect();
    let proof = RangeProof::from_bytes(proof_bytes)
        .map_err(|error| BackendError::Encoding(format!("invalid range proof: {error}")))?;
    let pc_gens = pedersen_gens(message_base, blind_base);
    let bp_gens = BulletproofGens::new(bits, aggregate_count);
    let mut transcript = transcript_with_statement(transcript_label, statement_digest);
    proof
        .verify_multiple(&bp_gens, &pc_gens, &mut transcript, &commitments, bits)
        .map_err(|error| BackendError::Verification(format!("range proof failed: {error}")))
}

fn residual_witnesses(
    state: &BatchState,
    inputs: &PublicInputs,
) -> BackendResult<(Vec<(u64, u64)>, u64)> {
    if state.openings.len() != inputs.truths.len() || state.ciphertexts.len() != inputs.truths.len()
    {
        return Err(BackendError::Input(
            "encryption state and public vectors have different dimensions".into(),
        ));
    }
    let mut witnesses = Vec::with_capacity(state.openings.len());
    let mut residual = 0u64;
    for (index, ((opening, truth), eta)) in state
        .openings
        .iter()
        .zip(inputs.truths.iter())
        .zip(inputs.etas.iter())
        .enumerate()
    {
        let (lower, upper) = inputs.domains[index];
        if opening.value < lower || opening.value > upper {
            return Err(BackendError::Input(format!(
                "measurement at coordinate {index} is outside the public domain"
            )));
        }
        let difference = (opening.value as i128)
            .checked_sub(*truth as i128)
            .ok_or_else(|| BackendError::Input("measurement difference overflow".into()))?;
        let deviation = difference.checked_mul(*eta as i128).ok_or_else(|| {
            BackendError::Input(format!("deviation overflow at coordinate {index}"))
        })?;
        let (plus, minus) = if deviation >= 0 {
            (deviation as u128, 0u128)
        } else {
            (0u128, deviation.unsigned_abs())
        };
        let plus = validate_range_value(plus, "positive residual")?;
        let minus = validate_range_value(minus, "negative residual")?;
        if plus > inputs.residual_bounds[index] || minus > inputs.residual_bounds[index] {
            return Err(BackendError::Input(format!(
                "residual exceeds public bound at coordinate {index}"
            )));
        }
        residual = residual
            .checked_add(plus)
            .and_then(|value| value.checked_add(minus))
            .ok_or_else(|| BackendError::Input("residual overflow".into()))?;
        witnesses.push((plus, minus));
    }
    if residual > inputs.round_residual_bound || residual > inputs.residual_max {
        return Err(BackendError::Input("residual exceeds public bound".into()));
    }
    Ok((witnesses, residual))
}

fn relation_residual_statement(
    commitment: &ResidualCommitments,
    truth: i64,
    eta: u64,
    message: &RistrettoPoint,
) -> RistrettoPoint {
    commitment.plus - commitment.minus + Scalar::from(eta) * scalar_from_i64(truth) * message
}

fn validate_ciphertext_openings(
    state: &BatchState,
    public_key: &RistrettoPoint,
    message: &RistrettoPoint,
) -> BackendResult<()> {
    if state.openings.len() != state.ciphertexts.len() {
        return Err(BackendError::Input(
            "ciphertext and opening counts differ".into(),
        ));
    }
    for (index, (opening, ciphertext)) in state
        .openings
        .iter()
        .zip(state.ciphertexts.iter())
        .enumerate()
    {
        let expected_u = opening.mask * RISTRETTO_BASEPOINT_POINT;
        let expected_v = scalar_from_i64(opening.value) * message + opening.mask * public_key;
        if ciphertext.u != expected_u || ciphertext.v != expected_v {
            return Err(BackendError::Input(format!(
                "ciphertext opening mismatch at coordinate {index}"
            )));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn prove_composed(
    public_key: &RistrettoPoint,
    message: &RistrettoPoint,
    state: &BatchState,
    inputs: &PublicInputs,
    commitments: &[ResidualCommitments],
    witness_masks: &[(Scalar, Scalar)],
    statement_digest: &[u8; 64],
    rng: &mut ChaCha20Rng,
) -> BackendResult<ComposedProof> {
    if commitments.len() != state.openings.len() || witness_masks.len() != state.openings.len() {
        return Err(BackendError::Input(
            "residual commitment dimension mismatch".into(),
        ));
    }
    let mut relation_announcements = Vec::with_capacity(state.ciphertexts.len());
    let mut or_announcements = Vec::with_capacity(state.ciphertexts.len());
    let mut relation_nonces = Vec::with_capacity(state.ciphertexts.len());
    let mut or_pending = Vec::with_capacity(state.ciphertexts.len());

    for (index, (opening, commitment)) in state.openings.iter().zip(commitments.iter()).enumerate()
    {
        let rx = Scalar::random(rng);
        let rmask = Scalar::random(rng);
        let rdelta = Scalar::random(rng);
        let relation_r1 = rmask * RISTRETTO_BASEPOINT_POINT;
        let relation_r2 = rx * message + rmask * public_key;
        let relation_r3 = Scalar::from(inputs.etas[index]) * rx * message + rdelta * blind_base();
        relation_announcements.push((relation_r1, relation_r2, relation_r3));
        relation_nonces.push((rx, rmask, rdelta));

        let (omega_plus, omega_minus) = witness_masks[index];
        let blind = blind_base();
        let plus_statement = commitment.plus;
        let minus_statement = commitment.minus;
        let plus_value = {
            let difference = (opening.value as i128) - (inputs.truths[index] as i128);
            let deviation = difference
                .checked_mul(inputs.etas[index] as i128)
                .ok_or_else(|| BackendError::Input("deviation overflow".into()))?;
            if deviation >= 0 {
                deviation as u64
            } else {
                0u64
            }
        };
        let true_plus = plus_value == 0;
        let true_nonce = Scalar::random(rng);
        let false_challenge = Scalar::random(rng);
        let false_response = Scalar::random(rng);
        let (plus_announcement, minus_announcement) = if true_plus {
            (
                true_nonce * blind,
                false_response * blind - false_challenge * minus_statement,
            )
        } else {
            (
                false_response * blind - false_challenge * plus_statement,
                true_nonce * blind,
            )
        };
        or_announcements.push((plus_announcement, minus_announcement));
        or_pending.push((
            true_plus,
            true_nonce,
            false_challenge,
            false_response,
            omega_plus,
            omega_minus,
        ));
    }

    let challenge =
        composed_challenge(statement_digest, &relation_announcements, &or_announcements);
    let mut relations = Vec::with_capacity(relation_nonces.len());
    let mut zero_sides = Vec::with_capacity(or_pending.len());
    for ((rx, rmask, rdelta), opening) in relation_nonces.iter().zip(state.openings.iter()) {
        let x = scalar_from_i64(opening.value);
        let delta = witness_masks[relations.len()].0 - witness_masks[relations.len()].1;
        relations.push(RelationResponse {
            x: *rx + challenge * x,
            mask: *rmask + challenge * opening.mask,
            delta_mask: *rdelta + challenge * delta,
        });
    }
    for (true_plus, true_nonce, false_challenge, false_response, omega_plus, omega_minus) in
        or_pending
    {
        if true_plus {
            let challenge_minus = false_challenge;
            let challenge_plus = challenge - challenge_minus;
            zero_sides.push(OrResponse {
                challenge_plus,
                response_plus: true_nonce + challenge_plus * omega_plus,
                response_minus: false_response,
            });
        } else {
            let challenge_plus = false_challenge;
            let challenge_minus = challenge - challenge_plus;
            zero_sides.push(OrResponse {
                challenge_plus,
                response_plus: false_response,
                response_minus: true_nonce + challenge_minus * omega_minus,
            });
        }
    }
    Ok(ComposedProof {
        challenge,
        relations,
        zero_sides,
    })
}

fn verify_composed(
    public_key: &RistrettoPoint,
    message: &RistrettoPoint,
    ciphertexts: &[Ciphertext],
    inputs: &PublicInputs,
    commitments: &[ResidualCommitments],
    proof: &ComposedProof,
    statement_digest: &[u8; 64],
) -> BackendResult<()> {
    if proof.relations.len() != ciphertexts.len()
        || proof.zero_sides.len() != ciphertexts.len()
        || commitments.len() != ciphertexts.len()
    {
        return Err(BackendError::Verification(
            "composed proof dimension mismatch".into(),
        ));
    }
    let mut relation_announcements = Vec::with_capacity(ciphertexts.len());
    let mut or_announcements = Vec::with_capacity(ciphertexts.len());
    for index in 0..ciphertexts.len() {
        let response = &proof.relations[index];
        let relation_r1 =
            response.mask * RISTRETTO_BASEPOINT_POINT - proof.challenge * ciphertexts[index].u;
        let relation_r2 = response.x * message + response.mask * public_key
            - proof.challenge * ciphertexts[index].v;
        let residual_statement = relation_residual_statement(
            &commitments[index],
            inputs.truths[index],
            inputs.etas[index],
            message,
        );
        let relation_r3 = Scalar::from(inputs.etas[index]) * response.x * message
            + response.delta_mask * blind_base()
            - proof.challenge * residual_statement;
        relation_announcements.push((relation_r1, relation_r2, relation_r3));

        let or = &proof.zero_sides[index];
        let challenge_minus = proof.challenge - or.challenge_plus;
        let blind = blind_base();
        let plus_announcement =
            or.response_plus * blind - or.challenge_plus * commitments[index].plus;
        let minus_announcement =
            or.response_minus * blind - challenge_minus * commitments[index].minus;
        or_announcements.push((plus_announcement, minus_announcement));
    }
    let expected = composed_challenge(statement_digest, &relation_announcements, &or_announcements);
    if expected != proof.challenge {
        return Err(BackendError::Verification(
            "composed Fiat-Shamir challenge mismatch".into(),
        ));
    }
    Ok(())
}

fn encode_composed(proof: &ComposedProof) -> Vec<u8> {
    let mut output = Vec::with_capacity(SCALAR_BYTES * (1 + proof.relations.len() * 6));
    append_scalar_bytes(&mut output, &proof.challenge);
    for relation in &proof.relations {
        append_scalar_bytes(&mut output, &relation.x);
        append_scalar_bytes(&mut output, &relation.mask);
        append_scalar_bytes(&mut output, &relation.delta_mask);
    }
    for zero_side in &proof.zero_sides {
        append_scalar_bytes(&mut output, &zero_side.challenge_plus);
        append_scalar_bytes(&mut output, &zero_side.response_plus);
        append_scalar_bytes(&mut output, &zero_side.response_minus);
    }
    output
}

fn parse_composed(reader: &mut Reader<'_>, count: usize) -> BackendResult<ComposedProof> {
    let challenge = reader.scalar()?;
    let mut relations = Vec::with_capacity(count);
    for _ in 0..count {
        relations.push(RelationResponse {
            x: reader.scalar()?,
            mask: reader.scalar()?,
            delta_mask: reader.scalar()?,
        });
    }
    let mut zero_sides = Vec::with_capacity(count);
    for _ in 0..count {
        zero_sides.push(OrResponse {
            challenge_plus: reader.scalar()?,
            response_plus: reader.scalar()?,
            response_minus: reader.scalar()?,
        });
    }
    Ok(ComposedProof {
        challenge,
        relations,
        zero_sides,
    })
}

fn encode_report(
    dimension_count: usize,
    bin_label: u64,
    ciphertexts: &[Ciphertext],
    commitments: &[ResidualCommitments],
    composed: &ComposedProof,
    domain_range_bytes: &[u8],
    residual_range_bytes: &[u8],
) -> BackendResult<Vec<u8>> {
    if dimension_count == 0
        || ciphertexts.len() != dimension_count
        || commitments.len() != dimension_count
    {
        return Err(BackendError::Input("report dimension mismatch".into()));
    }
    let domain_length = u32::try_from(domain_range_bytes.len())
        .map_err(|_| BackendError::Input("domain proof is too large".into()))?;
    let residual_length = u32::try_from(residual_range_bytes.len())
        .map_err(|_| BackendError::Input("residual proof is too large".into()))?;
    let composed_bytes = encode_composed(composed);
    let point_bytes = dimension_count
        .checked_mul(POINT_BYTES)
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| BackendError::Input("report point length overflow".into()))?;
    let capacity = 28usize
        .checked_add(point_bytes)
        .and_then(|value| value.checked_add(composed_bytes.len()))
        .and_then(|value| value.checked_add(domain_range_bytes.len()))
        .and_then(|value| value.checked_add(residual_range_bytes.len()))
        .ok_or_else(|| BackendError::Input("report length overflow".into()))?;
    let mut output = Vec::with_capacity(capacity);
    let dimension_u32 = u32::try_from(dimension_count)
        .map_err(|_| BackendError::Input("dimension count exceeds u32".into()))?;
    output.extend_from_slice(PACKET_MAGIC);
    output.extend_from_slice(&dimension_u32.to_le_bytes());
    output.extend_from_slice(&bin_label.to_le_bytes());
    output.extend_from_slice(&domain_length.to_le_bytes());
    output.extend_from_slice(&residual_length.to_le_bytes());
    for ciphertext in ciphertexts {
        append_point_bytes(&mut output, &ciphertext.u);
        append_point_bytes(&mut output, &ciphertext.v);
    }
    for commitment in commitments {
        append_point_bytes(&mut output, &commitment.plus);
        append_point_bytes(&mut output, &commitment.minus);
    }
    output.extend_from_slice(&composed_bytes);
    output.extend_from_slice(domain_range_bytes);
    output.extend_from_slice(residual_range_bytes);
    Ok(output)
}

fn parse_report(bytes: &[u8]) -> BackendResult<ParsedReport> {
    let mut reader = Reader::new(bytes);
    if reader.take(PACKET_MAGIC.len())? != PACKET_MAGIC {
        return Err(BackendError::Encoding(
            "invalid residual-bin packet magic".into(),
        ));
    }
    let dimension_count = reader.u32()? as usize;
    if dimension_count == 0 {
        return Err(BackendError::Encoding(
            "packet dimension count is zero".into(),
        ));
    }
    let bin_label = reader.u64()?;
    let domain_length = reader.u32()? as usize;
    let residual_length = reader.u32()? as usize;
    let header_bytes = reader.offset;
    let ciphertext_offset = reader.offset;
    let ciphertext_bytes = dimension_count
        .checked_mul(2 * POINT_BYTES)
        .ok_or_else(|| BackendError::Encoding("ciphertext length overflow".into()))?;
    if reader.remaining() < ciphertext_bytes {
        return Err(BackendError::Encoding("truncated ciphertext vector".into()));
    }
    let mut ciphertexts = Vec::with_capacity(dimension_count);
    for _ in 0..dimension_count {
        ciphertexts.push(Ciphertext {
            u: reader.point()?,
            v: reader.point()?,
        });
    }
    let commitment_offset = reader.offset;
    let commitment_bytes = dimension_count
        .checked_mul(2 * POINT_BYTES)
        .ok_or_else(|| BackendError::Encoding("commitment length overflow".into()))?;
    if reader.remaining() < commitment_bytes {
        return Err(BackendError::Encoding(
            "truncated residual commitments".into(),
        ));
    }
    let mut commitments = Vec::with_capacity(dimension_count);
    for _ in 0..dimension_count {
        commitments.push(ResidualCommitments {
            plus: reader.point()?,
            minus: reader.point()?,
        });
    }
    let composed_offset = reader.offset;
    let composed_scalars = dimension_count
        .checked_mul(6)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| BackendError::Encoding("composed proof length overflow".into()))?;
    let composed_bytes = SCALAR_BYTES
        .checked_mul(composed_scalars)
        .ok_or_else(|| BackendError::Encoding("composed proof length overflow".into()))?;
    if reader.remaining() < composed_bytes {
        return Err(BackendError::Encoding("truncated composed proof".into()));
    }
    let composed = parse_composed(&mut reader, dimension_count)?;
    let range_bytes = domain_length
        .checked_add(residual_length)
        .ok_or_else(|| BackendError::Encoding("range proof length overflow".into()))?;
    if reader.remaining() < range_bytes {
        return Err(BackendError::Encoding("truncated range proofs".into()));
    }
    let domain_range_bytes = reader.take(domain_length)?.to_vec();
    let residual_range_bytes = reader.take(residual_length)?.to_vec();
    if !reader.at_end() {
        return Err(BackendError::Encoding(
            "packet contains trailing bytes".into(),
        ));
    }
    Ok(ParsedReport {
        dimension_count,
        bin_label,
        ciphertexts,
        commitments,
        composed,
        domain_range_bytes,
        residual_range_bytes,
        header_bytes,
        ciphertext_offset,
        commitment_offset,
        composed_offset,
    })
}

fn residual_range_values_and_masks(
    state: &BatchState,
    inputs: &PublicInputs,
    commitments: &[ResidualCommitments],
    residual: u64,
    witness_masks: &[(Scalar, Scalar)],
) -> BackendResult<RangeFamily> {
    let message = message_base();
    let mut values = Vec::with_capacity(inputs.truths.len() * 4 + 2);
    let mut masks = Vec::with_capacity(inputs.truths.len() * 4 + 2);
    if witness_masks.len() != inputs.truths.len() {
        return Err(BackendError::Input("residual mask count mismatch".into()));
    }
    let mut aggregate_commitment = RistrettoPoint::identity();
    let mut aggregate_mask = Scalar::from(0u64);
    for (index, ((opening, truth), commitment)) in state
        .openings
        .iter()
        .zip(inputs.truths.iter())
        .zip(commitments.iter())
        .enumerate()
    {
        let difference = (opening.value as i128) - (*truth as i128);
        let deviation = difference
            .checked_mul(inputs.etas[index] as i128)
            .ok_or_else(|| BackendError::Input("deviation overflow".into()))?;
        let (plus, minus) = if deviation >= 0 {
            (deviation as u64, 0u64)
        } else {
            (0u64, deviation.unsigned_abs() as u64)
        };
        let (omega_plus, omega_minus) = witness_masks[index];
        values.extend_from_slice(&[
            plus,
            inputs.residual_bounds[index]
                .checked_sub(plus)
                .ok_or_else(|| {
                    BackendError::Input("positive residual complement underflow".into())
                })?,
            minus,
            inputs.residual_bounds[index]
                .checked_sub(minus)
                .ok_or_else(|| {
                    BackendError::Input("negative residual complement underflow".into())
                })?,
        ]);
        masks.extend_from_slice(&[omega_plus, -omega_plus, omega_minus, -omega_minus]);
        aggregate_commitment += commitment.plus + commitment.minus;
        aggregate_mask += omega_plus + omega_minus;
    }
    let lower = inputs.interval_lower;
    let upper = inputs.interval_upper;
    let offset = residual
        .checked_sub(lower)
        .ok_or_else(|| BackendError::Input("residual is below claimed bucket".into()))?;
    let width = upper
        .checked_sub(lower)
        .ok_or_else(|| BackendError::Input("invalid bucket interval".into()))?;
    let offset_commitment = aggregate_commitment - Scalar::from(lower) * message;
    let complement_commitment = Scalar::from(width) * message - offset_commitment;
    values.extend_from_slice(&[
        offset,
        width
            .checked_sub(offset)
            .ok_or_else(|| BackendError::Input("bucket offset complement underflow".into()))?,
    ]);
    masks.extend_from_slice(&[aggregate_mask, -aggregate_mask]);
    let mut expected = derive_residual_commitments(&message, commitments, inputs);
    expected.push(offset_commitment);
    expected.push(complement_commitment);
    let public_maximum = inputs
        .residual_bounds
        .iter()
        .copied()
        .max()
        .unwrap_or(0)
        .max(width);
    let bits = range_bits(public_maximum)?;
    Ok((values, masks, expected, bits))
}

fn domain_range_values_and_masks(
    state: &BatchState,
    inputs: &PublicInputs,
    message: &RistrettoPoint,
) -> BackendResult<RangeFamily> {
    if state.openings.len() != inputs.truths.len() || state.ciphertexts.len() != inputs.truths.len()
    {
        return Err(BackendError::Input(
            "encryption state and public vectors have different dimensions".into(),
        ));
    }
    let mut values = Vec::with_capacity(inputs.truths.len() * 2);
    let mut masks = Vec::with_capacity(inputs.truths.len() * 2);
    let mut maximum = 0u64;
    for (index, opening) in state.openings.iter().enumerate() {
        let (lower, upper) = inputs.domains[index];
        if opening.value < lower || opening.value > upper {
            return Err(BackendError::Input(format!(
                "measurement at coordinate {index} is outside the public domain"
            )));
        }
        let lower_gap = u64::try_from((opening.value as i128) - (lower as i128))
            .map_err(|_| BackendError::Input("lower domain gap is invalid".into()))?;
        let upper_gap = u64::try_from((upper as i128) - (opening.value as i128))
            .map_err(|_| BackendError::Input("upper domain gap is invalid".into()))?;
        values.extend_from_slice(&[lower_gap, upper_gap]);
        masks.extend_from_slice(&[opening.mask, -opening.mask]);
        maximum = maximum.max(upper.saturating_sub(lower) as u64);
    }
    let expected = derive_domain_commitments(message, &state.ciphertexts, inputs);
    Ok((values, masks, expected, range_bits(maximum)?))
}

fn residual_range_expected_commitments(
    message: &RistrettoPoint,
    commitments: &[ResidualCommitments],
    inputs: &PublicInputs,
) -> BackendResult<(Vec<RistrettoPoint>, usize)> {
    let lower = inputs.interval_lower;
    let upper = inputs.interval_upper;
    let width = upper
        .checked_sub(lower)
        .ok_or_else(|| BackendError::Input("invalid bucket interval".into()))?;
    let aggregate = commitments
        .iter()
        .fold(RistrettoPoint::identity(), |acc, item| {
            acc + item.plus + item.minus
        });
    let offset_commitment = aggregate - Scalar::from(lower) * message;
    let complement_commitment = Scalar::from(width) * message - offset_commitment;
    let mut expected = derive_residual_commitments(message, commitments, inputs);
    expected.push(offset_commitment);
    expected.push(complement_commitment);
    let maximum = inputs
        .residual_bounds
        .iter()
        .copied()
        .max()
        .unwrap_or(0)
        .max(width);
    Ok((expected, range_bits(maximum)?))
}

fn make_residual_commitments(
    witnesses: &[(u64, u64)],
    rng: &mut ChaCha20Rng,
) -> (Vec<ResidualCommitments>, Vec<(Scalar, Scalar)>) {
    let message = message_base();
    let blind = blind_base();
    let mut commitments = Vec::with_capacity(witnesses.len());
    let mut masks = Vec::with_capacity(witnesses.len());
    for (plus, minus) in witnesses {
        let omega_plus = Scalar::random(rng);
        let omega_minus = Scalar::random(rng);
        commitments.push(ResidualCommitments {
            plus: Scalar::from(*plus) * message + omega_plus * blind,
            minus: Scalar::from(*minus) * message + omega_minus * blind,
        });
        masks.push((omega_plus, omega_minus));
    }
    (commitments, masks)
}

#[allow(clippy::too_many_arguments)]
fn prove_residual_bin_impl(
    state_bytes: &[u8],
    public_key_bytes: &[u8],
    truths: Vec<i64>,
    etas: Vec<u64>,
    domains_flat: Vec<i64>,
    delta_bin: u64,
    context: &[u8],
    seed: Option<u64>,
) -> BackendResult<Vec<u8>> {
    if context.is_empty() {
        return Err(BackendError::Input("context must be nonempty".into()));
    }
    let state = parse_state(state_bytes)?;
    let public_key = parse_public_key(public_key_bytes)?;
    let message = message_base();
    validate_ciphertext_openings(&state, &public_key, &message)?;
    let provisional = derive_public_inputs(
        truths.clone(),
        etas.clone(),
        domains_flat.clone(),
        delta_bin,
        0,
    )?;
    let (witnesses, residual) = residual_witnesses(&state, &provisional)?;
    let bin_label = derive_bin(residual, delta_bin, provisional.residual_max)?;
    let inputs = derive_public_inputs(truths, etas, domains_flat, delta_bin, bin_label)?;
    let blind = blind_base();
    let mut rng = derive_rng(seed)?;
    let (commitments, witness_masks) = make_residual_commitments(&witnesses, &mut rng);
    let digest = statement_digest(
        &public_key,
        context,
        &inputs,
        &state.ciphertexts,
        &commitments,
    );

    let (domain_values, domain_masks, domain_expected, domain_bits) =
        domain_range_values_and_masks(&state, &inputs, &message)?;
    let domain_count = next_power_of_two(
        domain_values
            .len()
            .checked_mul(1)
            .ok_or_else(|| BackendError::Input("domain aggregation count overflow".into()))?,
    )?;
    let domain_range_bytes = prove_range_family(
        domain_values,
        domain_masks,
        domain_expected,
        domain_bits,
        domain_count,
        message,
        public_key,
        &digest,
        b"BRPTD/domain-range/v1",
        &mut rng,
    )?;

    let (residual_values, residual_masks, residual_expected, residual_bits) =
        residual_range_values_and_masks(&state, &inputs, &commitments, residual, &witness_masks)?;
    let residual_count = next_power_of_two(residual_values.len())?;
    let residual_range_bytes = prove_range_family(
        residual_values,
        residual_masks,
        residual_expected,
        residual_bits,
        residual_count,
        message,
        blind,
        &digest,
        b"BRPTD/residual-range/v1",
        &mut rng,
    )?;
    let composed = prove_composed(
        &public_key,
        &message,
        &state,
        &inputs,
        &commitments,
        &witness_masks,
        &digest,
        &mut rng,
    )?;
    encode_report(
        state.ciphertexts.len(),
        bin_label,
        &state.ciphertexts,
        &commitments,
        &composed,
        &domain_range_bytes,
        &residual_range_bytes,
    )
}

fn verify_residual_bin_impl(
    report_bytes: &[u8],
    public_key_bytes: &[u8],
    truths: Vec<i64>,
    etas: Vec<u64>,
    domains_flat: Vec<i64>,
    delta_bin: u64,
    context: &[u8],
) -> BackendResult<(u64, u64, u64, u64)> {
    if context.is_empty() {
        return Err(BackendError::Input("context must be nonempty".into()));
    }
    let report = parse_report(report_bytes)?;
    let public_key = parse_public_key(public_key_bytes)?;
    let inputs = derive_public_inputs(truths, etas, domains_flat, delta_bin, report.bin_label)?;
    if report.dimension_count != inputs.truths.len() {
        return Err(BackendError::Verification(
            "report dimension does not match public inputs".into(),
        ));
    }
    let message = message_base();
    let digest = statement_digest(
        &public_key,
        context,
        &inputs,
        &report.ciphertexts,
        &report.commitments,
    );
    let domain_expected = derive_domain_commitments(&message, &report.ciphertexts, &inputs);
    let domain_maximum = inputs
        .domains
        .iter()
        .map(|(lower, upper)| {
            u64::try_from((*upper as i128) - (*lower as i128)).unwrap_or(u64::MAX)
        })
        .max()
        .unwrap_or(0);
    let domain_bits = range_bits(domain_maximum)?;
    let domain_count = next_power_of_two(
        inputs
            .truths
            .len()
            .checked_mul(2)
            .ok_or_else(|| BackendError::Input("domain aggregation count overflow".into()))?,
    )?;
    verify_range_family(
        &report.domain_range_bytes,
        domain_expected,
        domain_bits,
        domain_count,
        message,
        public_key,
        &digest,
        b"BRPTD/domain-range/v1",
    )?;

    let (residual_expected, residual_bits) =
        residual_range_expected_commitments(&message, &report.commitments, &inputs)?;
    let residual_count = next_power_of_two(
        inputs
            .truths
            .len()
            .checked_mul(4)
            .and_then(|value| value.checked_add(2))
            .ok_or_else(|| BackendError::Input("residual aggregation count overflow".into()))?,
    )?;
    verify_range_family(
        &report.residual_range_bytes,
        residual_expected,
        residual_bits,
        residual_count,
        message,
        blind_base(),
        &digest,
        b"BRPTD/residual-range/v1",
    )?;
    verify_composed(
        &public_key,
        &message,
        &report.ciphertexts,
        &inputs,
        &report.commitments,
        &report.composed,
        &digest,
    )?;
    Ok((
        inputs.bin_label,
        inputs.interval_lower,
        inputs.interval_upper,
        inputs.interval_upper,
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn prove_residual_bin(
    state: Vec<u8>,
    public_key: Vec<u8>,
    truths: Vec<i64>,
    etas: Vec<u64>,
    domains_flat: Vec<i64>,
    delta_bin: u64,
    context: Vec<u8>,
    seed: Option<u64>,
) -> PyResult<Vec<u8>> {
    Ok(prove_residual_bin_impl(
        &state,
        &public_key,
        truths,
        etas,
        domains_flat,
        delta_bin,
        &context,
        seed,
    )?)
}

#[pyfunction]
fn verify_residual_bin(
    report: Vec<u8>,
    public_key: Vec<u8>,
    truths: Vec<i64>,
    etas: Vec<u64>,
    domains_flat: Vec<i64>,
    delta_bin: u64,
    context: Vec<u8>,
) -> (bool, String, u64, u64, u64, u64) {
    match verify_residual_bin_impl(
        &report,
        &public_key,
        truths,
        etas,
        domains_flat,
        delta_bin,
        &context,
    ) {
        Ok((label, lower, upper, proxy)) => (true, "ok".into(), label, lower, upper, proxy),
        Err(error) => (false, error.to_string(), 0, 0, 0, 0),
    }
}

#[pyfunction]
fn inspect_residual_bin_report(report: Vec<u8>) -> PyResult<Vec<(String, u64)>> {
    let parsed = parse_report(&report)?;
    let ciphertext_bytes = parsed
        .dimension_count
        .checked_mul(2 * POINT_BYTES)
        .ok_or_else(|| BackendError::Encoding("ciphertext length overflow".into()))?;
    let commitment_bytes = parsed
        .dimension_count
        .checked_mul(2 * POINT_BYTES)
        .ok_or_else(|| BackendError::Encoding("commitment length overflow".into()))?;
    let composed_scalars = parsed
        .dimension_count
        .checked_mul(6)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| BackendError::Encoding("composed proof length overflow".into()))?;
    let composed_bytes = SCALAR_BYTES
        .checked_mul(composed_scalars)
        .ok_or_else(|| BackendError::Encoding("composed proof length overflow".into()))?;
    let mut fields = vec![
        ("header_bytes".into(), parsed.header_bytes as u64),
        ("ciphertext_offset".into(), parsed.ciphertext_offset as u64),
        ("ciphertext_bytes".into(), ciphertext_bytes as u64),
        (
            "residual_commitment_offset".into(),
            parsed.commitment_offset as u64,
        ),
        ("residual_commitment_bytes".into(), commitment_bytes as u64),
        (
            "composed_proof_offset".into(),
            parsed.composed_offset as u64,
        ),
        ("composed_proof_bytes".into(), composed_bytes as u64),
        (
            "domain_range_proof_bytes".into(),
            parsed.domain_range_bytes.len() as u64,
        ),
        (
            "residual_range_proof_bytes".into(),
            parsed.residual_range_bytes.len() as u64,
        ),
        ("packet_bytes".into(), report.len() as u64),
        ("dimension_count".into(), parsed.dimension_count as u64),
        ("bin_label".into(), parsed.bin_label),
    ];
    fields.shrink_to_fit();
    Ok(fields)
}

#[pymodule]
fn brptd_ristretto_backend(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_demo_keypair, m)?)?;
    m.add_function(wrap_pyfunction!(encrypt_measurements, m)?)?;
    m.add_function(wrap_pyfunction!(prove_residual_bin, m)?)?;
    m.add_function(wrap_pyfunction!(verify_residual_bin, m)?)?;
    m.add_function(wrap_pyfunction!(inspect_residual_bin_report, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(
        measurements: Vec<i64>,
        truths: Vec<i64>,
        etas: Vec<u64>,
        domains: Vec<i64>,
        delta_bin: u64,
        context: &[u8],
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let (public_key, _) = generate_demo_keypair(Some(37)).expect("key generation");
        let (state, _) =
            encrypt_measurements(measurements, public_key.clone(), Some(41)).expect("encryption");
        let report = prove_residual_bin_impl(
            &state,
            &public_key,
            truths,
            etas,
            domains,
            delta_bin,
            context,
            Some(43),
        )
        .expect("proof generation");
        (public_key, state, report)
    }

    #[test]
    fn roundtrip_positive_negative_and_zero_residuals() {
        let cases = [
            (vec![26, 65, 72], vec![25, 80, 60], vec![4, 1, 1], 3u64),
            (vec![24, 85, 55], vec![25, 80, 60], vec![4, 1, 1], 1u64),
            (vec![25, 80, 60], vec![25, 80, 60], vec![4, 1, 1], 0u64),
        ];
        for (measurements, truths, etas, label) in cases {
            let (public_key, _state, report) = fixture(
                measurements,
                truths.clone(),
                etas.clone(),
                vec![24, 27, 60, 85, 55, 75],
                8,
                b"rust-roundtrip",
            );
            let result = verify_residual_bin_impl(
                &report,
                &public_key,
                truths,
                etas,
                vec![24, 27, 60, 85, 55, 75],
                8,
                b"rust-roundtrip",
            )
            .expect("verification");
            assert_eq!(result.0, label);
        }
    }

    #[test]
    fn last_bucket_and_aggregation_padding_verify() {
        for measurement in [0i64, 7, 8, 39, 40] {
            let (public_key, _state, report) = fixture(
                vec![measurement],
                vec![0],
                vec![1],
                vec![0, 40],
                8,
                b"rust-boundary",
            );
            let result = verify_residual_bin_impl(
                &report,
                &public_key,
                vec![0],
                vec![1],
                vec![0, 40],
                8,
                b"rust-boundary",
            )
            .expect("boundary verification");
            assert_eq!(result.0, (measurement as u64) / 8);
            assert!(result.2 >= result.1);
        }
        let (public_key, _state, report) = fixture(
            vec![1, 2, 3, 4, 5],
            vec![0, 0, 0, 0, 0],
            vec![1, 1, 1, 1, 1],
            vec![0, 10, 0, 10, 0, 10, 0, 10, 0, 10],
            4,
            b"rust-padding",
        );
        verify_residual_bin_impl(
            &report,
            &public_key,
            vec![0, 0, 0, 0, 0],
            vec![1, 1, 1, 1, 1],
            vec![0, 10, 0, 10, 0, 10, 0, 10, 0, 10],
            4,
            b"rust-padding",
        )
        .expect("padded aggregation verification");
    }

    #[test]
    fn bucket_interval_uses_global_domain_maximum_not_round_bound() {
        let (public_key, _state, report) = fixture(
            vec![100],
            vec![50],
            vec![1],
            vec![0, 100],
            10,
            b"global-bucket-bound",
        );
        let result = verify_residual_bin_impl(
            &report,
            &public_key,
            vec![50],
            vec![1],
            vec![0, 100],
            10,
            b"global-bucket-bound",
        )
        .expect("global bucket verification");
        assert_eq!(result.0, 5);
        assert_eq!(result.1, 50);
        assert_eq!(result.2, 59);
        assert_eq!(result.3, 59);
    }

    #[test]
    fn domain_context_and_packet_tampering_are_rejected() {
        let (public_key, _state, report) = fixture(
            vec![26, 65, 72],
            vec![25, 80, 60],
            vec![4, 1, 1],
            vec![24, 27, 60, 85, 55, 75],
            8,
            b"rust-tamper",
        );
        assert!(
            verify_residual_bin_impl(
                &report,
                &public_key,
                vec![25, 80, 60],
                vec![4, 1, 1],
                vec![24, 27, 60, 85, 55, 75],
                8,
                b"wrong-context",
            )
            .is_err()
        );

        let mut tampered = report.clone();
        tampered[28] ^= 1;
        assert!(
            verify_residual_bin_impl(
                &tampered,
                &public_key,
                vec![25, 80, 60],
                vec![4, 1, 1],
                vec![24, 27, 60, 85, 55, 75],
                8,
                b"rust-tamper",
            )
            .is_err()
        );

        let (domain_key, _) = generate_demo_keypair(Some(62)).expect("domain key");
        let (domain_state, _) = encrypt_measurements(vec![101], domain_key.clone(), Some(63))
            .expect("domain encryption");
        assert!(
            prove_residual_bin_impl(
                &domain_state,
                &domain_key,
                vec![100],
                vec![1],
                vec![0, 100],
                8,
                b"domain",
                Some(64),
            )
            .is_err()
        );
    }
}
