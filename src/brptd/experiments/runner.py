"""BRPTD 自身攻击矩阵、桶敏感性及可恢复制品编排。"""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from brptd.artifacts import write_manifest, write_round_metrics, write_summary, write_trial_metrics
from brptd.data import (
    BMAQ_SPEC,
    IBRL_SPEC,
    SparsePanel,
    build_clean_truth,
    build_fold_boundaries,
    fit_training_transform,
    load_bmaq,
    load_ibrl,
    restrict_to_active_prefix,
    select_fold_blocks,
    select_training_panel,
)
from brptd.metrics import (
    RoundMetrics,
    TrialMetrics,
    average_ranks,
    crse_ratio,
    invalid_round_rate,
    malicious_weight_share,
    proof_acceptance_rate,
    standardized_crse,
    summarize_trial_metrics,
    worker_spearman,
)
from brptd.numeric import (
    ResidualBinConfig,
    compute_residual_bin,
    decode,
    encode,
    exact_residual,
)
from brptd.robustness import PPCHState
from brptd.simulation import (
    ATTACKS,
    DEFAULT_ATTACK_PARAMETERS,
    AggregationPreview,
    AttackParameters,
    BaseScenario,
    FangEvaluator,
    build_attack_scenario,
    build_base_scenario,
    derive_trial_seeds,
    optimize_fang_round,
)

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]


class ExperimentError(RuntimeError):
    """实验配置、数据可用性或证明模式不满足契约。"""


PROOF_MODES = ("contract", "sampled", "full")


@dataclass(frozen=True)
class ExperimentConfig:
    """主实验和桶扫描共用的冻结运行配置。"""

    proof_mode: str = "sampled"
    bin_count: int = 8192
    attacks: tuple[str, ...] = ATTACKS
    trial_ids: tuple[int, ...] = tuple(range(202600, 202620))
    bootstrap_resamples: int = 10_000
    bootstrap_seed: int = 20260901
    ibrl_worker_count: int = 50
    bmaq_worker_count: int = 12
    ibrl_malicious_count: int = 15
    bmaq_malicious_count: int = 4
    block_length: int = 15
    minimum_coverage: float = 0.80
    outer_starts: tuple[float, ...] = (0.40, 0.55, 0.70, 0.85)
    outer_fraction: float = 0.15
    embargo_rounds: int = 15
    blocks_per_fold: int = 5
    bucket_counts: tuple[int, ...] = tuple(2**power for power in range(3, 14))
    attack_parameters: AttackParameters = DEFAULT_ATTACK_PARAMETERS

    def __post_init__(self) -> None:
        if self.proof_mode not in PROOF_MODES:
            raise ExperimentError(f"proof_mode 必须属于 {PROOF_MODES}")
        if self.bin_count <= 0 or self.bootstrap_resamples <= 0 or self.block_length != 15:
            raise ExperimentError("bin_count、bootstrap_resamples 或 block_length 非法")
        if not 0 < self.minimum_coverage <= 1:
            raise ExperimentError("minimum_coverage 必须位于 (0, 1]")
        if tuple(self.attacks) != tuple(dict.fromkeys(self.attacks)) or not self.attacks:
            raise ExperimentError("attacks 必须非空且不可重复")
        if any(attack not in ATTACKS for attack in self.attacks):
            raise ExperimentError("attacks 包含未支持攻击")
        if len(self.trial_ids) != 20 or tuple(self.trial_ids) != tuple(range(202600, 202620)):
            raise ExperimentError("论文主试验固定使用 202600 至 202619 共 20 个 trial_id")
        if self.ibrl_worker_count != 50 or self.bmaq_worker_count != 12:
            raise ExperimentError("论文主试验固定 IBRL 50 人、BMAQ 12 站")
        if self.ibrl_malicious_count != 15 or self.bmaq_malicious_count != 4:
            raise ExperimentError("论文主攻击矩阵固定 IBRL 15、BMAQ 4 名恶意者")
        if len(self.outer_starts) != 4 or any(
            not math.isfinite(float(value)) or not 0.0 < float(value) < 1.0 for value in self.outer_starts
        ):
            raise ExperimentError("outer_starts 必须包含四个有限且位于 (0, 1) 的起点")
        normalized_starts = tuple(float(value) for value in self.outer_starts)
        if tuple(sorted(normalized_starts)) != normalized_starts or len(set(normalized_starts)) != len(
            normalized_starts
        ):
            raise ExperimentError("outer_starts 必须严格递增")
        if not math.isfinite(self.outer_fraction) or not 0.0 < self.outer_fraction <= 1.0:
            raise ExperimentError("outer_fraction 必须位于 (0, 1]")
        if self.embargo_rounds != 15 or self.blocks_per_fold != 5:
            raise ExperimentError("论文时间折固定 15 轮隔离区和每折 5 个块")
        if self.bucket_counts != tuple(2**power for power in range(3, 14)):
            raise ExperimentError("桶扫描固定使用 K=2^3 至 2^13")
        if not isinstance(self.attack_parameters, AttackParameters):
            raise ExperimentError("attack_parameters 必须是 AttackParameters")

    def as_manifest_fragment(self) -> dict[str, Any]:
        return {
            "attacks": list(self.attacks),
            "bin_count": self.bin_count,
            "block_length": self.block_length,
            "bootstrap_resamples": self.bootstrap_resamples,
            "bootstrap_seed": self.bootstrap_seed,
            "bmaq_malicious_count": self.bmaq_malicious_count,
            "bmaq_worker_count": self.bmaq_worker_count,
            "bucket_counts": list(self.bucket_counts),
            "embargo_rounds": self.embargo_rounds,
            "ibrl_malicious_count": self.ibrl_malicious_count,
            "ibrl_worker_count": self.ibrl_worker_count,
            "minimum_coverage": self.minimum_coverage,
            "outer_fraction": self.outer_fraction,
            "outer_starts": list(self.outer_starts),
            "blocks_per_fold": self.blocks_per_fold,
            "proof_mode": self.proof_mode,
            "trial_ids": list(self.trial_ids),
            "attack_parameters": self.attack_parameters.as_manifest_fragment(),
        }


@dataclass(frozen=True)
class ExperimentRun:
    """一次运行产生的内存记录和已发布文件路径。"""

    round_metrics: tuple[RoundMetrics, ...]
    trial_metrics: tuple[TrialMetrics, ...]
    proof_failures: tuple[str, ...]
    output_directory: Path | None = None


@dataclass
class _ProofAuditor:
    mode: str
    dataset: str
    attack: str
    selected_trial_id: int
    trial_id: int
    bin_config: ResidualBinConfig
    failures: list[str] = field(default_factory=list)
    sampled_honest_done: bool = False
    sampled_malicious_done: bool = False

    def should_run_real(self, round_index: int, is_malicious: bool, present: bool, round_count: int) -> bool:
        if not present:
            return False
        if self.mode == "full":
            return True
        if self.mode != "sampled" or self.trial_id != self.selected_trial_id:
            return False
        if round_index == 0 and not is_malicious and not self.sampled_honest_done:
            self.sampled_honest_done = True
            return True
        if round_index == round_count - 1 and is_malicious and not self.sampled_malicious_done:
            self.sampled_malicious_done = True
            return True
        return False

    def verify(
        self,
        *,
        report: FloatArray,
        reference: FloatArray,
        worker_id: str,
        round_index: int,
        is_malicious: bool,
        present: bool,
        round_count: int,
    ) -> bool:
        """先执行整数契约，再按模式选择真实 Ristretto 证明。"""

        if not present:
            return False
        try:
            encoded_report = tuple(encode(float(value)) for value in report)
            encoded_truth = tuple(encode(float(value)) for value in reference)
            residual = exact_residual(encoded_report, encoded_truth, self.bin_config)
            bucket = compute_residual_bin(residual, self.bin_config)
            if not bucket.lower <= residual <= bucket.upper:
                return False
        except (OverflowError, ValueError) as error:
            self.failures.append(f"contract:{self.dataset}:{self.attack}:{round_index}:{worker_id}:{error}")
            return False
        if not self.should_run_real(round_index, is_malicious, present, round_count):
            return True
        try:
            # 真实证明仅服务研究一致性检查。密钥来自 testing 辅助，不构成部署密钥管理。
            from brptd.proofs import build_proof_context, verify_residual_bin
            from brptd.proofs.testing import (
                encrypt_measurements_deterministic,
                generate_test_keypair,
                prove_residual_bin_deterministic,
            )

            seed_material = f"{self.dataset}|{self.attack}|{self.trial_id}|{round_index}|{worker_id}".encode()
            seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
            keypair = generate_test_keypair(seed=seed)
            context = build_proof_context(
                task="brptd-experiment",
                dataset=self.dataset,
                worker=worker_id,
                round_id=round_index,
                truths=encoded_truth,
                etas=self.bin_config.etas,
                measurement_domains=self.bin_config.domains,
                delta_bin=self.bin_config.delta_bin,
                public_key=keypair.public_key,
            )
            encrypted = encrypt_measurements_deterministic(encoded_report, keypair.public_key, seed=seed + 1)
            proof = prove_residual_bin_deterministic(
                encrypted,
                encoded_truth,
                self.bin_config.etas,
                self.bin_config.domains,
                self.bin_config.delta_bin,
                keypair.public_key,
                context,
                seed=seed + 2,
            )
            result = verify_residual_bin(
                proof,
                encoded_truth,
                self.bin_config.etas,
                self.bin_config.domains,
                self.bin_config.delta_bin,
                keypair.public_key,
                context,
            )
            return result.label == bucket.label and result.lower == bucket.lower and result.upper == bucket.upper
        except Exception as error:
            self.failures.append(f"real-proof:{self.dataset}:{self.attack}:{round_index}:{worker_id}:{error}")
            return False


def _scores(
    reports: FloatArray, present: BoolArray, reference: FloatArray, config: ResidualBinConfig
) -> tuple[FloatArray, FloatArray, IntArray]:
    exact = np.zeros(reports.shape[0], dtype=np.float64)
    proxy = np.zeros(reports.shape[0], dtype=np.float64)
    labels = np.zeros(reports.shape[0], dtype=np.int64)
    encoded_truth = tuple(encode(float(value)) for value in reference)
    for index in np.flatnonzero(present):
        encoded = tuple(encode(float(value)) for value in reports[index])
        residual = exact_residual(encoded, encoded_truth, config)
        bucket = compute_residual_bin(residual, config)
        exact[index] = float(decode(residual, config.normalization_scale))
        proxy[index] = float(bucket.proxy)
        labels[index] = bucket.label
    return exact, proxy, labels


def _weighted_update(
    reports: FloatArray, present: BoolArray, weights: Sequence[float], fallback: FloatArray
) -> FloatArray:
    weight_array = np.asarray(weights, dtype=np.float64)
    active = present & (weight_array > 0.0)
    total = float(np.sum(weight_array[active], dtype=np.float64))
    if total <= 0.0:
        return np.array(fallback, dtype=np.float64, copy=True)
    updated = np.sum(reports[active] * weight_array[active, None], axis=0, dtype=np.float64) / total
    return np.asarray(updated, dtype=np.float64)


class _RoundFangEvaluator(FangEvaluator):
    """把当前桶残差 PP-CH 预览适配给 Fang，接口中没有攻击标签。"""

    def __init__(self, reference: FloatArray, bin_config: ResidualBinConfig) -> None:
        self._reference = np.array(reference, dtype=np.float64, copy=True)
        self._bin_config = bin_config

    def preview(self, reports: FloatArray, present: BoolArray, state: Any) -> AggregationPreview:
        if not isinstance(state, PPCHState):
            raise ExperimentError("Fang 预览状态必须为 PPCHState")
        _exact, proxy, _labels = _scores(reports, present, self._reference, self._bin_config)
        decision = state.preview(
            tuple(float(value) for value in proxy),
            tuple(bool(value) for value in present),
            tuple(bool(value) for value in present),
        )
        estimate = _weighted_update(reports, present, decision.final_weights, self._reference)
        return AggregationPreview(
            estimate=estimate, weights=np.asarray(decision.final_weights), valid_update=decision.valid_update
        )


def _initial_reference(reports: FloatArray, present: BoolArray, fallback: FloatArray) -> FloatArray:
    available = reports[present]
    return np.median(available, axis=0) if available.size else np.array(fallback, dtype=np.float64, copy=True)


def execute_attack_scenario(
    *,
    base: BaseScenario,
    attack: str,
    malicious_count: int,
    trial_id: int,
    fold: int,
    block: int,
    proof_mode: str = "contract",
    bin_count: int = 8192,
    sample_real_proofs: bool = False,
    attack_parameters: AttackParameters = DEFAULT_ATTACK_PARAMETERS,
) -> ExperimentRun:
    """执行一个 15 轮 BRPTD 场景，并输出桶/精确内部对照指标。"""

    if proof_mode not in PROOF_MODES:
        raise ExperimentError("未知 proof_mode")
    if attack not in ATTACKS or base.round_count != 15:
        raise ExperimentError("仅接受五类主攻击和固定 15 轮场景")
    if fold < 0 or block < 0:
        raise ExperimentError("fold 与 block 必须非负")
    if not isinstance(attack_parameters, AttackParameters):
        raise ExperimentError("attack_parameters 必须是 AttackParameters")
    bin_config = ResidualBinConfig.from_standardized_domains(base.standardized_domains, bin_count=bin_count)
    mask = base.malicious_mask(malicious_count)
    if attack == "fang":
        reports_all: FloatArray | None = None
    else:
        reports_all = np.array(
            build_attack_scenario(base, attack, malicious_count, parameters=attack_parameters).reports,
            dtype=np.float64,
            copy=True,
        )
    initial_reports = base.honest_reports[0] if reports_all is None else reports_all[0]
    exact_reference = _initial_reference(initial_reports, base.present[0], base.truth[0])
    bucket_reference = np.array(exact_reference, dtype=np.float64, copy=True)
    exact_state = PPCHState(base.worker_count)
    bucket_state = PPCHState(base.worker_count)
    auditor = _ProofAuditor(
        proof_mode,
        base.dataset,
        attack,
        trial_id if sample_real_proofs else -1,
        trial_id,
        bin_config,
    )
    rounds: list[RoundMetrics] = []
    exact_squared = 0.0
    bucket_squared = 0.0
    spearmans: list[float] = []
    shares: list[float] = []
    acceptances: list[float] = []
    valid_updates: list[bool] = []
    uncalculable_spearman = 0
    uncalculable_ratio = 0

    for round_index in range(base.round_count):
        present = np.array(base.present[round_index], dtype=np.bool_, copy=True)
        if reports_all is None:
            reports = np.array(base.honest_reports[round_index], dtype=np.float64, copy=True)
            reports = optimize_fang_round(
                reports=reports,
                present=present,
                truth=base.truth[round_index],
                malicious_mask=mask,
                standardized_domains=base.standardized_domains,
                evaluator=_RoundFangEvaluator(bucket_reference, bin_config),
                state=bucket_state,
                step_size=attack_parameters.fang_step_size,
                maximum_steps=attack_parameters.fang_maximum_steps,
                tolerance=attack_parameters.fang_tolerance,
            )
        else:
            reports = np.array(reports_all[round_index], dtype=np.float64, copy=True)
        exact_scores, proxy_scores, _labels = _scores(reports, present, exact_reference, bin_config)
        # 真正验证在当前桶参考向量下进行，失败后不会进入 bucket_state。
        _bucket_exact, proxy_scores, _labels = _scores(reports, present, bucket_reference, bin_config)
        verified = np.zeros(base.worker_count, dtype=np.bool_)
        for worker_index in np.flatnonzero(present):
            verified[worker_index] = auditor.verify(
                report=reports[worker_index],
                reference=bucket_reference,
                worker_id=base.worker_ids[worker_index],
                round_index=round_index,
                is_malicious=bool(mask[worker_index]),
                present=True,
                round_count=base.round_count,
            )
        exact_decision = exact_state.update(
            tuple(float(value) for value in exact_scores),
            tuple(bool(value) for value in present),
            tuple(bool(value) for value in present),
        )
        bucket_decision = bucket_state.update(
            tuple(float(value) for value in proxy_scores),
            tuple(bool(value) for value in present),
            tuple(bool(value) for value in verified),
        )
        exact_reference = _weighted_update(reports, present, exact_decision.final_weights, exact_reference)
        bucket_reference = _weighted_update(reports, present, bucket_decision.final_weights, bucket_reference)
        exact_round = standardized_crse(exact_reference, base.truth[round_index])
        bucket_round = standardized_crse(bucket_reference, base.truth[round_index])
        exact_squared += exact_round * exact_round
        bucket_squared += bucket_round * bucket_round
        rank_indices = np.flatnonzero(present)
        ranking_exact = exact_scores[rank_indices]
        ranking_proxy = proxy_scores[rank_indices]
        spearman = worker_spearman(ranking_exact, ranking_proxy)
        if spearman is None:
            uncalculable_spearman += 1
        else:
            spearmans.append(spearman)
        share = malicious_weight_share(np.asarray(bucket_decision.final_weights), mask)
        if share is not None:
            shares.append(share)
        acceptance = proof_acceptance_rate(verified[present]) if np.any(present) else 0.0
        acceptances.append(acceptance)
        valid_updates.append(bucket_decision.valid_update)
        ratio = crse_ratio(bucket_round, exact_round)
        if ratio is None:
            uncalculable_ratio += 1
        rounds.append(
            RoundMetrics(
                dataset=base.dataset,
                attack=attack,
                trial_id=trial_id,
                fold=fold,
                block=block,
                round_index=round_index,
                nominal_malicious_ratio=malicious_count / base.worker_count,
                actual_malicious_ratio=float(np.mean(mask)),
                proof_mode=proof_mode,
                exact_crse=exact_round,
                bucket_crse=bucket_round,
                crse_ratio=ratio,
                exact_worker_ranks=average_ranks(ranking_exact),
                proxy_worker_ranks=average_ranks(ranking_proxy),
                spearman=spearman,
                malicious_weight_share=share,
                proof_acceptance_rate=acceptance,
                valid_update=bucket_decision.valid_update,
            )
        )
    if proof_mode == "sampled" and sample_real_proofs:
        if not auditor.sampled_honest_done:
            auditor.failures.append(f"sampled-proof:{base.dataset}:{attack}:honest-first-report-unavailable")
        if not auditor.sampled_malicious_done:
            auditor.failures.append(f"sampled-proof:{base.dataset}:{attack}:malicious-last-report-unavailable")
    exact_trial = math.sqrt(exact_squared)
    bucket_trial = math.sqrt(bucket_squared)
    trial_ratio = crse_ratio(bucket_trial, exact_trial)
    if trial_ratio is None:
        uncalculable_ratio += 1
    trial = TrialMetrics(
        dataset=base.dataset,
        attack=attack,
        trial_id=trial_id,
        fold=fold,
        block=block,
        nominal_malicious_ratio=malicious_count / base.worker_count,
        actual_malicious_ratio=float(np.mean(mask)),
        proof_mode=proof_mode,
        exact_crse=exact_trial,
        bucket_crse=bucket_trial,
        crse_ratio=trial_ratio,
        mean_spearman=float(np.mean(spearmans)) if spearmans else None,
        malicious_weight_share=float(np.mean(shares)) if shares else None,
        proof_acceptance_rate=float(np.mean(acceptances)),
        invalid_round_rate=invalid_round_rate(np.asarray(valid_updates, dtype=np.bool_)),
        uncalculable_spearman_count=uncalculable_spearman,
        uncalculable_ratio_count=uncalculable_ratio,
    )
    return ExperimentRun(tuple(rounds), (trial,), tuple(auditor.failures))


def _load_panel(dataset: str, data_root: Path) -> tuple[SparsePanel, tuple[Path, ...]]:
    """加载已校验的原始文件，并保留清单所需的精确文件集合。"""

    if dataset == "ibrl":
        candidates = tuple(sorted(data_root.glob("ibrl/*.txt")))
        if len(candidates) != 1:
            raise ExperimentError("IBRL 数据目录必须恰好包含一个已解压 .txt 文件")
        return load_ibrl(candidates[0]), candidates
    if dataset == "bmaq":
        candidates = tuple(sorted(data_root.glob("bmaq/**/*.csv")))
        if len(candidates) != 12:
            raise ExperimentError("BMAQ 数据目录必须恰好包含 12 个站点 CSV")
        return load_bmaq(candidates), candidates
    raise ExperimentError(f"未知数据集：{dataset}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_data_hashes(paths: Sequence[Path], data_root: Path) -> dict[str, str]:
    """以相对路径和 SHA256 固化一次运行实际读取的原始输入。"""

    result: dict[str, str] = {}
    for path in paths:
        try:
            name = path.relative_to(data_root).as_posix()
        except ValueError:
            name = path.name
        if name in result:
            raise ExperimentError(f"原始数据哈希路径重复：{name}")
        result[name] = _sha256_file(path)
    return result


def _physical_interval(panel: SparsePanel, indices: Sequence[int]) -> dict[str, Any]:
    """把稳定的轮次索引转换为面板中可复核的物理时间范围。"""

    selected = tuple(int(index) for index in indices)
    if not selected:
        return {"round_count": 0, "time_start": None, "time_end": None}
    return {
        "round_count": len(selected),
        "time_start": panel.timestamps[selected[0]].isoformat(),
        "time_end": panel.timestamps[selected[-1]].isoformat(),
    }


def _transform_manifest(transform: Any) -> dict[str, Any]:
    """保存仅由训练期拟合的数值变换，防止实验结果脱离其尺度契约。"""

    return {
        "center": [float(value) for value in transform.center],
        "mad": [float(value) for value in transform.mad],
        "scale": [float(value) for value in transform.scale],
        "selected_worker_ids": list(transform.selected_worker_ids),
        "sigma_h": [float(value) for value in transform.sigma_h],
        "sigma_m": [float(value) for value in transform.sigma_m],
        "standardized_domains": [[float(lower), float(upper)] for lower, upper in transform.standardized_domains],
        "training_indices": list(transform.training_indices),
    }


def _dataset_trials(
    dataset: str, data_root: Path, config: ExperimentConfig
) -> Iterable[tuple[BaseScenario, int, int, int, Mapping[str, Any]]]:
    source_panel, source_paths = _load_panel(dataset, data_root)
    source_hashes = _source_data_hashes(source_paths, data_root)
    spec = IBRL_SPEC if dataset == "ibrl" else BMAQ_SPEC
    worker_count = config.ibrl_worker_count if dataset == "ibrl" else config.bmaq_worker_count
    if dataset == "ibrl":
        minimum_active_workers = math.ceil(config.minimum_coverage * worker_count)
        raw = restrict_to_active_prefix(source_panel, minimum_active_workers=minimum_active_workers)
        availability_schedule: Mapping[str, Any] = {
            "minimum_active_workers": minimum_active_workers,
            "policy": "raw-arrival-active-prefix",
            "retained_round_count": raw.round_count,
            "retained_time": _physical_interval(raw, tuple(range(raw.round_count))),
            "source_round_count": source_panel.round_count,
            "source_time": _physical_interval(source_panel, tuple(range(source_panel.round_count))),
        }
    else:
        raw = source_panel
        availability_schedule = {
            "policy": "untrimmed",
            "retained_round_count": raw.round_count,
            "retained_time": _physical_interval(raw, tuple(range(raw.round_count))),
            "source_round_count": source_panel.round_count,
        }
    boundaries = build_fold_boundaries(
        raw.round_count,
        outer_starts=config.outer_starts,
        outer_fraction=config.outer_fraction,
        embargo_rounds=config.embargo_rounds,
    )
    trial_index = 0
    for boundary in boundaries:
        panel = select_training_panel(raw, boundary.training_indices, worker_count) if dataset == "ibrl" else raw
        folds = select_fold_blocks(
            panel,
            (boundary,),
            block_length=config.block_length,
            blocks_per_fold=config.blocks_per_fold,
            minimum_coverage=config.minimum_coverage,
        )
        fold = folds[0]
        transform = fit_training_transform(panel, spec, boundary.training_indices)
        for block in fold.blocks:
            if trial_index >= len(config.trial_ids):
                raise ExperimentError("构造的时间块数量超过固定 20 次试验")
            block_panel = panel.take_rounds(block.indices)
            truth = build_clean_truth(block_panel, transform)
            trial_id = config.trial_ids[trial_index]
            base = build_base_scenario(
                dataset=dataset,
                worker_ids=panel.worker_ids,
                truth=truth,
                present=block_panel.present,
                standardized_domains=transform.standardized_domains,
                sigma_h=transform.sigma_h,
                seeds=derive_trial_seeds(dataset, boundary.fold_id, block.block_id, trial_id),
            )
            yield (
                base,
                boundary.fold_id,
                block.block_id,
                trial_id,
                {
                    **base.as_manifest_fragment(),
                    "attack_report_seeds": {attack: base.attack_seed(attack) for attack in config.attacks},
                    "block": block.block_id,
                    "fold": boundary.fold_id,
                    "physical_time": {
                        "block": _physical_interval(panel, block.indices),
                        "embargo": _physical_interval(panel, boundary.embargo_indices),
                        "outer": _physical_interval(panel, boundary.outer_indices),
                        "training": _physical_interval(panel, boundary.training_indices),
                    },
                    "availability_schedule": dict(availability_schedule),
                    "source_data_hashes": source_hashes,
                    "training_transform": _transform_manifest(transform),
                    "trial_id": trial_id,
                },
            )
            trial_index += 1
    if trial_index != 20:
        raise ExperimentError("每个数据集必须构造恰好 20 个时间块")


def _git_commit() -> str | None:
    try:
        root = Path(__file__).resolve().parents[3]
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _dependency_versions() -> dict[str, str]:
    names = ("numpy", "pandas", "scipy", "pooch")
    output: dict[str, str] = {}
    for name in names:
        try:
            output[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            output[name] = "unavailable"
    return output


def _publish_run(
    *,
    output_directory: Path,
    config: ExperimentConfig,
    round_records: Sequence[RoundMetrics],
    trial_records: Sequence[TrialMetrics],
    proof_failures: Sequence[str],
    scenario_manifest: Sequence[Mapping[str, Any]],
) -> ExperimentRun:
    output_directory.mkdir(parents=True, exist_ok=True)
    summary = summarize_trial_metrics(
        trial_records,
        resamples=config.bootstrap_resamples,
        seed=config.bootstrap_seed,
    )
    write_round_metrics(output_directory / "round_metrics.csv", round_records)
    write_trial_metrics(output_directory / "trial_metrics.csv", trial_records)
    write_summary(output_directory / "summary.csv", summary)
    manifest = {
        "code_commit": _git_commit(),
        "config": config.as_manifest_fragment(),
        "dependency_versions": _dependency_versions(),
        "failure_reasons": list(proof_failures),
        "proof_mode": config.proof_mode,
        "scenarios": list(scenario_manifest),
        "source_data_hashes": {
            str(scenario["dataset"]): scenario["source_data_hashes"]
            for scenario in scenario_manifest
            if isinstance(scenario.get("dataset"), str) and isinstance(scenario.get("source_data_hashes"), Mapping)
        },
    }
    write_manifest(output_directory / "manifest.json", manifest)
    return ExperimentRun(tuple(round_records), tuple(trial_records), tuple(proof_failures), output_directory)


DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig()


def run_attack_matrix(
    *,
    data_root: str | Path,
    output_directory: str | Path,
    config: ExperimentConfig = DEFAULT_EXPERIMENT_CONFIG,
) -> ExperimentRun:
    """运行两数据集、五攻击、20 次配对 trial 的主攻击矩阵。"""

    rounds: list[RoundMetrics] = []
    trials: list[TrialMetrics] = []
    failures: list[str] = []
    scenarios: list[Mapping[str, Any]] = []
    for dataset in ("ibrl", "bmaq"):
        malicious_count = config.ibrl_malicious_count if dataset == "ibrl" else config.bmaq_malicious_count
        for base, fold, block, trial_id, scenario in _dataset_trials(dataset, Path(data_root), config):
            scenarios.append(
                {
                    **scenario,
                    "actual_malicious_ratio": malicious_count / base.worker_count,
                    "malicious_count": malicious_count,
                }
            )
            for attack in config.attacks:
                result = execute_attack_scenario(
                    base=base,
                    attack=attack,
                    malicious_count=malicious_count,
                    trial_id=trial_id,
                    fold=fold,
                    block=block,
                    proof_mode=config.proof_mode,
                    bin_count=config.bin_count,
                    sample_real_proofs=trial_id == config.trial_ids[0],
                    attack_parameters=config.attack_parameters,
                )
                rounds.extend(result.round_metrics)
                trials.extend(result.trial_metrics)
                failures.extend(result.proof_failures)
    if len(trials) != 200 or len(rounds) != 3_000:
        raise ExperimentError(f"主矩阵记录数量错误：trial={len(trials)}，round={len(rounds)}")
    return _publish_run(
        output_directory=Path(output_directory),
        config=config,
        round_records=rounds,
        trial_records=trials,
        proof_failures=failures,
        scenario_manifest=scenarios,
    )


def run_bucket_sensitivity(
    *,
    data_root: str | Path,
    output_directory: str | Path,
    config: ExperimentConfig = DEFAULT_EXPERIMENT_CONFIG,
) -> ExperimentRun:
    """在五攻击与 10/30/50% 嵌套恶意前缀下扫描 K=2^3..2^13。"""

    rounds: list[RoundMetrics] = []
    trials: list[TrialMetrics] = []
    failures: list[str] = []
    scenarios: list[Mapping[str, Any]] = []
    ratio_counts = {"ibrl": (5, 15, 25), "bmaq": (1, 4, 6)}
    for dataset in ("ibrl", "bmaq"):
        for base, fold, block, trial_id, scenario in _dataset_trials(dataset, Path(data_root), config):
            for malicious_count in ratio_counts[dataset]:
                scenarios.append(
                    {
                        **scenario,
                        "actual_malicious_ratio": malicious_count / base.worker_count,
                        "malicious_count": malicious_count,
                    }
                )
                for bin_count in config.bucket_counts:
                    for attack in config.attacks:
                        result = execute_attack_scenario(
                            base=base,
                            attack=attack,
                            malicious_count=malicious_count,
                            trial_id=trial_id,
                            fold=fold,
                            block=block,
                            proof_mode=config.proof_mode,
                            bin_count=bin_count,
                            sample_real_proofs=(
                                trial_id == config.trial_ids[0]
                                and malicious_count == ratio_counts[dataset][0]
                                and bin_count == config.bin_count
                            ),
                            attack_parameters=config.attack_parameters,
                        )
                        # 每个 K 单独建组，proof_mode 字段编码其数值以维持稳定 CSV 模式。
                        relabeled_trials = [
                            TrialMetrics(**{**record.__dict__, "proof_mode": f"{record.proof_mode}:K={bin_count}"})
                            for record in result.trial_metrics
                        ]
                        relabeled_rounds = [
                            RoundMetrics(**{**record.__dict__, "proof_mode": f"{record.proof_mode}:K={bin_count}"})
                            for record in result.round_metrics
                        ]
                        rounds.extend(relabeled_rounds)
                        trials.extend(relabeled_trials)
                        failures.extend(result.proof_failures)
    if len(trials) != 6_600 or len(rounds) != 99_000:
        raise ExperimentError(f"桶扫描记录数量错误：trial={len(trials)}，round={len(rounds)}")
    return _publish_run(
        output_directory=Path(output_directory),
        config=config,
        round_records=rounds,
        trial_records=trials,
        proof_failures=failures,
        scenario_manifest=scenarios,
    )
