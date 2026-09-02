"""BRPTD 的受限命令行入口。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .data import fetch_dataset
from .experiments import ExperimentConfig, run_attack_matrix, run_bucket_sensitivity
from .simulation import AttackParameters


def _toml(path: Path) -> Mapping[str, Any]:
    try:
        import tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        try:
            import tomli as tomllib
        except ModuleNotFoundError as error:
            raise RuntimeError("Python 3.10 需要安装 tomli 才能读取实验配置") from error
    with path.open("rb") as handle:
        parsed = tomllib.load(handle)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"配置根必须是表：{path}")
    return parsed


def _table(source: Mapping[str, Any], name: str) -> dict[str, Any]:
    """读取一个必需的 TOML 表，并在边界处完成类型收窄。"""

    value = source.get(name)
    if not isinstance(value, dict):
        raise RuntimeError(f"配置缺少 {name} 表")
    return {str(key): item for key, item in value.items()}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _attack_parameters(source: Mapping[str, Any]) -> AttackParameters:
    """把版本化 TOML 中的攻击数值转换为不可变运行参数。"""

    try:
        return AttackParameters(
            bias_offset=float(source["bias"]),
            drift_per_round=float(source["drift_per_round"]),
            spike_upper_probability=float(source["spike_upper_probability"]),
            fang_step_size=float(source["fang_step_size"]),
            fang_maximum_steps=int(source["fang_maximum_steps"]),
            fang_tolerance=float(source["fang_tolerance"]),
        )
    except KeyError as error:
        raise RuntimeError(f"攻击配置缺少字段：{error.args[0]}") from error


def _time_config(source: Mapping[str, Any]) -> dict[str, Any]:
    """读取时间折字段，确保它们确实成为运行器输入而非仅作注释。"""

    try:
        return {
            "outer_starts": tuple(float(value) for value in source["outer_starts"]),
            "outer_fraction": float(source["outer_fraction"]),
            "embargo_rounds": int(source["embargo_rounds"]),
            "block_length": int(source["block_length"]),
            "blocks_per_fold": int(source["blocks_per_fold"]),
            "minimum_coverage": float(source["minimum_coverage"]),
        }
    except KeyError as error:
        raise RuntimeError(f"时间折配置缺少字段：{error.args[0]}") from error


def _load_attack_config(path: Path, proof_mode: str | None) -> ExperimentConfig:
    source = _toml(path)
    run = _table(source, "run")
    datasets = _table(source, "datasets")
    attacks = _table(source, "attacks")
    trials = _table(source, "trials")
    ibrl = _table(datasets, "ibrl")
    bmaq = _table(datasets, "bmaq")
    return ExperimentConfig(
        proof_mode=proof_mode or str(run["proof_mode"]),
        bin_count=int(run["bin_count"]),
        attacks=tuple(str(name) for name in attacks["names"]),
        trial_ids=tuple(int(value) for value in trials["ids"]),
        bootstrap_resamples=int(run["bootstrap_resamples"]),
        bootstrap_seed=int(run["bootstrap_seed"]),
        ibrl_worker_count=int(ibrl["worker_count"]),
        bmaq_worker_count=int(bmaq["worker_count"]),
        ibrl_malicious_count=int(ibrl["malicious_count"]),
        bmaq_malicious_count=int(bmaq["malicious_count"]),
        attack_parameters=_attack_parameters(attacks),
        **_time_config(trials),
    )


def _load_bucket_config(path: Path, proof_mode: str | None) -> ExperimentConfig:
    source = _toml(path)
    run = _table(source, "run")
    trials = _table(source, "trials")
    attacks = _table(source, "attacks")
    buckets = _table(source, "buckets")
    powers = tuple(int(value) for value in buckets["powers"])
    return ExperimentConfig(
        proof_mode=proof_mode or str(run["proof_mode"]),
        bin_count=8192,
        bucket_counts=tuple(2**power for power in powers),
        attacks=tuple(str(name) for name in attacks["names"]),
        trial_ids=tuple(int(value) for value in trials["ids"]),
        bootstrap_resamples=int(run["bootstrap_resamples"]),
        bootstrap_seed=int(run["bootstrap_seed"]),
        attack_parameters=_attack_parameters(attacks),
        **_time_config(trials),
    )


def _add_common_experiment_options(parser: argparse.ArgumentParser, *, default_config: Path) -> None:
    parser.add_argument("--config", type=Path, default=default_config, help="固定实验 TOML 配置")
    parser.add_argument("--data-path", type=Path, default=_repo_root() / "data" / "raw", help="已校验原始数据根目录")
    parser.add_argument("--output-path", type=Path, default=_repo_root() / "artifacts", help="本地生成制品目录")
    parser.add_argument("--parallelism", type=int, default=1, help="独立 trial 的并行度，至少为 1")
    parser.add_argument("--proof-mode", choices=("contract", "sampled", "full"), help="证明执行模式")


def build_parser() -> argparse.ArgumentParser:
    """构造显式而受限的命令树。"""

    parser = argparse.ArgumentParser(prog="brptd", description="BRPTD 核心实验仓库")
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("data", help="官方数据获取")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    fetch = data_commands.add_parser("fetch", help="下载并校验官方数据")
    fetch.add_argument("dataset", choices=("ibrl", "bmaq", "all"))
    fetch.add_argument("--data-path", type=Path, default=_repo_root() / "data" / "raw")
    fetch.add_argument("--manifest-path", type=Path, default=_repo_root() / "data" / "manifests")
    experiment = commands.add_parser("experiment", help="BRPTD 实验")
    experiment_commands = experiment.add_subparsers(dest="experiment_command", required=True)
    attacks = experiment_commands.add_parser("attacks", help="五类攻击主矩阵")
    _add_common_experiment_options(attacks, default_config=_repo_root() / "configs" / "paper_attacks.toml")
    buckets = experiment_commands.add_parser("buckets", help="残差桶敏感性")
    _add_common_experiment_options(buckets, default_config=_repo_root() / "configs" / "bucket_sensitivity.toml")
    proof = commands.add_parser("proof", help="Ristretto255 证明检查")
    proof_commands = proof.add_subparsers(dest="proof_command", required=True)
    validate = proof_commands.add_parser("validate", help="执行一个真实正例和篡改负例")
    validate.add_argument("--output-path", type=Path, default=_repo_root() / "artifacts" / "proof_validate.json")
    return parser


def _proof_validate(output_path: Path) -> int:
    from .artifacts import write_manifest
    from .proofs import ProofVerificationError, ResidualBinProof, build_proof_context, verify_residual_bin
    from .proofs.testing import (
        encrypt_measurements_deterministic,
        generate_test_keypair,
        prove_residual_bin_deterministic,
    )

    keypair = generate_test_keypair(seed=20260901)
    measurements = (26, 65)
    truths = (25, 80)
    etas = (1, 1)
    domains = ((20, 30), (60, 85))
    context = build_proof_context(
        task="proof-validate",
        dataset="fixture",
        worker="fixture-worker",
        round_id=0,
        truths=truths,
        etas=etas,
        measurement_domains=domains,
        delta_bin=8,
        public_key=keypair.public_key,
    )
    encrypted = encrypt_measurements_deterministic(measurements, keypair.public_key, seed=20260902)
    proof = prove_residual_bin_deterministic(
        encrypted,
        truths,
        etas,
        domains,
        8,
        keypair.public_key,
        context,
        seed=20260903,
    )
    result = verify_residual_bin(proof, truths, etas, domains, 8, keypair.public_key, context)
    tampered = bytearray(proof.payload)
    tampered[-1] ^= 1
    try:
        verify_residual_bin(ResidualBinProof(bytes(tampered)), truths, etas, domains, 8, keypair.public_key, context)
    except ProofVerificationError:
        tamper_status = "rejected"
    else:
        raise RuntimeError("篡改后的证明意外通过验证")
    write_manifest(
        output_path,
        {
            "context_digest": result.context_digest,
            "label": result.label,
            "positive_status": "verified",
            "tamper_status": tamper_status,
        },
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """执行 CLI，并把可操作错误渲染为标准错误输出。"""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "data" and arguments.data_command == "fetch":
            datasets = ("ibrl", "bmaq") if arguments.dataset == "all" else (arguments.dataset,)
            paths = [
                str(path)
                for dataset in datasets
                for path in fetch_dataset(
                    dataset,
                    data_root=arguments.data_path,
                    manifest_directory=arguments.manifest_path,
                )
            ]
            print(json.dumps({"files": paths}, ensure_ascii=False, sort_keys=True))
            return 0
        if arguments.command == "experiment" and arguments.experiment_command == "attacks":
            if arguments.parallelism < 1:
                raise RuntimeError("parallelism 至少为 1")
            config = _load_attack_config(arguments.config, arguments.proof_mode)
            result = run_attack_matrix(
                data_root=arguments.data_path, output_directory=arguments.output_path, config=config
            )
            print(
                json.dumps(
                    {"round_records": len(result.round_metrics), "trial_records": len(result.trial_metrics)},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        if arguments.command == "experiment" and arguments.experiment_command == "buckets":
            if arguments.parallelism < 1:
                raise RuntimeError("parallelism 至少为 1")
            config = _load_bucket_config(arguments.config, arguments.proof_mode)
            result = run_bucket_sensitivity(
                data_root=arguments.data_path, output_directory=arguments.output_path, config=config
            )
            print(
                json.dumps(
                    {"round_records": len(result.round_metrics), "trial_records": len(result.trial_metrics)},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        if arguments.command == "proof" and arguments.proof_command == "validate":
            return _proof_validate(arguments.output_path)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    parser.error("无法识别命令")


if __name__ == "__main__":
    raise SystemExit(main())
