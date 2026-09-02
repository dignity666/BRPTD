"""官方数据下载、解压与 SHA256 完整性校验。"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

import pooch


class DataFetchError(RuntimeError):
    """下载、解压或清单完整性校验失败。"""


def load_manifest(dataset: str, manifest_directory: str | Path = "data/manifests") -> dict[str, Any]:
    """读取并验证版本化数据清单的最小结构。"""

    if dataset not in {"ibrl", "bmaq"}:
        raise DataFetchError(f"未知数据集：{dataset}")
    source = Path(manifest_directory) / f"{dataset}.json"
    try:
        parsed = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise DataFetchError(f"找不到数据清单：{source}") from error
    except json.JSONDecodeError as error:
        raise DataFetchError(f"数据清单 JSON 无法解析：{source}") from error
    dataset_entry = parsed.get("dataset") if isinstance(parsed, dict) else None
    if not isinstance(parsed, dict) or not isinstance(dataset_entry, dict) or dataset_entry.get("id") != dataset:
        raise DataFetchError(f"数据清单与请求的数据集不一致：{source}")
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_archive_hash(manifest: dict[str, Any]) -> str:
    archive = manifest.get("archive")
    if not isinstance(archive, dict):
        raise DataFetchError("数据清单缺少 archive")
    raw_hash = archive.get("sha256")
    if not isinstance(raw_hash, str) or len(raw_hash) != 64:
        # 在网络请求前拒绝未知归档，避免把远端临时内容升级为可信输入。
        raise DataFetchError("清单未锁定压缩包 SHA256；请先在受控下载后补录 archive.sha256，再执行 fetch")
    try:
        int(raw_hash, 16)
    except ValueError as error:
        raise DataFetchError("archive.sha256 不是十六进制 SHA256") from error
    return raw_hash.lower()


def _download_once(
    url: str,
    destination: Path,
    archive_name: str,
    archive_hash: str,
    *,
    connect_timeout: float,
    total_timeout: float,
) -> Path:
    downloader = pooch.HTTPDownloader(timeout=(connect_timeout, total_timeout), progressbar=False)
    try:
        cached = pooch.retrieve(
            url=url,
            known_hash=f"sha256:{archive_hash}",
            path=destination,
            fname=archive_name,
            downloader=downloader,
            progressbar=False,
        )
    except Exception as error:  # Pooch 会把 requests 与哈希错误归一到异常文本。
        raise DataFetchError(f"Pooch 下载或校验失败：{error}") from error
    result = Path(cached)
    if _sha256(result) != archive_hash:
        raise DataFetchError("Pooch 返回文件的 SHA256 与清单不一致")
    return result


def _copy_checked(source: Path, target: Path, expected_hash: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    with source.open("rb") as input_handle, temporary.open("wb") as output_handle:
        shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
        output_handle.flush()
    actual = _sha256(temporary)
    if actual != expected_hash:
        temporary.unlink(missing_ok=True)
        raise DataFetchError(f"解压文件哈希不一致：{target.name}")
    temporary.replace(target)
    return target


def _extract_ibrl(archive: Path, output_root: Path, manifest: dict[str, Any]) -> tuple[Path, ...]:
    files = manifest.get("extraction", {}).get("files")
    if not isinstance(files, list) or len(files) != 1 or not isinstance(files[0], dict):
        raise DataFetchError("IBRL 清单未声明唯一解压文件")
    entry = files[0]
    target = output_root / str(entry["path"])
    temporary = target.with_suffix(target.suffix + ".inflated")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with gzip.open(archive, "rb") as input_handle, temporary.open("wb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            output_handle.flush()
        return (_copy_checked(temporary, target, str(entry["sha256"]).lower()),)
    except OSError as error:
        raise DataFetchError(f"IBRL gzip 解压失败：{error}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _extract_bmaq(archive: Path, output_root: Path, manifest: dict[str, Any]) -> tuple[Path, ...]:
    raw_files = manifest.get("extraction", {}).get("files")
    upstream = manifest.get("archive", {}).get("upstream_member")
    if not isinstance(raw_files, list) or not isinstance(upstream, str):
        raise DataFetchError("BMAQ 清单缺少嵌套 ZIP 或文件列表")
    nested_path = output_root / "_nested.zip"
    try:
        with zipfile.ZipFile(archive) as outer:
            if upstream not in outer.namelist():
                raise DataFetchError("BMAQ 外层 ZIP 缺少声明的嵌套包")
            with outer.open(upstream) as input_handle, nested_path.open("wb") as output_handle:
                shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
        result: list[Path] = []
        with zipfile.ZipFile(nested_path) as inner:
            for raw_entry in raw_files:
                if not isinstance(raw_entry, dict):
                    raise DataFetchError("BMAQ 文件清单项非法")
                name = str(raw_entry["path"])
                members = [member for member in inner.namelist() if member.endswith(name)]
                if len(members) != 1:
                    raise DataFetchError(f"BMAQ 嵌套 ZIP 无法唯一定位文件：{name}")
                temporary = output_root / (name + ".inflated")
                with inner.open(members[0]) as input_handle, temporary.open("wb") as output_handle:
                    shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
                result.append(_copy_checked(temporary, output_root / name, str(raw_entry["sha256"]).lower()))
                temporary.unlink(missing_ok=True)
        return tuple(result)
    except zipfile.BadZipFile as error:
        raise DataFetchError(f"BMAQ ZIP 解压失败：{error}") from error
    finally:
        nested_path.unlink(missing_ok=True)


def fetch_dataset(
    dataset: str,
    *,
    data_root: str | Path = "data/raw",
    manifest_directory: str | Path = "data/manifests",
    connect_timeout: float = 10.0,
    total_timeout: float = 120.0,
) -> tuple[Path, ...]:
    """一次重试的受控下载，并校验压缩包与每个解压文件 SHA256。"""

    if connect_timeout <= 0 or total_timeout <= 0:
        raise DataFetchError("连接和总时限必须为正数")
    manifest = load_manifest(dataset, manifest_directory)
    archive_hash = _require_archive_hash(manifest)
    source = manifest.get("source", {})
    url = source.get("download_url") if isinstance(source, dict) else None
    if not isinstance(url, str) or not url.startswith("https://"):
        raise DataFetchError("清单未声明 HTTPS 下载地址")
    archive_name = manifest.get("archive", {}).get("filename")
    if not isinstance(archive_name, str) or not archive_name:
        raise DataFetchError("清单未声明压缩包文件名")
    root = Path(data_root)
    cache = root / "downloads" / dataset
    cache.mkdir(parents=True, exist_ok=True)
    downloaded: Path | None = None
    failure: DataFetchError | None = None
    for attempt in range(2):
        try:
            downloaded = _download_once(
                url,
                cache,
                archive_name,
                archive_hash,
                connect_timeout=connect_timeout,
                total_timeout=total_timeout,
            )
            break
        except DataFetchError as error:
            failure = error
            if attempt == 0:
                time.sleep(2.0)
    if downloaded is None:
        raise failure if failure is not None else DataFetchError("下载未产生文件")
    output_root = root / dataset
    if dataset == "ibrl":
        return _extract_ibrl(downloaded, output_root, manifest)
    if dataset == "bmaq":
        return _extract_bmaq(downloaded, output_root, manifest)
    raise DataFetchError(f"未知数据集：{dataset}")
