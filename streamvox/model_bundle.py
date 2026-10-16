from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .utils.log import logger
from .utils.performance import timed_block
MANIFEST_FILENAME = "manifest.json"


NON_RUNTIME_MANIFEST_ASSET_NAMES = {
    "LICENSE",
    "NOTICE",
    "README.md",
    "configuration.json",
}


class ModelBundleError(Exception):
    pass


@dataclass(slots=True)
class BundleValidationResult:
    model: str
    manifest_path: Path
    asset_paths: list[Path]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_local_manifest(model_dir: str | Path) -> dict[str, Any]:
    path = manifest_path_for(model_dir)
    if not path.exists():
        raise ModelBundleError(f"missing model manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_path_for(model_dir: str | Path) -> Path:
    return Path(model_dir) / MANIFEST_FILENAME


def is_runtime_manifest_asset(asset_path: str | Path) -> bool:
    """
    判断 manifest 中的资产是否属于运行期必需模型文件。

    参数:
        asset_path: manifest assets[*].path 中声明的相对路径。

    返回:
        返回 True 表示该文件参与 runtime 启动前的存在性和 sha256 校验；返回 False 表示该文件属于文档或仓库元信息。

    异常:
        本函数不主动抛出异常；空路径和非模型文件会被判定为非运行期资产。
    """
    # manifest 路径统一按 POSIX 风格判断，避免 Windows 反斜杠和 Linux 正斜杠导致规则不一致。
    normalized_path = str(asset_path).replace("\\", "/").strip()
    if not normalized_path:
        return False

    # README、LICENSE、NOTICE 等文件不参与模型推理，远端仓库展示内容变化不应该阻断客户启动 runtime。
    if normalized_path in NON_RUNTIME_MANIFEST_ASSET_NAMES:
        return False

    # 当前 StreamVox 模型的实际权重、ONNX、tokenizer、embedding 等运行资产都在 ckpt/ 下。
    # 只校验 ckpt/ 可以避免仓库说明文档漂移影响运行，同时仍然保护真正参与推理的二进制资产。
    return normalized_path.startswith("ckpt/")


def validate_local_bundle(model_dir: str | Path, model: str, verify_model_sha256: bool = True) -> BundleValidationResult:
    """
    校验本地模型 bundle 的 manifest 和资产文件。

    参数:
        model_dir: 本地模型 bundle 根目录。
        model: 调用方期望加载的模型名称。
        verify_model_sha256: 是否根据 manifest 中的 sha256 字段校验每个资产文件。

    返回:
        返回 bundle 校验结果，包含模型名、manifest 路径和资产路径列表。

    异常:
        当 manifest 缺失、模型名不匹配、资产缺失或 sha256 不匹配时抛出 ModelBundleError。
    """
    logger.info(f"[{model}] 正在验证模型文件....")
    root = Path(model_dir)
    manifest_path = manifest_path_for(root)
    manifest = load_local_manifest(root)

    if manifest.get("model") != model:
        raise ModelBundleError("local model manifest does not match requested model")

    asset_paths: list[Path] = []
    # 验证耗时
    with timed_block(f"[{model}]: 模型验证成功"):
        for asset in manifest.get("assets", []):
            # 只对运行期模型资产做启动前强校验；仓库文档类文件可能被平台重写或用户清理，不能影响推理链路。
            if not is_runtime_manifest_asset(asset["path"]):
                continue
            path = root / asset["path"]
            if not path.exists():
                raise ModelBundleError(f"missing model asset: {path}")
            # 用户关闭 sha256 校验时仍然检查资产是否存在，保证 runtime 后续拿到的路径至少是完整清单中的真实文件。
            if verify_model_sha256:
                actual_sha = sha256_file(path)  # sha256验证
                if actual_sha != asset["sha256"]:
                    raise ModelBundleError(f"sha256 mismatch for model asset: {path.name}")
            asset_paths.append(path)


    return BundleValidationResult(model=model, manifest_path=manifest_path, asset_paths=asset_paths)


def build_local_manifest(
    model: str,
    root_dir: str | Path,
    asset_paths: list[str | Path],
    sources: list[str] | None = None,
    adapter: str | None = None,
) -> dict[str, Any]:
    root = Path(root_dir).resolve()
    assets = []
    for asset in asset_paths:
        path = Path(asset).resolve()
        assets.append(
            {
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "sha256": sha256_file(path),
            }
        )
    return {
        "model": model,
        "sources": sources or ["local"],
        "adapter": adapter or model,
        "assets": sorted(assets, key=lambda item: item["path"]),
    }


def write_local_manifest(
    model: str,
    root_dir: str | Path,
    asset_paths: list[str | Path],
    sources: list[str] | None = None,
    adapter: str | None = None,
) -> Path:
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = build_local_manifest(
        model=model,
        root_dir=root,
        asset_paths=asset_paths,
        sources=sources,
        adapter=adapter,
    )
    target = manifest_path_for(root)
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a local StreamVox model manifest.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--asset", action="append", required=True)
    parser.add_argument("--source", action="append")
    parser.add_argument("--adapter")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    path = write_local_manifest(
        model=args.model,
        root_dir=args.root_dir,
        asset_paths=args.asset,
        sources=args.source,
        adapter=args.adapter,
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
