from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


# 关键路径：脚本默认从仓库根目录下的 `ckpts/` 工作，但并不绑定任何具体模型。
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = REPO_ROOT / "ckpts" / "voxcpm2-gguf"


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    参数:
        无。

    返回:
        返回 argparse 解析后的参数对象。

    边界异常:
        参数非法时由 argparse 自动打印帮助并退出。
    """

    parser = argparse.ArgumentParser(
        description="根据模型目录下当前实际存在的 ckpt 文件，重建根 manifest.json。"
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="模型 bundle 根目录，例如 ckpts/voxcpm2-gguf",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="显式指定写入 manifest 的 model 字段；未传时优先复用旧 manifest，再回退到目录名",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="显式指定写入 manifest 的 adapter 字段；未传时优先复用旧 manifest，再回退到 model",
    )
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        default=None,
        help="显式指定写入 manifest 的 sources；可多次传入，未传时优先复用旧 manifest，再回退到 ['local']",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要生成的关键信息，不真正写回 manifest.json",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """
    计算单个文件的 sha256。

    参数:
        path: 待计算文件路径。

    返回:
        返回文件的 64 位十六进制 sha256 字符串。

    边界异常:
        文件缺失或不可读时由上层统一抛出异常。
    """

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_if_exists(path: Path) -> dict[str, Any]:
    """
    按“存在则读取，不存在则返回空字典”的策略读取 JSON。

    参数:
        path: JSON 文件路径。

    返回:
        文件存在时返回解析后的对象；文件不存在时返回空字典。

    边界异常:
        JSON 非法或不是对象时抛出异常，避免把损坏的旧 manifest 当默认值继续传播。
    """

    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"json payload must be an object: {path}")
    return payload


def resolve_bundle_root(model_dir: str) -> Path:
    """
    解析并校验模型 bundle 根目录。

    参数:
        model_dir: 命令行传入的模型目录。

    返回:
        返回已经 resolve 的绝对路径。

    边界异常:
        目录不存在，或缺少 `ckpt/` 子目录时抛出 FileNotFoundError。
    """

    bundle_root = Path(model_dir).expanduser().resolve()
    if not bundle_root.exists():
        raise FileNotFoundError(f"bundle root does not exist: {bundle_root}")
    ckpt_root = bundle_root / "ckpt"
    if not ckpt_root.exists():
        raise FileNotFoundError(f"bundle root does not contain ckpt directory: {ckpt_root}")
    return bundle_root


def collect_ckpt_assets(bundle_root: Path) -> list[Path]:
    """
    收集根 manifest 需要记录的资产文件。

    参数:
        bundle_root: 模型 bundle 根目录。

    返回:
        返回 `ckpt/` 下全部文件的稳定排序结果。

    边界异常:
        `ckpt/` 为空时抛出异常，避免写出一个看似成功但实际不可启动的 manifest。
    """

    ckpt_root = bundle_root / "ckpt"

    # 业务意图：
    # - 当前 StreamVox 的运行期校验只关心 manifest 中 `ckpt/` 下的资产。
    # - 因此这里严格只扫描 `ckpt/`，不再把 README、LICENSE 等顶层文件塞进 manifest。
    # - 这样以后不论你替换 ONNX 精度、GGUF 文件，还是补 tokenizer，只要文件在 `ckpt/` 内，
    #   重跑脚本就能自动同步到 manifest。
    asset_paths = sorted(path.resolve() for path in ckpt_root.rglob("*") if path.is_file())
    if not asset_paths:
        raise FileNotFoundError(f"no asset files found under ckpt directory: {ckpt_root}")
    return asset_paths


def resolve_manifest_identity(bundle_root: Path, args: argparse.Namespace) -> tuple[str, str, list[str]]:
    """
    决定 manifest 的 `model / adapter / sources` 字段。

    参数:
        bundle_root: 模型 bundle 根目录。
        args: 命令行参数。

    返回:
        返回 `(model, adapter, sources)`。

    边界异常:
        旧 manifest 字段类型异常时会抛出清晰错误，避免生成脏配置。
    """

    existing_manifest = read_json_if_exists(bundle_root / "manifest.json")

    # model 字段优先级：
    # 1. 用户显式传参
    # 2. 复用旧 manifest
    # 3. 使用目录名
    model_name = str(args.model or existing_manifest.get("model") or bundle_root.name).strip()
    if not model_name:
        raise ValueError("resolved model name must not be empty")

    # adapter 字段默认跟随旧 manifest 或 model 名，避免无意义地把已有配置洗掉。
    adapter_name = str(args.adapter or existing_manifest.get("adapter") or model_name).strip()
    if not adapter_name:
        raise ValueError("resolved adapter name must not be empty")

    # sources 字段同样优先复用旧值；如果旧值不存在，则回退到最保守的 local。
    if args.sources:
        sources = [str(item).strip() for item in args.sources if str(item).strip()]
    else:
        existing_sources = existing_manifest.get("sources")
        if existing_sources is None:
            sources = ["local"]
        elif not isinstance(existing_sources, list):
            raise TypeError("existing manifest field 'sources' must be a list when present")
        else:
            sources = [str(item).strip() for item in existing_sources if str(item).strip()]
            if not sources:
                sources = ["local"]

    return model_name, adapter_name, sources


def build_manifest_payload(
    *,
    bundle_root: Path,
    model_name: str,
    adapter_name: str,
    sources: list[str],
) -> dict[str, Any]:
    """
    根据当前 `ckpt/` 实际文件生成根 manifest 结构。

    参数:
        bundle_root: 模型 bundle 根目录。
        model_name: 将写入 manifest 的模型名。
        adapter_name: 将写入 manifest 的适配器名。
        sources: 将写入 manifest 的来源列表。

    返回:
        返回可直接序列化为 JSON 的 manifest 对象。

    边界异常:
        文件不可读时由 sha256 计算阶段抛出异常。
    """

    assets: list[dict[str, str]] = []
    for asset_path in collect_ckpt_assets(bundle_root):
        assets.append(
            {
                "path": str(asset_path.relative_to(bundle_root)).replace("\\", "/"),
                "sha256": sha256_file(asset_path),
            }
        )

    return {
        "model": model_name,
        "sources": sources,
        "adapter": adapter_name,
        "assets": assets,
    }


def write_manifest(bundle_root: Path, payload: dict[str, Any], *, dry_run: bool) -> Path:
    """
    把 manifest 写回模型根目录。

    参数:
        bundle_root: 模型 bundle 根目录。
        payload: 已构造好的 manifest 内容。
        dry_run: 为 True 时只打印目标路径，不真正写盘。

    返回:
        返回目标 manifest 路径。

    边界异常:
        目录不可写时由 `write_text` 抛出异常。
    """

    manifest_path = bundle_root / "manifest.json"
    if dry_run:
        print(f"[dry-run] would write: {manifest_path}")
        return manifest_path
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path


def print_summary(
    *,
    bundle_root: Path,
    payload: dict[str, Any],
    manifest_path: Path,
    dry_run: bool,
) -> None:
    """
    输出本次重建摘要。

    参数:
        bundle_root: 模型 bundle 根目录。
        payload: 本次生成的 manifest 内容。
        manifest_path: 目标 manifest 路径。
        dry_run: 是否为 dry-run 模式。

    返回:
        无返回值，只打印摘要。

    边界异常:
        本函数不做 IO 写入，不主动抛异常。
    """

    mode_label = "dry-run" if dry_run else "write"
    print(f"[{mode_label}] bundle_root={bundle_root}")
    print(f"[{mode_label}] model={payload['model']}")
    print(f"[{mode_label}] adapter={payload['adapter']}")
    print(f"[{mode_label}] sources={payload['sources']}")
    print(f"[{mode_label}] manifest={manifest_path}")
    print(f"[{mode_label}] ckpt_asset_count={len(payload['assets'])}")


def main() -> None:
    """
    脚本入口。

    参数:
        无。

    返回:
        无返回值；成功时会重建模型根目录下的 manifest.json。

    边界异常:
        模型目录非法、ckpt 为空或文件不可读时会直接抛错，便于调用方立即定位问题。
    """

    args = parse_args()
    bundle_root = resolve_bundle_root(args.model_dir)
    model_name, adapter_name, sources = resolve_manifest_identity(bundle_root, args)
    payload = build_manifest_payload(
        bundle_root=bundle_root,
        model_name=model_name,
        adapter_name=adapter_name,
        sources=sources,
    )
    manifest_path = write_manifest(bundle_root, payload, dry_run=args.dry_run)
    print_summary(
        bundle_root=bundle_root,
        payload=payload,
        manifest_path=manifest_path,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
