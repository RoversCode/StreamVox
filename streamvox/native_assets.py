"""从固定公开归档准备可再分发的三后端原生库。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
from tempfile import NamedTemporaryFile, TemporaryDirectory
from urllib.request import Request, urlopen
from zipfile import ZipFile


LLAMA_COMMIT = "d0a6dfeb28a09831d904fc4d910ddb740da82834"
LLAMA_TAG = "b8683"


@dataclass(frozen=True, slots=True)
class NativeArchive:
    """记录不可漂移的公开原生资产。

    Args:
        filename: 缓存文件名。
        url: 官方发行下载地址。
        sha256: 归档内容摘要。
    """

    filename: str
    url: str
    sha256: str


LLAMA_ARCHIVES = (
    NativeArchive("llama-b8683-bin-win-cuda-12.4-x64.zip",
                  "https://github.com/ggml-org/llama.cpp/releases/download/b8683/llama-b8683-bin-win-cuda-12.4-x64.zip",
                  "9e594d41e90eb72cd01915d9031fad9842029e1786eed33dbcc9d9091780afea"),
    NativeArchive("llama-b8683-bin-win-vulkan-x64.zip",
                  "https://github.com/ggml-org/llama.cpp/releases/download/b8683/llama-b8683-bin-win-vulkan-x64.zip",
                  "d337941196b5b03227fc151da3b181eb27c2d6ee571803fd8abc9583aeb6356a"),
)
OPENMP_ARCHIVE = NativeArchive(
    "llvm-openmp-15.0.5-h91493d7_0.tar.bz2",
    "https://api.anaconda.org/download/conda-forge/llvm-openmp/15.0.5/win-64/llvm-openmp-15.0.5-h91493d7_0.tar.bz2",
    "b37927ab18ae051745d0cbc9503ea76153b2681dadea3ce6076c909ea879ab1e",
)
# 官方 Windows 归档内的微软 OpenMP 文件不属于可再分发清单，因此使用公开 LLVM 构建替换。
OPENMP_MEMBER = "Library/bin/libomp.dll"
OPENMP_DLL = "libomp140.x86_64.dll"
NATIVE_DLL_SHA256 = {
    "ggml-base.dll": "301f8143737e903e8bfb14c4be62b62a8381be1074132a40438b13072c475d74",
    "ggml-cpu-alderlake.dll": "c2abf093940c8ad54f9c0e632806e2ff6d72700c9105f57c324d31ad7b1cc938",
    "ggml-cpu-cannonlake.dll": "3f7a3ce226f5827d8b4774fe9a9485499e932ae74a9266466fc6c32b926aa3b1",
    "ggml-cpu-cascadelake.dll": "ac6688d2befde8037fe2cdd461171153b85c9022ed9c41d48677f538aaaf6d5d",
    "ggml-cpu-cooperlake.dll": "01c288ed18fed564fe6268d348a5a5483fedcb98290a68076fede37ea74f3dde",
    "ggml-cpu-haswell.dll": "f682c0b0672db2929e2cdf83bc954fbe897842d9a737ebbfc4135b0ae4031ea8",
    "ggml-cpu-icelake.dll": "11353cdab8a7dedbec968519adcd9d2c504a9a911beaecec5f0212da97bf40df",
    "ggml-cpu-ivybridge.dll": "adb65391c7c50f093ee36c785adfeab38b05132c5d7f86fa0f0f2e8ddb47436e",
    "ggml-cpu-piledriver.dll": "9c2ace2431c7d0a84d09deb7681cf2ad4fe5ace1cb583cb967b4de5467feb145",
    "ggml-cpu-sandybridge.dll": "7d7bc8cb0a222df061bf4b08be6aa680f29c27f2632296d10b3b8ed8472f044d",
    "ggml-cpu-sapphirerapids.dll": "4139be8692249e16ecf87001370775bd82a80cb723f40294b07e9680d96f2f2a",
    "ggml-cpu-skylakex.dll": "21a424bc5222615dd9490b3d306fc19623f7ef78610b4032af8962e3cef238fa",
    "ggml-cpu-sse42.dll": "6798bc005cfcfe13d7a5f2e96627f99779cd3d1d28222d1a3acbe76a242c90da",
    "ggml-cpu-x64.dll": "fa309588ce99bea28b5d687ad51a6489a7501ff858fa58b175020e3af0e6af1f",
    "ggml-cpu-zen4.dll": "bb03ad39d0c70a10d8d2ce7270360cbfd6983b50794b2b10b37d6a373d50a583",
    "ggml-cuda.dll": "29e7bf4dc559fa2e3cdcf853ef9fdf558d1e53eef692e76cf2008351dd8a8474",
    "ggml-rpc.dll": "77023feca15da6c475f4987eab26395a5bc079ba5b98b8f2a9ed5a148cb59837",
    "ggml-vulkan.dll": "c59df39c9a07f279fc6aab39c4f9181baee53266020295f0f07a7a145e6d5763",
    "ggml.dll": "dc579f62164e87df6f804be159348b5b0e4f199f746acc0791cb5c63910a669c",
    OPENMP_DLL: "89faf9654bdcbc72af33ef5a2d02268215f2f53ec2da183041996e1ac0c2c9f0",
    "llama.dll": "10c5b7e7d44a19daf6aff56fb7f3b83ecdee8078b058b0604050e77d002e4476",
    "mtmd.dll": "8ec940f47efdb16d4c50812acc7ab3acce5a96d4b36b3f2aac9e523a8bbf6beb",
}


def sha256_file(path: Path) -> str:
    """分块计算原生文件或下载归档的摘要。

    Args:
        path: 待校验的文件。
    Returns:
        十六进制摘要。
    """
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def native_manifest() -> dict:
    """返回原生库、公开归档及 OpenMP 替换来源的固定清单。

    Returns:
        可以直接写入发行包的来源材料。
    """
    return {"llama_tag": LLAMA_TAG, "llama_commit": LLAMA_COMMIT,
            "archives": [asdict(archive) for archive in LLAMA_ARCHIVES],
            "openmp_override": {**asdict(OPENMP_ARCHIVE), "member": OPENMP_MEMBER,
                                "destination": OPENMP_DLL, "license": "Apache-2.0 WITH LLVM-exception"},
            "dll_sha256": dict(NATIVE_DLL_SHA256)}


def download_archive(archive: NativeArchive, cache_dir: Path) -> Path:
    """缓存固定归档，拒绝使用损坏缓存或下载不匹配内容。

    Args:
        archive: 包含固定地址和摘要的资产。
        cache_dir: 可复用的下载缓存目录。
    Returns:
        已完成摘要校验的归档路径。
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / archive.filename
    if destination.exists():
        if sha256_file(destination) != archive.sha256:
            raise ValueError(f"原生资产缓存摘要不匹配：{destination}")
        return destination
    request = Request(archive.url, headers={"User-Agent": "StreamVox-native-assets/2"})
    temporary_path = None
    try:
        with NamedTemporaryFile(dir=cache_dir, suffix=".download", delete=False) as handle:
            temporary_path = Path(handle.name)
            with urlopen(request, timeout=120) as response:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
        if sha256_file(temporary_path) != archive.sha256:
            raise ValueError(f"下载的原生归档摘要不匹配：{archive.filename}")
        temporary_path.rename(destination)
        return destination
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def merge_native_archives(archive_paths: list[Path], openmp_path: Path, destination: Path) -> None:
    """只提取已批准 DLL，同名内容一致才合并，并单独替换 OpenMP。

    Args:
        archive_paths: 已校验的 llama.cpp 归档。
        openmp_path: 已校验的公开 LLVM OpenMP 归档。
        destination: 新建的原生库暂存目录。
    Returns:
        无返回值；出现冲突、缺库或摘要异常立即失败。
    """
    destination.mkdir(parents=True, exist_ok=True)
    for archive_path in archive_paths:
        with ZipFile(archive_path) as archive:
            for member in archive.infolist():
                name = PurePosixPath(member.filename).name
                if name not in NATIVE_DLL_SHA256 or name == OPENMP_DLL:
                    continue
                with archive.open(member) as handle:
                    digest = hashlib.file_digest(handle, "sha256").hexdigest()
                target = destination / name
                # 重名库必须逐字节对应同一版本，禁止后写入的后端静默覆盖公共依赖。
                if target.exists() and sha256_file(target) != digest:
                    raise ValueError(f"原生归档存在同名 DLL 冲突：{name}")
                if digest != NATIVE_DLL_SHA256[name]:
                    raise ValueError(f"原生 DLL 摘要不匹配：{name}")
                if target.exists():
                    continue
                with archive.open(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
    # 不使用归档自带的微软 OpenMP；只读取公开 LLVM 包中的固定普通文件，禁止通用路径解压。
    with tarfile.open(openmp_path, "r:bz2") as archive:
        member = archive.getmember(OPENMP_MEMBER)
        if not member.isfile():
            raise ValueError("LLVM OpenMP 资产必须是普通 DLL 文件。")
        with archive.extractfile(member) as source, (destination / OPENMP_DLL).open("xb") as output:
            shutil.copyfileobj(source, output)
    validate_native_directory(destination, require_manifest=False)
    (destination / "native-manifest.json").write_text(
        json.dumps(native_manifest(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def validate_native_directory(directory: Path, *, require_manifest: bool = True) -> None:
    """核对完整三后端 DLL 清单与公开来源材料。

    Args:
        directory: 待打包的原生库目录。
        require_manifest: 是否要求来源清单已经写入。
    Returns:
        无返回值；缺库、多余文件或内容异常立即失败。
    """
    expected = set(NATIVE_DLL_SHA256)
    if require_manifest:
        expected.add("native-manifest.json")
    if not directory.is_dir() or {path.name for path in directory.iterdir()} != expected:
        raise ValueError("原生库目录必须精确包含已批准的三后端 DLL 与来源清单。")
    for name, digest in NATIVE_DLL_SHA256.items():
        if sha256_file(directory / name) != digest:
            raise ValueError(f"原生 DLL 摘要不匹配：{name}")
    if require_manifest:
        actual = json.loads((directory / "native-manifest.json").read_text(encoding="utf-8"))
        if actual != native_manifest():
            raise ValueError("原生库来源清单与当前固定资产不一致。")


def prepare_native_assets(cache_dir: Path, destination: Path) -> Path:
    """在独立暂存目录完成合并与审计后装配原生库。

    Args:
        cache_dir: 固定归档下载缓存。
        destination: 项目或发行暂存目录中的原生库目录。
    Returns:
        已完成准备的原生库目录。
    """
    archives = [download_archive(archive, cache_dir) for archive in LLAMA_ARCHIVES]
    openmp = download_archive(OPENMP_ARCHIVE, cache_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".native-", dir=destination.parent) as work:
        merged = Path(work) / "bin"
        merge_native_archives(archives, openmp, merged)
        if destination.exists():
            unknown = set(path.name for path in destination.iterdir()) - set(NATIVE_DLL_SHA256) - {"native-manifest.json"}
            if unknown:
                raise ValueError(f"原生库目标目录含未知文件：{sorted(unknown)}")
            # 只替换明确批准的发行文件，保留目录本身，避免递归清理用户工作区。
            for source in merged.iterdir():
                source.replace(destination / source.name)
        else:
            merged.rename(destination)
    validate_native_directory(destination)
    return destination


def main() -> None:
    """准备公开来源的 Windows 三后端 DLL。"""
    parser = argparse.ArgumentParser(description="准备固定公开来源的 StreamVox 原生库。")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    cache_dir = args.cache_dir or project_root / ".artifacts/native" / LLAMA_TAG
    if not cache_dir.is_absolute():
        cache_dir = project_root / cache_dir
    print(prepare_native_assets(cache_dir, project_root / "streamvox/bin"))


if __name__ == "__main__":
    main()
