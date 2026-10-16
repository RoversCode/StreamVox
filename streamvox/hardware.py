from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
import platform
import shutil
import subprocess

from .utils.log import logger

# 硬件资源探测与「当前模型档位」兼容性评估。
# 典型调用链：assess_and_log_hardware → probe_system_resources → assess_model_hardware
# → log_hardware_compatibility；探测实现分散在 _probe_* 私有函数中。


@dataclass(frozen=True, slots=True)
class GPUDeviceInfo:
    """单块显卡的探测结果，供显存门槛比对与初始化日志使用。"""

    index: int  # 与 nvidia-smi 的 GPU 序号一致；Windows display 兜底时自增编号
    name: str  # 驱动/系统报告的显卡名称
    backend: str  # 探测路径：cuda（nvidia-smi）或 display（Windows wmic 兜底）
    total_vram_gb: float  # 该卡总显存（GB）；多卡时由 SystemResourceSnapshot.max_vram_gb 取最大


@dataclass(frozen=True, slots=True)
class SystemResourceSnapshot:
    """一次 probe 得到的整机资源快照，作为 assess_model_hardware 的输入。"""

    os_name: str  # 小写平台名，便于日志与后续按系统分支扩展
    machine: str  # 架构字符串（如 x86_64、arm64）
    total_ram_gb: float  # 物理内存总量：与模型 min_ram / recommended_ram 比对
    # Linux 上优先用 MemAvailable（含可回收 cache），避免 sysconf(SC_AVPHYS_PAGES) 严重低估
    available_ram_gb: float  # 当前可用内存：仅用于日志参考，门槛判断用 total 更稳定
    gpus: tuple[GPUDeviceInfo, ...] = ()  # 空元组表示未识别到 GPU，不等同于「一定是 CPU 推理」

    @property
    def max_vram_gb(self) -> float:
        # 多卡时取最大显存，与「单卡跑大模型」的常见假设一致；无卡时返回 0
        if not self.gpus:
            return 0.0
        return max(gpu.total_vram_gb for gpu in self.gpus)


@dataclass(frozen=True, slots=True)
class ModelFootprintProfile:
    """
    当前模型的硬件要求描述。

    这里不再做“全局模型档位表”推断，只保存当前模型自己声明的最低/推荐资源要求。
    """

    profile_id: str  # 档位标识，日志里与模型名一起展示
    parameter_billion: float  # 参数量（B），仅作人类可读信息
    quantization: str  # 量化描述（如 mixed q5/q8），不参与数值比较
    min_vram_gb: float  # 最低显存（GB）；0 表示不在 assess 里检查显存下限
    recommended_vram_gb: float  # 推荐显存（GB）；0 表示不检查推荐显存
    min_ram_gb: float  # 最低物理内存总量（GB）
    recommended_ram_gb: float  # 推荐物理内存总量（GB）
    notes: str = ""  # 给人看的说明，不参与逻辑


@dataclass(frozen=True, slots=True)
class HardwareCompatibilityReport:
    """assess_model_hardware 的输出：原始快照 + 模型要求 + 是否满足最低/推荐档。"""

    snapshot: SystemResourceSnapshot  # 当时探测到的机器资源
    requirement: ModelFootprintProfile  # 当前模型声明的档位要求
    meets_minimum: bool  # 显存与内存均满足「最低」门槛时为 True
    meets_recommended: bool  # 显存与内存均满足「推荐」门槛时为 True


def _has_cuda_vram_probe(snapshot: SystemResourceSnapshot) -> bool:
    """是否至少有一块卡来自 nvidia-smi（当前唯一视为「显存数值较可信」的路径）。"""
    return any(gpu.backend == "cuda" for gpu in snapshot.gpus)


def _log_non_nvidia_vram_disclaimer(snapshot: SystemResourceSnapshot) -> None:
    """
    对 AMD / Intel 等用户说明：本模块未接 rocm-smi、Intel 工具等，避免误读告警。

    何时打印：没有 cuda 探针结果时（Linux 上常见为仅有 A/I 卡；Windows 上可能为 WMI 兜底或完全未枚举到）。
    """
    if _has_cuda_vram_probe(snapshot):
        return
    logger.info(
        "[Hardware Check] 说明：当前检测未通过 NVIDIA 驱动接口读取显存。"
        "若您使用 AMD 或 Intel 显卡，本次日志中的显存数值及「是否满足显存门槛」的结论均无参考价值，请直接忽略；"
        "是否可运行请以实际推理效果或厂商工具（如 rocm-smi、GPU-Z 等）为准。"
    )


def probe_system_resources() -> SystemResourceSnapshot:
    """探测当前机器的基础资源，用于初始化阶段做兼容性提示。"""
    # 内存走 OS API；GPU 列表尽力探测，失败则 gpus 为空（由上层日志说明）
    total_ram_bytes, available_ram_bytes = _probe_ram_bytes()
    return SystemResourceSnapshot(
        os_name=platform.system().lower(),
        machine=platform.machine().lower(),
        total_ram_gb=_bytes_to_gb(total_ram_bytes),
        available_ram_gb=_bytes_to_gb(available_ram_bytes),
        gpus=tuple(_probe_gpus()),
    )


def assess_model_hardware(
    snapshot: SystemResourceSnapshot,
    requirement: ModelFootprintProfile,
) -> HardwareCompatibilityReport:
    """
    只对比“当前要实例化的目标模型”和“当前机器资源”。

    为什么不再做全局推荐表：
    - streamvox 在接入模型时，理论上应当明确知道该模型自己的最低/推荐资源要求；
    - 初始化阶段真正重要的是：这台机器能不能跑当前模型，而不是它还能跑哪些别的模型。
    """
    # min_vram_gb 为 0 表示「不显式要求独显显存」（例如纯 CPU 路径），跳过 VRAM 检查
    min_vram_ok = snapshot.max_vram_gb >= requirement.min_vram_gb if requirement.min_vram_gb > 0 else True
    min_ram_ok = snapshot.total_ram_gb >= requirement.min_ram_gb
    # 推荐显存为 0 时同样视为不检查该项，避免无意义告警
    recommended_vram_ok = (
        snapshot.max_vram_gb >= requirement.recommended_vram_gb
        if requirement.recommended_vram_gb > 0
        else True
    )
    recommended_ram_ok = snapshot.total_ram_gb >= requirement.recommended_ram_gb
    return HardwareCompatibilityReport(
        snapshot=snapshot,
        requirement=requirement,
        meets_minimum=min_vram_ok and min_ram_ok,
        meets_recommended=recommended_vram_ok and recommended_ram_ok,
    )


def log_hardware_compatibility(
    report: HardwareCompatibilityReport,
    *,
    active_model_name: str,
) -> None:
    """输出当前设备与当前模型要求的比对结果。"""
    snapshot = report.snapshot
    requirement = report.requirement
    # 先打系统与内存概况，再逐卡打 GPU，最后汇总模型档位与是否达标，方便从日志一眼定位问题
    logger.info(
        "[Hardware Check] 系统=%s, 架构=%s, 内存=%.1fGB, 可用内存=%.1fGB",
        snapshot.os_name,
        snapshot.machine,
        snapshot.total_ram_gb,
        snapshot.available_ram_gb,
    )

    if snapshot.gpus:
        # 多卡时每条一行，与 max_vram_gb「取最大」的评估策略对照阅读
        for gpu in snapshot.gpus:
            logger.info(
                "[Hardware Check] GPU[%s]=%s, backend=%s, 显存=%.1fGB",
                gpu.index,
                gpu.name,
                gpu.backend,
                gpu.total_vram_gb,
            )
    else:
        logger.warning("[Hardware Check] 未检测到可识别显卡，将按 CPU/内存条件评估当前模型")

    # 无 nvidia-smi 成功路径时单独说明，避免 A/I 用户把「显存不足」类告警当成真实结论
    _log_non_nvidia_vram_disclaimer(snapshot)

    logger.info(
        "[Hardware Check] 当前模型=%s, 要求档位=%s, 参数量=%.1fB, 量化=%s, 最低显存=%.1fGB, 推荐显存=%.1fGB, 最低内存=%.1fGB, 推荐内存=%.1fGB",
        active_model_name,
        requirement.profile_id,
        requirement.parameter_billion,
        requirement.quantization,
        requirement.min_vram_gb,
        requirement.recommended_vram_gb,
        requirement.min_ram_gb,
        requirement.recommended_ram_gb,
    )

    # 分三档输出：推荐达标 > 仅最低达标 > 最低不达标（只 return 是为了避免重复打多条总结）
    if report.meets_recommended:
        logger.info("[Hardware Check] 当前硬件满足 %s 的推荐配置，可流畅运行该模型", active_model_name)
        return
    if report.meets_minimum:
        logger.warning("[Hardware Check] 当前硬件仅满足 %s 的最低配置，可能可以运行但不保证流畅", active_model_name)
        return
    logger.warning("[Hardware Check] 当前硬件不满足 %s 的最低配置，运行可能失败或非常缓慢", active_model_name)


def assess_and_log_hardware(
    *,
    active_model_name: str,
    requirement: ModelFootprintProfile,
) -> HardwareCompatibilityReport:
    """
    引擎/模型初始化时的「一站式」硬件评估：探测 → 对比档位 → 写日志 → 返回报告。

    为什么拆成三步再包一层：
    - probe / assess / log 可单独单测或复用（例如只探测不打日志）；
    - 本函数保证调用方一次调用即可完成初始化阶段的友好提示，并把结构化结果交给上层（如需 UI 或遥测）。
    """
    # 1) 采集本机内存与 GPU 列表
    snapshot = probe_system_resources()
    # 2) 与当前模型的 ModelFootprintProfile 做门槛对比，得到结构化结论
    report = assess_model_hardware(snapshot, requirement)
    # 3) 把结论写入日志（推荐/最低/不足三档），便于运维与用户自查
    log_hardware_compatibility(report, active_model_name=active_model_name)
    return report


def _probe_gpus() -> list[GPUDeviceInfo]:
    # Linux/macOS 等常见路径：有 nvidia-smi 则优先拿到准确显存；Windows 下若仍无 NVIDIA 信息再尝试显示适配器
    gpus = _probe_nvidia_gpus()
    if gpus:
        return gpus
    if platform.system().lower().startswith("win"):
        return _probe_windows_display_gpus()
    return []


def _probe_nvidia_gpus() -> list[GPUDeviceInfo]:
    # 依赖系统 PATH 中的 nvidia-smi；不存在则静默返回空列表，由上层按「无 GPU 信息」处理
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        # 超时、权限或驱动异常时不阻断初始化，仅视为探测失败
        return []

    gpus: list[GPUDeviceInfo] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            index = int(parts[0])
            # nvidia-smi 的 memory.total 通常为 MiB；÷1024 得到 GiB，与常见「GB」显存标称口径一致
            total_vram_gb = float(parts[2]) / 1024.0
        except ValueError:
            continue
        gpus.append(
            GPUDeviceInfo(
                index=index,
                name=parts[1],
                backend="cuda",
                total_vram_gb=round(total_vram_gb, 1),
            )
        )
    return gpus


def _probe_windows_display_gpus() -> list[GPUDeviceInfo]:
    """
    Windows 下优先给出一个尽力而为的显卡列表。

    这里不用 PowerShell 脚本化复杂逻辑，避免初始化阶段引入额外环境依赖；
    只在拿不到 NVIDIA 信息时，尝试用 wmic 补一个通用兜底。
    """
    if shutil.which("wmic") is None:
        return []
    try:
        result = subprocess.run(
            ["wmic", "path", "win32_VideoController", "get", "Name,AdapterRAM", "/format:csv"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        # wmic 在新系统上可能不可用或超时，失败则返回空，不阻塞启动
        return []

    gpus: list[GPUDeviceInfo] = []
    next_index = 0
    for line in result.stdout.splitlines():
        if not line or "AdapterRAM" in line:
            continue
        parts = [part.strip() for part in line.split(",") if part.strip()]
        if len(parts) < 2:
            continue
        # CSV 列顺序依赖 wmic 输出：最后一列为 AdapterRAM，倒数第二列为 Name
        name = parts[-2]
        ram_text = parts[-1]
        try:
            total_vram_gb = round(int(ram_text) / (1024 ** 3), 1)
        except ValueError:
            continue
        gpus.append(
            GPUDeviceInfo(
                index=next_index,
                name=name,
                backend="display",
                total_vram_gb=total_vram_gb,
            )
        )
        next_index += 1
    return gpus


def _probe_ram_bytes() -> tuple[int, int]:
    # Windows 用 GlobalMemoryStatusEx；Linux 用 /proc/meminfo（与 free 命令口径一致）
    if os.name == "nt":
        return _probe_ram_bytes_windows()
    if platform.system() == "Linux":
        return _probe_ram_bytes_linux()
    return _probe_ram_bytes_posix()


def _linux_meminfo_bytes_from_text(content: str) -> tuple[int, int] | None:
    """
    解析 /proc/meminfo 文本，返回 (总物理内存字节, 可用内存估计字节)。

    为什么不用 sysconf(SC_AVPHYS_PAGES) 做 Linux 可用内存：
    - SC_AVPHYS_PAGES 往往接近「完全空闲页」，不把可回收的 page cache 算进去；
    - 大内存机器在缓存吃满时会出现「总量几百 GB、可用却只有几 GB」的假象，和 free/top 完全对不上。
    内核 3.14+ 的 MemAvailable 专门解决该问题；老内核回退到 MemFree+Buffers+Cached 的粗估。
    """
    mem_total_kib: int | None = None
    mem_available_kib: int | None = None
    mem_free_kib: int | None = None
    buffers_kib: int | None = None
    cached_kib: int | None = None

    for line in content.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        key, value = parts[0], parts[1]
        try:
            kib = int(value)
        except ValueError:
            continue
        if key == "MemTotal:":
            mem_total_kib = kib
        elif key == "MemAvailable:":
            mem_available_kib = kib
        elif key == "MemFree:":
            mem_free_kib = kib
        elif key == "Buffers:":
            buffers_kib = kib
        elif key == "Cached:":
            cached_kib = kib

    if mem_total_kib is None:
        return None

    if mem_available_kib is not None:
        avail_kib = mem_available_kib
    else:
        # 无 MemAvailable 时的粗估（略乐观，但比 SC_AVPHYS_PAGES 更符合用户直觉）
        mf = mem_free_kib or 0
        bc = (buffers_kib or 0) + (cached_kib or 0)
        avail_kib = mf + bc

    kib_to_bytes = 1024
    return mem_total_kib * kib_to_bytes, avail_kib * kib_to_bytes


def _probe_ram_bytes_linux() -> tuple[int, int]:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            parsed = _linux_meminfo_bytes_from_text(f.read())
    except OSError:
        parsed = None
    if parsed is not None:
        return parsed
    return _probe_ram_bytes_posix()


def _probe_ram_bytes_posix() -> tuple[int, int]:
    # 非 Linux POSIX：总量用 SC_PHYS_PAGES；可用量仍用 SC_AVPHYS_PAGES（macOS 等无统一 meminfo）
    page_size = os.sysconf("SC_PAGE_SIZE")
    total_pages = os.sysconf("SC_PHYS_PAGES")
    avail_pages = os.sysconf("SC_AVPHYS_PAGES")
    return page_size * total_pages, page_size * avail_pages


def _probe_ram_bytes_windows() -> tuple[int, int]:
    # MEMORYSTATUSEX 结构体必须与 MSDN 定义一致；dwLength 必须先设为结构体大小，否则 API 调用失败
    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return int(status.ullTotalPhys), int(status.ullAvailPhys)


def _bytes_to_gb(value: int) -> float:
    # 使用 1024^3 与常见「GB」展示一致；保留一位小数避免日志过于冗长
    return round(value / (1024 ** 3), 1)


__all__ = [
    "GPUDeviceInfo",
    "SystemResourceSnapshot",
    "ModelFootprintProfile",
    "HardwareCompatibilityReport",
    "probe_system_resources",
    "assess_model_hardware",
    "log_hardware_compatibility",
    "assess_and_log_hardware",
]
