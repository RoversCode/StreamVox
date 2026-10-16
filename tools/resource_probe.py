from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ResourceSample:
    """
    单次资源采样快照。

    参数:
        timestamp: 采样发生时的单调时钟时间，便于后续计算相对耗时。
        label: 当前业务阶段标签，用于把峰值归因到初始化、构建 prompt 或合成阶段。
        rss_mib: 当前主进程及其子进程的常驻内存总量，单位为 MiB。
        gpu_total_used_mib: 指定物理 GPU 的整卡已用显存，单位为 MiB。
        gpu_total_delta_mib: 相对基线的整卡显存增量，单位为 MiB。
        gpu_process_used_mib: 当前进程树在指定物理 GPU 上的显存占用，单位为 MiB。

    返回:
        数据类仅承载采样结果，不主动执行计算。

    异常:
        本数据类不抛出异常；采样失败由 ResourceMonitor 降级为 None 或 0。
    """

    timestamp: float
    label: str
    rss_mib: float
    gpu_total_used_mib: int | None
    gpu_total_delta_mib: int | None
    gpu_process_used_mib: int


@dataclass(slots=True)
class PhasePeak:
    """
    单个业务阶段的资源峰值汇总。

    参数:
        label: 业务阶段名称。
        peak_rss_mib: 阶段内主进程树的最大常驻内存。
        peak_gpu_total_delta_mib: 阶段内整卡显存相对基线的最大增量。
        peak_gpu_process_used_mib: 阶段内当前进程树的最大显存占用。

    返回:
        数据类仅用于结构化输出，最终由 benchmark 脚本序列化为 JSON。

    异常:
        本数据类不执行外部命令，不会主动抛出异常。
    """

    label: str
    peak_rss_mib: float
    peak_gpu_total_delta_mib: int | None
    peak_gpu_process_used_mib: int


@dataclass(slots=True)
class ResourceSummary:
    """
    一次完整模型测试的资源摘要。

    参数:
        baseline_gpu_used_mib: 测试开始前指定物理 GPU 的整卡已用显存。
        peak_rss_mib: 全流程主进程树的最大常驻内存。
        peak_gpu_total_delta_mib: 全流程整卡显存相对基线的最大增量。
        peak_gpu_process_used_mib: 全流程当前进程树的最大显存占用。
        phase_peaks: 按业务阶段聚合的峰值列表。

    返回:
        数据类用于把采样线程收集到的数据交给调用方。

    异常:
        本数据类不主动抛出异常。
    """

    baseline_gpu_used_mib: int | None
    peak_rss_mib: float
    peak_gpu_total_delta_mib: int | None
    peak_gpu_process_used_mib: int
    phase_peaks: list[PhasePeak] = field(default_factory=list)


class ResourceMonitor:
    """
    进程树资源监控器。

    参数:
        physical_gpu_index: nvidia-smi 视角下的物理 GPU 编号；本任务固定为 1。
        root_pid: 被测进程的根进程号，默认是当前 Python 进程。
        interval_seconds: 后台采样间隔；间隔越短越容易捕获瞬时峰值，但 nvidia-smi 调用开销也越高。

    返回:
        调用 stop 后返回 ResourceSummary，包含内存和显存的全流程峰值与分阶段峰值。

    异常:
        nvidia-smi 不存在或瞬时查询失败时不会中断测试，显存字段会降级为 None 或 0。
    """

    def __init__(
        self,
        *,
        physical_gpu_index: int,
        root_pid: int | None = None,
        interval_seconds: float = 0.2,
    ) -> None:
        # 被测进程根节点，后续会递归收集它拉起的 decoder worker 等子进程。
        self.root_pid = root_pid if root_pid is not None else os.getpid()
        # 显存采样使用物理卡编号，避免 CUDA_VISIBLE_DEVICES 重映射后报告口径混乱。
        self.physical_gpu_index = physical_gpu_index
        # 采样间隔保留为实例字段，方便不同测试任务按成本调整。
        self.interval_seconds = interval_seconds
        # 当前阶段标签由 benchmark 脚本在关键业务步骤前更新。
        self._label = "startup"
        # 后台线程停止信号，确保测试结束后可以干净退出。
        self._stop_event = threading.Event()
        # 采样结果只在 monitor 内部追加，最终统一聚合，避免调用方在热路径做统计。
        self._samples: list[ResourceSample] = []
        # 线程对象延迟创建，避免构造 monitor 就开始执行外部命令。
        self._thread: threading.Thread | None = None
        # 测试开始前的整卡显存基线，用于从整卡占用里扣掉已有后台进程。
        self.baseline_gpu_used_mib = self._query_gpu_total_used_mib()

    def start(self) -> None:
        """
        启动后台采样线程。

        参数:
            无。

        返回:
            无返回值；采样数据保存在实例内部。

        异常:
            重复启动会被忽略，避免调用方在阶段切换时误创建多个采样线程。
        """
        # 已经启动过时直接返回，保证监控线程在一次测试中只有一个写入者。
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="streamvox-resource-monitor", daemon=True)
        self._thread.start()

    def set_label(self, label: str) -> None:
        """
        更新当前业务阶段标签。

        参数:
            label: 后续采样需要归属的阶段名称。

        返回:
            无返回值。

        异常:
            空字符串会被替换为 unknown，避免输出阶段名不可读。
        """
        # 阶段标签用于结果可读性，不参与控制流程，因此这里做温和归一化。
        self._label = label.strip() if label.strip() else "unknown"

    def stop(self) -> ResourceSummary:
        """
        停止后台采样并聚合结果。

        参数:
            无。

        返回:
            返回 ResourceSummary，包含全流程和分阶段峰值。

        异常:
            即使从未 start，也会执行一次同步采样后返回摘要，方便失败路径仍能输出资源信息。
        """
        # 结束前补一帧，捕获 shutdown 前后可能残留的峰值状态。
        self._record_sample()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 4))
        return self._build_summary()

    def _run(self) -> None:
        """
        后台采样循环。

        参数:
            无。

        返回:
            无返回值；循环结束条件由 stop 触发。

        异常:
            单次采样异常被捕获在底层查询函数中，避免资源监控影响模型测试。
        """
        # 以固定间隔采样，业务线程无需等待 nvidia-smi 返回。
        while not self._stop_event.is_set():
            self._record_sample()
            self._stop_event.wait(self.interval_seconds)

    def _record_sample(self) -> None:
        """
        采集并保存一帧资源数据。

        参数:
            无。

        返回:
            无返回值。

        异常:
            外部命令失败时使用降级值继续记录，保证测试主流程不被监控工具打断。
        """
        # 当前进程树可能包含模型解码 worker，必须每次动态重新发现子进程。
        process_ids = self._collect_process_tree_pids()
        # RSS 直接从 procfs 读取，避免引入 psutil 这类额外测试依赖。
        rss_mib = self._sum_rss_mib(process_ids)
        # 整卡显存增量用于兜底覆盖 Vulkan 等 nvidia-smi 进程列表不稳定的场景。
        gpu_total_used_mib = self._query_gpu_total_used_mib()
        gpu_total_delta_mib = None
        if gpu_total_used_mib is not None and self.baseline_gpu_used_mib is not None:
            gpu_total_delta_mib = max(0, gpu_total_used_mib - self.baseline_gpu_used_mib)
        # 进程显存用于更精确地归因到当前测试进程树。
        gpu_process_used_mib = self._query_gpu_process_used_mib(process_ids)
        self._samples.append(
            ResourceSample(
                timestamp=time.monotonic(),
                label=self._label,
                rss_mib=rss_mib,
                gpu_total_used_mib=gpu_total_used_mib,
                gpu_total_delta_mib=gpu_total_delta_mib,
                gpu_process_used_mib=gpu_process_used_mib,
            )
        )

    def _collect_process_tree_pids(self) -> set[int]:
        """
        收集当前根进程及所有子孙进程号。

        参数:
            无。

        返回:
            返回进程号集合，至少包含 root_pid。

        异常:
            procfs 瞬时读取失败时跳过对应进程，避免子进程退出导致测试失败。
        """
        # 先构建 ppid 到子进程的索引，后续从 root_pid 广度遍历即可覆盖 worker 进程。
        children_by_parent: dict[int, list[int]] = {}
        for stat_path in Path("/proc").glob("[0-9]*/stat"):
            try:
                stat_text = stat_path.read_text(encoding="utf-8", errors="ignore")
                close_paren_index = stat_text.rfind(")")
                fields_after_name = stat_text[close_paren_index + 2 :].split()
                pid = int(stat_path.parent.name)
                parent_pid = int(fields_after_name[1])
            except (OSError, ValueError, IndexError):
                continue
            children_by_parent.setdefault(parent_pid, []).append(pid)

        # 从根进程递归展开，保证 decoder 子进程的 CPU 内存也被计入。
        collected = {self.root_pid}
        pending = [self.root_pid]
        while pending:
            current_pid = pending.pop()
            for child_pid in children_by_parent.get(current_pid, []):
                if child_pid in collected:
                    continue
                collected.add(child_pid)
                pending.append(child_pid)
        return collected

    def _sum_rss_mib(self, process_ids: set[int]) -> float:
        """
        汇总进程树 RSS。

        参数:
            process_ids: 需要纳入统计的进程号集合。

        返回:
            返回 RSS 总量，单位为 MiB。

        异常:
            单个进程退出或状态文件不可读时跳过该进程。
        """
        total_kib = 0
        for pid in process_ids:
            status_path = Path("/proc") / str(pid) / "status"
            try:
                for line in status_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if line.startswith("VmRSS:"):
                        total_kib += int(line.split()[1])
                        break
            except (OSError, ValueError, IndexError):
                continue
        return total_kib / 1024.0

    def _query_gpu_total_used_mib(self) -> int | None:
        """
        查询指定物理 GPU 的整卡已用显存。

        参数:
            无。

        返回:
            成功时返回 MiB 整数；失败时返回 None。

        异常:
            nvidia-smi 不存在、GPU 编号无效或命令失败时统一降级为 None。
        """
        command = [
            "nvidia-smi",
            f"--id={self.physical_gpu_index}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ]
        try:
            output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL, timeout=2.0)
            return int(output.strip().splitlines()[0])
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return None

    def _query_gpu_process_used_mib(self, process_ids: set[int]) -> int:
        """
        查询当前进程树在指定物理 GPU 上的显存占用。

        参数:
            process_ids: 当前根进程及子进程号集合。

        返回:
            返回进程树显存占用总和，单位为 MiB；查询失败时返回 0。

        异常:
            nvidia-smi 查询失败时不抛出异常，避免资源探针影响模型执行。
        """
        command = [
            "nvidia-smi",
            f"--id={self.physical_gpu_index}",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ]
        try:
            output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL, timeout=2.0)
        except (OSError, subprocess.SubprocessError):
            return 0

        total_mib = 0
        for raw_line in output.splitlines():
            # 空行通常表示当前 GPU 没有 compute 进程，直接跳过即可。
            if not raw_line.strip():
                continue
            try:
                pid_text, memory_text = [part.strip() for part in raw_line.split(",", 1)]
                if int(pid_text) in process_ids:
                    total_mib += int(memory_text.split()[0])
            except (ValueError, IndexError):
                continue
        return total_mib

    def _build_summary(self) -> ResourceSummary:
        """
        聚合所有采样帧。

        参数:
            无。

        返回:
            返回全流程和分阶段峰值。

        异常:
            没有采样帧时返回零值摘要，方便失败路径仍能被 JSON 序列化。
        """
        if not self._samples:
            return ResourceSummary(
                baseline_gpu_used_mib=self.baseline_gpu_used_mib,
                peak_rss_mib=0.0,
                peak_gpu_total_delta_mib=None,
                peak_gpu_process_used_mib=0,
                phase_peaks=[],
            )

        peak_gpu_total_delta_mib = self._max_optional(sample.gpu_total_delta_mib for sample in self._samples)
        phase_peaks: list[PhasePeak] = []
        labels = list(dict.fromkeys(sample.label for sample in self._samples))
        for label in labels:
            label_samples = [sample for sample in self._samples if sample.label == label]
            phase_peaks.append(
                PhasePeak(
                    label=label,
                    peak_rss_mib=max(sample.rss_mib for sample in label_samples),
                    peak_gpu_total_delta_mib=self._max_optional(
                        sample.gpu_total_delta_mib for sample in label_samples
                    ),
                    peak_gpu_process_used_mib=max(sample.gpu_process_used_mib for sample in label_samples),
                )
            )

        return ResourceSummary(
            baseline_gpu_used_mib=self.baseline_gpu_used_mib,
            peak_rss_mib=max(sample.rss_mib for sample in self._samples),
            peak_gpu_total_delta_mib=peak_gpu_total_delta_mib,
            peak_gpu_process_used_mib=max(sample.gpu_process_used_mib for sample in self._samples),
            phase_peaks=phase_peaks,
        )

    def _max_optional(self, values: object) -> int | None:
        """
        计算可空整数序列最大值。

        参数:
            values: 可能包含 None 的整数可迭代对象。

        返回:
            存在有效整数时返回最大值；全部为空时返回 None。

        异常:
            本函数只处理可迭代输入，不主动抛出业务异常。
        """
        # 过滤 None 后再求最大值，避免显存采样不可用时污染统计。
        normalized_values = [value for value in values if value is not None]
        if not normalized_values:
            return None
        return max(normalized_values)
