from __future__ import annotations

import os
from dataclasses import dataclass

from .utils.log import logger


@dataclass(frozen=True, slots=True)
class OpenMPRuntimePolicy:
    """
    中文说明：
        描述当前进程对 OpenMP 运行时冲突的兼容策略。

    核心职责：
        1. 统一管理 StreamVox 在 Windows 下与第三方原生库共存时的 OpenMP 环境变量；
        2. 把“是否由 SDK 自动注入兼容变量”显式暴露出来，便于日志和单元测试校验；
        3. 避免兼容策略散落在 engine、worker、模型 runtime 等多个入口中。

    核心入参意义：
        无。该数据结构只承载策略结果。

    预期输出：
        包含最终环境变量值，以及本次是否由 StreamVox 自动注入。

    可能的异常抛出：
        本数据结构本身不主动抛出异常。
    """

    duplicate_lib_ok: str | None
    injected_by_streamvox: bool


def ensure_openmp_runtime_compat(*, source: str) -> OpenMPRuntimePolicy:
    """
    中文说明：
        在进程最早期统一注入 OpenMP 兼容环境，规避 Windows 下
        `libomp140.x86_64.dll` 与 `libiomp5md.dll` 双运行时并存导致的
        `OMP: Error #15` 进程级崩溃。

    核心职责：
        1. 把兼容策略收敛到单一入口，避免不同模块各自偷偷写环境变量；
        2. 仅在高风险平台（Windows）注入最小必要配置；
        3. 如果用户已经显式配置环境变量，则严格尊重用户现场，不覆盖其决定。

    核心入参意义：
        source: 当前调用来源，用于日志定位到底是包入口、主进程还是 worker 注入了策略。

    预期输出：
        返回本次生效的 OpenMP 兼容策略结果。

    可能的异常抛出：
        本函数不主动抛出异常；若日志模块异常，也应尽量避免影响主流程。
    """
    existing = os.environ.get("KMP_DUPLICATE_LIB_OK")
    injected = False

    # 业务原因：
    # Windows 打包组合里，llama.cpp backend 与 onnxruntime-directml 可能各自携带不同 OpenMP runtime。
    # 这类冲突不是 Python 层能彻底消除的链接问题，因此这里做“进程级兜底”，保证 SDK 至少能继续执行。
    # 若用户已经显式配置该变量，则必须尊重用户现场，避免覆盖外部部署策略。
    if os.name == "nt" and not existing:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        existing = "TRUE"
        injected = True
        # logger.warning(
        #     "[Runtime Compat] 检测到 Windows 原生推理栈可能存在 OpenMP 运行时冲突，"
        #     "已自动注入 KMP_DUPLICATE_LIB_OK=TRUE 以避免 OMP Error #15；"
        #     "这属于兼容兜底策略，如需更严格控制可在进程启动前自行覆盖该环境变量。"
        # )
    elif existing:
        logger.debug(
            "[Runtime Compat] 复用外部已提供的 KMP_DUPLICATE_LIB_OK=%s (source=%s)",
            existing,
            source,
        )

    return OpenMPRuntimePolicy(
        duplicate_lib_ok=existing,
        injected_by_streamvox=injected,
    )
