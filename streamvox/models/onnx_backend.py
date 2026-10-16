"""统一处理调用方明确选择的 ONNX 后端。"""

from __future__ import annotations

from .base import ExecutionTarget


def resolve_explicit_onnx_backend(target: ExecutionTarget) -> tuple[str, int | None] | None:
    """显式后端必须可用，未显式选择时交回模型原有默认策略。

    Args:
        target: 当前模型的 ONNX 执行目标。
    Returns:
        已确认的后端与设备索引；没有显式选择时返回 None。
    """
    if target.backend_preference is None:
        return None
    import onnxruntime as ort
    from streamvox_runtime import ensure_single_onnxruntime_distribution

    ensure_single_onnxruntime_distribution()
    providers = {"directml": "DmlExecutionProvider", "cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider"}
    requested = target.backend_preference
    if requested not in providers:
        raise ValueError(f"未知 ONNX 后端：{requested!r}")
    if providers[requested] not in ort.get_available_providers():
        raise RuntimeError(f"显式请求的 {providers[requested]} 不可用")
    if requested == "cpu":
        return "cpu", None
    return ("dml" if requested == "directml" else requested), target.device_index or 0


def verify_session_provider(session: object, providers: list) -> None:
    """禁止 ONNX 在初始化失败后悄悄用别的后端继续。

    Args:
        session: 刚创建的 ONNX 会话。
        providers: 模型明确要求的提供程序顺序。
    Returns:
        主后端吻合时返回，否则抛出 RuntimeError。
    """
    if not providers:
        return
    requested = providers[0][0] if isinstance(providers[0], tuple) else providers[0]
    actual = session.get_providers()
    if not actual or actual[0] != requested:
        raise RuntimeError(f"ONNX 后端初始化发生回退：请求 {requested}，实际 {actual}")
    session.disable_fallback()
