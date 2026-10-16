"""集中注册 SDK 内置模型，让包入口保持简洁。"""

from .moss_tts_nano import MossTTSNanoOnnxAdapter
from .qwen3 import Qwen3TTSClone06BAdapter, Qwen3TTSClone17BAdapter
from .registry import default_registry
from .s2_pro import S2Pro4BGGUFAdapter
from .voxcpm2 import VoxCPM2GGUFAdapter


def register_default_models() -> None:
    """把 SDK 提供的模型登记到共享目录，重复调用不会重复注册。

    Returns:
        无返回值；登记后的模型通过共享目录查询。
    """
    # 同一 qwen3 clone GGUF 系列按模型大小注册多个 adapter，runtime 仍共享一套实现。
    for adapter in (
        Qwen3TTSClone06BAdapter(), Qwen3TTSClone17BAdapter(), S2Pro4BGGUFAdapter(),
        VoxCPM2GGUFAdapter(), MossTTSNanoOnnxAdapter(),
    ):
        if adapter.model_name not in default_registry.list_models():
            default_registry.register(adapter)
