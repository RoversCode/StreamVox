"""防止 S2 编码器和流式解码器重新启用非确定性的快速归约。"""

from streamvox.models.s2_pro.codec_common import create_session_options


def test_codec_uses_deterministic_reductions():
    """参考音频的码本和固定缓存解码必须能重复计算。"""
    assert create_session_options().use_deterministic_compute is True
