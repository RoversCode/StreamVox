from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np


@dataclass
class DecodeRequest:
    """主进程发给 S2-Pro decoder worker 的消息。"""

    task_id: Union[str, int]
    msg_type: str = "DECODE_CHUNK"
    codes: Optional[np.ndarray] = None
    is_final: bool = False
    decoder_state: Optional[dict[str, np.ndarray]] = None
    decoder_tail_codes: Optional[np.ndarray] = None


@dataclass
class DecoderResponse:
    """S2-Pro decoder worker 返回给主进程的消息。"""

    task_id: Union[str, int]
    msg_type: str = "AUDIO"
    index: int = 0
    audio: Optional[np.ndarray] = None
    decoder_state: Optional[dict[str, np.ndarray]] = None
    decoder_tail_codes: Optional[np.ndarray] = None
    compute_time: float = 0.0
    recv_time: float = 0.0


@dataclass
class DecodeResult:
    responses: list[DecoderResponse] = field(default_factory=list)

    @property
    def audio(self) -> np.ndarray | None:
        pcms = [item.audio for item in self.responses if item.audio is not None]
        return np.concatenate(pcms) if pcms else None
