"""混合运行时里小型胶水算子的轻量实现。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np




def _silu(value: np.ndarray) -> np.ndarray:
    """执行 SiLU 激活，避免为一个点算子单独起 PyTorch runtime。"""

    value = np.asarray(value, dtype=np.float32)
    return (value / (1.0 + np.exp(-value))).astype(np.float32, copy=False)


@dataclass(frozen=True)
class LinearLayerRuntime:
    """描述一个可直接在 numpy 上执行的线性层。

    业务意图：
    - 这些投影层本质上都是很小的 dense matmul。
    - 继续单独走 ORT 或 PyTorch，会额外引入 graph launch 与 host/device 往返。
    - 这里把权重抽成 numpy，直接在编排层做轻量前向，先消掉最不合理的 runtime 边界。
    """

    weight: np.ndarray
    bias: np.ndarray | None


    def forward(self, value: np.ndarray) -> np.ndarray:
        """执行线性层前向，保持输入前缀维不变。"""

        value = np.asarray(value, dtype=np.float32)
        output = np.matmul(value, self.weight.transpose())
        if self.bias is not None:
            output = output + self.bias
        return np.asarray(output, dtype=np.float32)


@dataclass(frozen=True)
class ScalarQuantizationLayerRuntime:
    """描述 FSQ 轻量前向。

    业务意图：
    - 当前推理处于 eval 模式，不需要保留训练期 straight-through estimator。
    - 因此这里只复刻 `Linear -> Tanh -> Round/Scale -> Linear` 的推理语义。
    """

    in_proj: LinearLayerRuntime
    out_proj: LinearLayerRuntime
    scale: float


    def forward(self, hidden: np.ndarray) -> np.ndarray:
        """执行 FSQ 推理前向。"""

        hidden = self.in_proj.forward(hidden)
        hidden = np.tanh(hidden).astype(np.float32, copy=False)
        hidden = (np.round(hidden * self.scale) / self.scale).astype(np.float32, copy=False)
        return self.out_proj.forward(hidden)


@dataclass(frozen=True)
class StopPredictorRuntime:
    """描述 stop predictor 的轻量前向。"""

    stop_proj: LinearLayerRuntime
    stop_head: LinearLayerRuntime


    def forward(self, lm_hidden: np.ndarray) -> np.ndarray:
        """执行 `stop_head(silu(stop_proj(lm_hidden)))`。"""

        hidden = self.stop_proj.forward(lm_hidden)
        hidden = _silu(hidden).astype(np.float32, copy=False)
        return self.stop_head.forward(hidden)
