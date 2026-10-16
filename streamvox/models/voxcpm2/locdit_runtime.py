"""LocDiT 单步 patch 生成运行时。

该模块负责把最终 M8 主链里与 `LocDiT` 强绑定的细节收口到一起，
避免 decode loop 在每步里反复拼装投影、时间网格和
ORT 调用细节。
"""

from __future__ import annotations

import numpy as np

from .lightweight_ops import LinearLayerRuntime
from .locdit_ort import LocDiTEulerOrtRunner


class LocDiTPatchRuntime:
    """负责执行单步 `LocDiT + CFM` patch 生成。

    参数说明：
    - ``lm_to_dit_proj``：`lm_hidden -> dit_hidden(lm)` 的轻量投影层。
    - ``res_to_dit_proj``：`residual_hidden -> dit_hidden(res)` 的轻量投影层。
    - ``locdit_runner``：已经具备 ORT buffer 复用能力的 Euler 求解器。
    - ``latent_dim``：单个 patch 的 latent 通道数。
    - ``patch_size``：单个 patch 的时间步长度。

    设计意图：
    - 把 `locdit` 相关的缓存都收敛到一个对象里，避免编排层知道太多细节。
    - 跨 decode step 复用 `dit_hidden`、`cond` 和时间网格缓存，减少高频小对象分配。
    """

    def __init__(
        self,
        *,
        lm_to_dit_proj: LinearLayerRuntime,
        res_to_dit_proj: LinearLayerRuntime,
        locdit_runner: LocDiTEulerOrtRunner,
        latent_dim: int,
        patch_size: int,
    ) -> None:
        """初始化 `LocDiT` 单步运行时。"""

        self.lm_to_dit_proj = lm_to_dit_proj
        self.res_to_dit_proj = res_to_dit_proj
        self.locdit_runner = locdit_runner

        # 关键静态形状：当前混合运行时单次只生成 1 个 patch，因此这里固定 batch=1。
        self.latent_dim = int(latent_dim)
        self.patch_size = int(patch_size)

        # `dit_hidden` 由两路投影拼接而成，这里预分配后续循环反复复用的缓冲区。
        self._lm_hidden_dim = int(self.lm_to_dit_proj.weight.shape[0])
        self._res_hidden_dim = int(self.res_to_dit_proj.weight.shape[0])
        self._dit_hidden = np.empty((1, self._lm_hidden_dim + self._res_hidden_dim), dtype=np.float32)

        # `cond` 的 ORT 约定布局固定为 `[B, D, P]`，这里提前常驻一份 buffer。
        self._dit_cond = np.empty((1, self.latent_dim, self.patch_size), dtype=np.float32)

        # `inference_timesteps` 是小集合重复取值，因此显式缓存时间网格。
        self._t_span_cache: dict[int, np.ndarray] = {}

    def decode_patch(
        self,
        *,
        lm_hidden: np.ndarray,
        residual_hidden: np.ndarray,
        prefix_patch: np.ndarray,
        inference_timesteps: int,
        cfg_value: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """执行一次 `LocDiT + CFM` 单步采样，返回 patch-major `curr_patch`。

        边界说明：
        - 当前实现固定单样本生成，因此输入 `lm_hidden / residual_hidden` 都应为一维向量。
        - `prefix_patch` 必须是 patch-major 布局 `[P, D]`，函数内部会转换成 ORT 所需的 `[1, D, P]`。
        """

        self._fill_dit_hidden(
            lm_hidden=np.asarray(lm_hidden, dtype=np.float32),
            residual_hidden=np.asarray(residual_hidden, dtype=np.float32),
        )
        self._fill_dit_condition(np.asarray(prefix_patch, dtype=np.float32))

        # 每个 patch 都需要一份新的扩散噪声；这里仍保持官方随机语义，
        # 但随机源改由调用方显式持有，避免全局 `np.random` 状态污染。
        dit_noise = self._sample_noise(rng)

        dit_t_span = self._get_t_span(inference_timesteps=int(inference_timesteps))

        curr_patch = self.locdit_runner.solve_euler(
            x=dit_noise,
            t_span=dit_t_span,
            mu=self._dit_hidden,
            cond=self._dit_cond,
            cfg_value=float(cfg_value),
            use_cfg_zero_star=True,
        ).squeeze(0).transpose(1, 0)
        return np.asarray(curr_patch, dtype=np.float32)

    def _fill_dit_hidden(
        self,
        *,
        lm_hidden: np.ndarray,
        residual_hidden: np.ndarray,
    ) -> None:
        """把两路 hidden 投影并写入复用中的 `dit_hidden` buffer。"""

        # 先分别投影再写入预分配缓冲区，避免每步都新建 `concatenate` 结果。
        lm_projected = self.lm_to_dit_proj.forward(lm_hidden.reshape(1, -1))
        res_projected = self.res_to_dit_proj.forward(residual_hidden.reshape(1, -1))
        self._dit_hidden[:, : self._lm_hidden_dim] = lm_projected
        self._dit_hidden[:, self._lm_hidden_dim :] = res_projected

    def _fill_dit_condition(self, prefix_patch: np.ndarray) -> None:
        """把 patch-major `prefix_patch` 写成 ORT 所需的 `[1, D, P]` 条件张量。"""

        expected_shape = (self.patch_size, self.latent_dim)
        if prefix_patch.shape != expected_shape:
            raise ValueError(f"prefix_patch shape mismatch: expected {expected_shape}, got {tuple(prefix_patch.shape)}")
        self._dit_cond[0] = prefix_patch.transpose(1, 0)

    def _get_t_span(self, *, inference_timesteps: int) -> np.ndarray:
        """获取并缓存指定步数对应的 CFM 时间网格。"""

        if inference_timesteps <= 0:
            raise ValueError("inference_timesteps must be greater than 0")

        cached = self._t_span_cache.get(inference_timesteps)
        if cached is not None:
            return cached

        # 这里保持与官方 CFM 完全一致的时间重参数化，避免优化引入语义偏移。
        dit_t_span = np.linspace(1.0, 0.0, inference_timesteps + 1, dtype=np.float32)
        dit_t_span = dit_t_span + (np.cos(np.pi / 2 * dit_t_span) - 1.0 + dit_t_span)
        self._t_span_cache[inference_timesteps] = np.asarray(dit_t_span, dtype=np.float32)
        return self._t_span_cache[inference_timesteps]

    def _sample_noise(self, rng: np.random.Generator) -> np.ndarray:
        """采样一份 `LocDiT` 扩散噪声，并兼容不同 numpy 版本。"""

        noise_shape = (1, self.latent_dim, self.patch_size)

        # 新版 numpy 支持直接指定 `dtype=np.float32`，这样可以少一次类型收缩拷贝。
        try:
            return rng.standard_normal(noise_shape, dtype=np.float32)
        except TypeError:
            return rng.standard_normal(noise_shape).astype(np.float32, copy=False)
