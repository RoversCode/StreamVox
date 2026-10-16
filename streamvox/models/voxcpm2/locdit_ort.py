"""按原有 NumPy 公式求解 LocDiT Euler 步骤并复用缓存。"""
from __future__ import annotations
import numpy as np

class LocDiTEulerOrtRunner:
    """描述一份带缓存与 I/O binding 的 LocDiT Euler runner。

    业务意图：
    - 当前 decode 每个 patch 会在 Euler 子步里重复调用同一份 `LocDiT estimator`。
    - 旧实现每一步都会新建 `x_in / mu_in / cond_in / t_in / dt_in`，并让 ORT 重新分配输出。
    - 这里把这些中间 buffer 固定下来：
      - 跨 patch 复用 runner 实例
      - 跨 Euler step 复用输入/输出数组
      - 在支持时优先使用 ORT `I/O binding`
    - 这样先消掉高频 host 分配与 Python 字典装箱，再为后续更激进的 fused graph 留出接口。
    """

    def __init__(self, session) -> None:
        """初始化 runner；构造时不立即分配具体 shape 的缓存。"""

        self.session = session
        self.input_names = {item.name for item in session.get_inputs()}
        self.output_name = session.get_outputs()[0].name
        self._supports_iobinding = hasattr(session, "io_binding") and hasattr(session, "run_with_iobinding")
        self._cached_batch = 0
        self._cached_channels = 0
        self._cached_patch = 0
        self._cached_mu_dim = 0
        self._x_in: np.ndarray | None = None
        self._mu_in: np.ndarray | None = None
        self._t_in: np.ndarray | None = None
        self._dt_in: np.ndarray | None = None
        self._cond_in: np.ndarray | None = None
        self._velocity_out: np.ndarray | None = None
        self._io_binding = None
        self._output_ortvalue = None
        active_providers = list(session.get_providers())
        self._active_provider = active_providers[0] if active_providers else ""

    def _ensure_buffers(
        self,
        *,
        batch_size: int,
        channels: int,
        patch_size: int,
        mu_dim: int,
    ) -> None:
        """按当前 batch/shape 申请或复用缓存。"""

        if (
            self._x_in is not None
            and self._cached_batch == int(batch_size)
            and self._cached_channels == int(channels)
            and self._cached_patch == int(patch_size)
            and self._cached_mu_dim == int(mu_dim)
        ):
            return

        doubled_batch = int(batch_size) * 2
        self._cached_batch = int(batch_size)
        self._cached_channels = int(channels)
        self._cached_patch = int(patch_size)
        self._cached_mu_dim = int(mu_dim)
        self._x_in = np.empty((doubled_batch, channels, patch_size), dtype=np.float32)
        self._mu_in = np.empty((doubled_batch, mu_dim), dtype=np.float32)
        self._t_in = np.empty((doubled_batch,), dtype=np.float32)
        self._dt_in = np.zeros((doubled_batch,), dtype=np.float32)
        self._cond_in = np.empty((doubled_batch, channels, patch_size), dtype=np.float32)
        self._velocity_out = np.empty((doubled_batch, channels, patch_size), dtype=np.float32)

        if self._supports_iobinding:
            try:
                import onnxruntime as ort
            except ImportError:
                self._supports_iobinding = False
                self._io_binding = None
                self._output_ortvalue = None
                return

            # 这里把输出固定绑定到预分配 numpy buffer。
            # 业务意图是避免每个 Euler 子步都让 ORT 新建一份输出数组。
            self._io_binding = self.session.io_binding()
            self._output_ortvalue = ort.OrtValue.ortvalue_from_numpy(self._velocity_out)

    def solve_euler(
        self,
        *,
        x: np.ndarray,
        t_span: np.ndarray,
        mu: np.ndarray,
        cond: np.ndarray,
        cfg_value: float,
        use_cfg_zero_star: bool,
    ) -> np.ndarray:
        """使用缓存与 I/O binding 执行一轮 Euler 求解。"""

        x = np.asarray(x, dtype=np.float32).copy()
        t_span = np.asarray(t_span, dtype=np.float32)
        mu = np.asarray(mu, dtype=np.float32)
        cond = np.asarray(cond, dtype=np.float32)

        batch_size, channels, patch_size = x.shape
        mu_dim = int(mu.shape[-1])
        self._ensure_buffers(
            batch_size=batch_size,
            channels=channels,
            patch_size=patch_size,
            mu_dim=mu_dim,
        )

        x_in = self._x_in
        mu_in = self._mu_in
        t_in = self._t_in
        dt_in = self._dt_in
        cond_in = self._cond_in
        velocity_out = self._velocity_out

        if x_in is None or mu_in is None or t_in is None or dt_in is None or cond_in is None or velocity_out is None:
            raise RuntimeError("LocDiT ORT runner buffers are not initialized")

        # `mu` 与 `cond` 在一个 patch 的 Euler 全流程内保持不变，只需预填一次。
        cond_in[:batch_size] = cond
        cond_in[batch_size:] = cond
        mu_in[:batch_size] = mu
        mu_in[batch_size:].fill(0.0)
        dt_in.fill(0.0)

        current_t = float(t_span[0])
        current_dt = float(t_span[0] - t_span[1])
        zero_init_steps = max(1, int(len(t_span) * 0.04))

        for step in range(1, len(t_span)):
            if use_cfg_zero_star and step <= zero_init_steps:
                dphi_dt = np.zeros_like(x, dtype=np.float32)
            else:
                x_in[:batch_size] = x
                x_in[batch_size:] = x
                t_in.fill(current_t)

                velocity = self._run_bound_velocity(
                    x_in=x_in,
                    mu_in=mu_in,
                    t_in=t_in,
                    cond_in=cond_in,
                    dt_in=dt_in,
                )
                dphi_dt = velocity[:batch_size]
                cfg_dphi_dt = velocity[batch_size:]

                if use_cfg_zero_star:
                    positive_flat = dphi_dt.reshape(batch_size, -1)
                    negative_flat = cfg_dphi_dt.reshape(batch_size, -1)
                    st_star = _optimized_scale(positive_flat, negative_flat).reshape(batch_size, 1, 1)
                else:
                    st_star = 1.0

                dphi_dt = cfg_dphi_dt * st_star + cfg_value * (dphi_dt - cfg_dphi_dt * st_star)

            x = x - current_dt * dphi_dt
            current_t = current_t - current_dt
            if step < len(t_span) - 1:
                current_dt = current_t - float(t_span[step + 1])

        return x

    def _run_bound_velocity(
        self,
        *,
        x_in: np.ndarray,
        mu_in: np.ndarray,
        t_in: np.ndarray,
        cond_in: np.ndarray,
        dt_in: np.ndarray,
    ) -> np.ndarray:
        """执行一次 estimator 前向，优先走 I/O binding。"""

        if self._supports_iobinding and self._io_binding is not None and self._output_ortvalue is not None:
            try:
                self._io_binding.clear_binding_inputs()
                self._io_binding.clear_binding_outputs()
                self._io_binding.bind_cpu_input("x", x_in)
                self._io_binding.bind_cpu_input("mu", mu_in)
                self._io_binding.bind_cpu_input("t", t_in)
                self._io_binding.bind_cpu_input("cond", cond_in)
                self._io_binding.bind_cpu_input("dt", dt_in)
                self._io_binding.bind_ortvalue_output(self.output_name, self._output_ortvalue)
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()
                return np.asarray(self._velocity_out, dtype=np.float32)
            except Exception:
                # 业务意图：
                # - 某些 provider 组合下，CPU numpy buffer + I/O binding 可能存在兼容差异。
                # - 这里一旦触发异常，就永久回退到稳妥的 `session.run`，避免主链推理被实验性优化阻塞。
                self._supports_iobinding = False
                self._io_binding = None
                self._output_ortvalue = None

        velocity = self.session.run(
            None,
            {
                "x": x_in,
                "mu": mu_in,
                "t": t_in,
                "cond": cond_in,
                "dt": dt_in,
            },
        )[0]
        return np.asarray(velocity, dtype=np.float32)



def _optimized_scale(positive_flat: np.ndarray, negative_flat: np.ndarray) -> np.ndarray:
    """计算 CFG-Zero-Star 的缩放系数。"""

    dot_product = np.sum(positive_flat * negative_flat, axis=1, keepdims=True)
    squared_norm = np.sum(negative_flat**2, axis=1, keepdims=True) + 1e-8
    return dot_product / squared_norm
