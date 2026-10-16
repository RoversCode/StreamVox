"""离线比较 LocDiT 中间输入输出；不进入发行 wheel。"""
from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
from streamvox.models.voxcpm2.locdit_ort import LocDiTEulerOrtRunner
from streamvox.models.voxcpm2.ort_runtime import create_ort_session, run_single_input

@dataclass(frozen=True)
class LocDiTOrtCompareResult:
    """描述一次 M4 ORT 对拍结果。"""

    locdit_onnx_path: str
    lm_to_dit_proj_onnx_path: str
    res_to_dit_proj_onnx_path: str
    case_dir: str
    providers: list[str]
    dit_hidden: dict
    curr_patch: dict

    def to_dict(self) -> dict:
        """转为 JSON 友好的字典。"""

        return asdict(self)









def compare_locdit_ort_against_case(
    *,
    locdit_onnx_path: str,
    lm_to_dit_proj_onnx_path: str,
    res_to_dit_proj_onnx_path: str,
    case_dir: str,
    prefer_gpu: bool = True,
) -> LocDiTOrtCompareResult:
    """用 M0 case 对拍 `lm_to_dit_proj + res_to_dit_proj + LocDiT + CFM` 的 ORT 输出。"""

    # [参考迁移保留]
    # 该入口只服务离线 case 对拍，不会被 runtime 初始化路径触发。
    case_root = Path(case_dir).resolve()
    manifest = json.loads((case_root / "manifest.json").read_text(encoding="utf-8"))
    cfg_value = float(manifest["case"]["cfg_value"])

    lm_hidden = _load_case_tensor(case_root, "lm_hidden")[None, ...]
    residual_hidden = _load_case_tensor(case_root, "residual_hidden")[None, ...]
    expected_dit_hidden = _load_case_tensor(case_root, "dit_hidden")
    dit_noise = _load_case_tensor(case_root, "dit_noise")[None, ...]
    dit_t_span = _load_case_tensor(case_root, "dit_t_span")
    dit_cond = _load_case_tensor(case_root, "dit_cond")[None, ...]
    expected_curr_patch = _load_case_tensor(case_root, "curr_patch")

    lm_proj_session, lm_bundle = create_ort_session(lm_to_dit_proj_onnx_path, prefer_gpu=prefer_gpu)
    res_proj_session, _ = create_ort_session(res_to_dit_proj_onnx_path, prefer_gpu=prefer_gpu)
    locdit_session, _ = create_ort_session(locdit_onnx_path, prefer_gpu=prefer_gpu)
    locdit_runner = LocDiTEulerOrtRunner(locdit_session)

    dit_hidden, curr_patch = _run_locdit_pipeline(
        lm_proj_session=lm_proj_session,
        res_proj_session=res_proj_session,
        locdit_session=locdit_session,
        locdit_runner=locdit_runner,
        lm_hidden=lm_hidden,
        residual_hidden=residual_hidden,
        dit_noise=dit_noise,
        dit_t_span=dit_t_span,
        dit_cond=dit_cond,
        cfg_value=cfg_value,
    )

    return LocDiTOrtCompareResult(
        locdit_onnx_path=str(Path(locdit_onnx_path).resolve()),
        lm_to_dit_proj_onnx_path=str(Path(lm_to_dit_proj_onnx_path).resolve()),
        res_to_dit_proj_onnx_path=str(Path(res_to_dit_proj_onnx_path).resolve()),
        case_dir=str(case_root),
        providers=lm_bundle.providers,
        dit_hidden=_compare_array(dit_hidden, expected_dit_hidden),
        curr_patch=_compare_array(curr_patch, expected_curr_patch),
    )









def write_locdit_ort_compare(
    *,
    locdit_onnx_path: str,
    lm_to_dit_proj_onnx_path: str,
    res_to_dit_proj_onnx_path: str,
    case_dir: str,
    output_path: str,
    prefer_gpu: bool = True,
) -> LocDiTOrtCompareResult:
    """执行对拍并写出 JSON。"""

    # [参考迁移保留]
    # 这里保留 JSON 落盘能力，方便后续回归时直接比较历史验收结果。
    result = compare_locdit_ort_against_case(
        locdit_onnx_path=locdit_onnx_path,
        lm_to_dit_proj_onnx_path=lm_to_dit_proj_onnx_path,
        res_to_dit_proj_onnx_path=res_to_dit_proj_onnx_path,
        case_dir=case_dir,
        prefer_gpu=prefer_gpu,
    )
    output_file = Path(output_path).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return result



def _run_locdit_pipeline(
    *,
    lm_proj_session=None,
    res_proj_session=None,
    locdit_session,
    locdit_runner: LocDiTEulerOrtRunner | None = None,
    lm_hidden: np.ndarray | None = None,
    residual_hidden: np.ndarray | None = None,
    dit_hidden: np.ndarray | None = None,
    dit_noise: np.ndarray,
    dit_t_span: np.ndarray,
    dit_cond: np.ndarray,
    cfg_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    """运行一次完整的 M4 ORT pipeline。"""

    # 这里同时兼容两类调用路径：
    # 1. M4 原始路径：两路投影仍然走 ORT session。
    # 2. M8.2 轻量路径：投影层已经在编排层内联，只把预先拼好的 `dit_hidden` 传进来。
    if dit_hidden is None:
        if lm_proj_session is None or res_proj_session is None:
            raise ValueError("lm_proj_session/res_proj_session 与 dit_hidden 至少要提供一组")
        if lm_hidden is None or residual_hidden is None:
            raise ValueError("未提供 dit_hidden 时，必须同时提供 lm_hidden 与 residual_hidden")
        dit_hidden_lm = run_single_input(lm_proj_session, "lm_hidden", lm_hidden)
        dit_hidden_res = run_single_input(res_proj_session, "residual_hidden", residual_hidden)
        dit_hidden = np.concatenate([dit_hidden_lm, dit_hidden_res], axis=-1)
    else:
        dit_hidden = np.asarray(dit_hidden, dtype=np.float32)
        if dit_hidden.ndim == 1:
            dit_hidden = dit_hidden[None, ...]

    if locdit_runner is None:
        locdit_runner = LocDiTEulerOrtRunner(locdit_session)

    curr_patch = locdit_runner.solve_euler(
        x=dit_noise,
        t_span=dit_t_span,
        mu=dit_hidden,
        cond=dit_cond,
        cfg_value=cfg_value,
        use_cfg_zero_star=True,
    ).squeeze(0).transpose(1, 0)
    return dit_hidden.squeeze(0), curr_patch






def _load_case_tensor(case_dir: Path, tensor_name: str) -> np.ndarray:
    """读取单个 case 张量。"""

    tensor_path = case_dir / f"{tensor_name}.npy"
    if not tensor_path.exists():
        raise FileNotFoundError(f"missing tensor file: {tensor_path}")
    return np.load(tensor_path).astype(np.float32)









def _compare_array(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    """计算数值误差指标。"""

    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    abs_diff = np.abs(actual - expected)
    return {
        "max_abs_diff": float(abs_diff.max(initial=0.0)),
        "mean_abs_diff": float(abs_diff.mean()) if abs_diff.size else 0.0,
    }

