import os
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

try:
    import torch
except Exception as e:
    raise RuntimeError("PyTorch가 필요합니다. pip install torch") from e

# VideoPose3D 레포를 git clone 후, sys.path에 루트를 추가해야 합니다.
# from common.model import TemporalModel 가 정상임포트되어야 합니다.
try:
    from common.model import TemporalModel  # VideoPose3D
except Exception as e:
    raise RuntimeError("VideoPose3D 임포트 실패. 레포 경로를 sys.path에 추가했는지 확인하세요.") from e


# COCO17 인덱스
L_SHO, R_SHO = 5, 6
L_HIP, R_HIP = 11, 12


class Pose3DError(RuntimeError):
    pass


def _pelvis_center_2d(P: np.ndarray) -> np.ndarray:
    return (P[L_HIP] + P[R_HIP]) / 2.0


def _shoulder_center_2d(P: np.ndarray) -> np.ndarray:
    return (P[L_SHO] + P[R_SHO]) / 2.0


def _normalize_2d_sequence(kps_2d_seq: np.ndarray, min_scale: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = int(kps_2d_seq.shape[0])
    out = np.zeros_like(kps_2d_seq, dtype=np.float32)
    scales = np.zeros((T,), dtype=np.float32)
    pelvises = np.zeros((T, 2), dtype=np.float32)
    for t in range(T):
        P = kps_2d_seq[t].astype(np.float32)
        pelvis = _pelvis_center_2d(P)
        shoulder_c = _shoulder_center_2d(P)
        P = P - pelvis
        scale = float(np.linalg.norm(shoulder_c - pelvis))
        if not np.isfinite(scale) or scale < min_scale:
            scale = 1.0
        out[t] = P / scale
        scales[t] = scale
        pelvises[t] = pelvis
    return out, scales, pelvises


def _pad_sequence_reflect(x: np.ndarray, pad: int) -> np.ndarray:
    if pad <= 0:
        return x
    T = x.shape[0]
    if T >= pad:
        left = np.flip(x[:pad], axis=0)
        right = np.flip(x[-pad:], axis=0)
    else:
        left = np.flip(np.repeat(x[:1], pad, axis=0), axis=0)
        right = np.flip(np.repeat(x[-1:], pad, axis=0), axis=0)
    return np.concatenate([left, x, right], axis=0)


def _receptive_field(filter_widths: List[int]) -> int:
    rf = 1
    for fw in filter_widths:
        rf += (fw - 1)
    return rf


def _normalize_3d_meaning(kps_3d_seq: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = int(kps_3d_seq.shape[0])
    out = np.zeros_like(kps_3d_seq, dtype=np.float32)
    R_list: List[np.ndarray] = []
    S_list: List[float] = []
    for t in range(T):
        P = kps_3d_seq[t].astype(np.float64)
        pelvis = (P[L_HIP] + P[R_HIP]) / 2.0
        shoulder = (P[L_SHO] + P[R_SHO]) / 2.0

        P = P - pelvis

        x_vec = (P[R_HIP] - P[L_HIP])
        x_axis = x_vec / (np.linalg.norm(x_vec) + eps)

        up_vec = (shoulder - pelvis)
        up_axis = up_vec / (np.linalg.norm(up_vec) + eps)

        z_axis = up_axis - np.dot(up_axis, x_axis) * x_axis
        z_axis = z_axis / (np.linalg.norm(z_axis) + eps)

        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / (np.linalg.norm(y_axis) + eps)

        R = np.stack([x_axis, y_axis, z_axis], axis=0)  # (3,3)
        P_rot = (R @ P.T).T

        shoulder_rot = (R @ (shoulder - pelvis).T).T
        scale = float(np.linalg.norm(shoulder_rot) + eps)
        P_norm = P_rot / scale

        out[t] = P_norm.astype(np.float32)
        R_list.append(R.astype(np.float32))
        S_list.append(scale)
    return out, np.array(R_list, dtype=np.float32), np.array(S_list, dtype=np.float32)


def _load_temporal_model(
    ckpt_path: str,
    filter_widths: List[int],
    causal: bool,
    device: str
) -> TemporalModel:
    model = TemporalModel(17, 2, 17, filter_widths=filter_widths, causal=causal)
    if not os.path.exists(ckpt_path):
        raise Pose3DError(f"체크포인트 파일을 찾을 수 없습니다: {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_pos"], strict=True)
    except Exception as e:
        raise Pose3DError(f"체크포인트 로드 실패: {e}")
    model = model.to(device).eval()
    return model


def _infer_center_crop(
    model: TemporalModel,
    norm_2d: np.ndarray,
    RF: int,
    device: str
) -> np.ndarray:
    pad = (RF - 1) // 2
    X = _pad_sequence_reflect(norm_2d, pad)          # (T+2*pad, 17, 2)
    T = norm_2d.shape[0]
    out = np.zeros((T, 17, 3), dtype=np.float32)
    with torch.no_grad():
        Xt = torch.from_numpy(X[None]).float().to(device)  # (1, T+2*pad, 17, 2)
        for i in range(T):
            s, e = i, i + 2 * pad + 1
            window = Xt[:, s:e, :, :]                      # (1, RF, 17, 2)
            y = model(window)                              # (1, t_out, 17, 3)
            y_np = y.detach().cpu().numpy()
            t_out = y_np.shape[1]
            center_idx = min((t_out - 1) // 2, pad)
            out[i] = y_np[0, center_idx]
    return out


def run_vpose3d(
    kps_pack: Dict[str, Any],
    ckpt_path: str = "/content/pretrained_h36m_cpn.bin",
    filter_widths: Optional[List[int]] = None,
    causal: bool = False,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    입력: kps_pack = {"kps_2d": (T,17,2), ...}
    출력: {
        "pose3d": (T,17,3) 의미 좌표계,
        "pose2d_norm": (T,17,2),
        "R_seq": (T,3,3),
        "S_seq": (T,),
        "meta": {"RF": int, "PAD": int, "device": str}
    }
    """
    if "kps_2d" not in kps_pack:
        raise Pose3DError("kps_pack에 'kps_2d'가 없습니다.")
    kps_2d = np.asarray(kps_pack["kps_2d"], dtype=np.float32)
    if kps_2d.ndim != 3 or kps_2d.shape[1:] != (17, 2):
        raise Pose3DError(f"kps_2d 형태가 (T,17,2)가 아닙니다: {kps_2d.shape}")

    filter_widths = filter_widths or [3, 3, 3, 3, 3]
    RF = _receptive_field(filter_widths)
    PAD = (RF - 1) // 2
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    norm_2d, scale_2d, pelvis_2d = _normalize_2d_sequence(kps_2d)

    model = _load_temporal_model(ckpt_path=ckpt_path, filter_widths=filter_widths, causal=causal, device=device)
    y3d_raw = _infer_center_crop(model, norm_2d, RF=RF, device=device)

    pose3d_meaning, R_seq, S_seq = _normalize_3d_meaning(y3d_raw)

    return {
        "pose3d": pose3d_meaning.astype(np.float32),
        "pose2d_norm": norm_2d.astype(np.float32),
        "R_seq": R_seq.astype(np.float32),
        "S_seq": S_seq.astype(np.float32),
        "meta": {"RF": int(RF), "PAD": int(PAD), "device": device}
    }