import os
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

try:
    import torch
except Exception as e:
    raise RuntimeError("PyTorch가 필요합니다. pip install torch") from e

# VideoPose3D 레포를 git clone 후, sys.path에 루트를 추가해야 합니다.
# 예) !git clone https://github.com/facebookresearch/VideoPose3D.git
#    import sys; sys.path.append("/content/VideoPose3D")
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
        model.load_state_dict(ckpt["model_pos"], strict=True)  # 체크포인트 구조와 동일해야 함
    except Exception as e:
        raise Pose3DError(f"체크포인트 로드 실패: {e}")
    model = model.to(device).eval()
    return model


def _extend_temporally_to_min_length(norm_2d: np.ndarray, min_len: int) -> np.ndarray:
    """
    입력 시퀀스 길이(T)가 min_len보다 짧을 때, 시간축으로 확장합니다.
    우선 반사(reflect) 패딩, 부족하면 반복(repeat)로 추가 보강하여 최소 길이 보장.
    """
    T = int(norm_2d.shape[0])
    if T >= min_len:
        return norm_2d

    need = min_len - T
    left = np.flip(norm_2d[:min(T, need)], axis=0)
    right = np.flip(norm_2d[-min(T, need):], axis=0)
    ext = np.concatenate([left, norm_2d, right], axis=0)
    if ext.shape[0] >= min_len:
        return ext

    rep_times = int(np.ceil((min_len - ext.shape[0]) / max(1, T)))
    reps = [norm_2d for _ in range(rep_times)]
    ext2 = np.concatenate([ext] + reps, axis=0)
    return ext2[:min_len]


def _ensure_min_window_and_pad_tensor(X: np.ndarray, need_len: int) -> np.ndarray:
    """
    패딩 후 텐서 X의 시간축 길이가 need_len 미만이면 뒤쪽 반복 패딩으로 보강합니다.
    """
    if X.shape[0] >= need_len:
        return X
    lack = need_len - X.shape[0]
    tail = np.repeat(X[-1:, :, :], repeats=lack, axis=0)
    return np.concatenate([X, tail], axis=0)


def _infer_center_crop(
    model: TemporalModel,
    norm_2d: np.ndarray,
    RF: int,
    device: str
) -> np.ndarray:
    """
    모델의 유효 커널 요구치를 만족하도록 입력을 확장/패딩하고,
    센터-크롭 슬라이딩 방식으로 프레임별 3D를 추론합니다.
    """
    # 안전 마진 포함 최소 길이(내부 스택에서 19 이상 요구되는 케이스 방지)
    min_needed = max(RF, 27)
    norm_2d_safe = _extend_temporally_to_min_length(norm_2d, min_needed)

    pad = (RF - 1) // 2
    X = _pad_sequence_reflect(norm_2d_safe, pad)      # (T'+2*pad, 17, 2)
    T_prime = norm_2d_safe.shape[0]
    need_total = max(T_prime + 2 * pad, min_needed + 2 * pad)
    X = _ensure_min_window_and_pad_tensor(X, need_total)

    out = np.zeros((T_prime, 17, 3), dtype=np.float32)

    with torch.no_grad():
        Xt = torch.from_numpy(X[None]).float().to(device)  # (1, T'', 17, 2)
        for i in range(T_prime):
            s = i
            e = i + 2 * pad + 1
            # 윈도 길이 보장
            if (e - s) < min_needed:
                e = s + min_needed
            if e > Xt.shape[1]:
                lack = e - Xt.shape[1]
                tail = Xt[:, -1:, :, :].repeat(1, lack, 1, 1)
                Xt = torch.cat([Xt, tail], dim=1)
            window = Xt[:, s:e, :, :]                      # (1, >=min_needed, 17, 2)

            y = model(window)                              # (1, t_out, 17, 3)
            y_np = y.detach().cpu().numpy()
            t_out = y_np.shape[1]
            center_idx = min((t_out - 1) // 2, (e - s - 1) // 2)
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
        "pose3d": (T',17,3) 의미 좌표계,
        "pose2d_norm": (T,17,2),
        "R_seq": (T',3,3),
        "S_seq": (T',),
        "meta": {"RF": int, "PAD": int, "device": str, "T_in": int, "T_used": int}
    }
    주의:
    - filter_widths는 체크포인트와 동일([3,3,3,3,3])을 기본으로 하며, 짧은 입력은 내부 시간축 확장으로 처리합니다.
    - T'은 내부 확장/경계 보정으로 인해 T와 다를 수 있습니다(안정성 우선).
    """
    if "kps_2d" not in kps_pack:
        raise Pose3DError("kps_pack에 'kps_2d'가 없습니다.")
    kps_2d = np.asarray(kps_pack["kps_2d"], dtype=np.float32)
    if kps_2d.ndim != 3 or kps_2d.shape[1:] != (17, 2):
        raise Pose3DError(f"kps_2d 형태가 (T,17,2)가 아닙니다: {kps_2d.shape}")

    # 체크포인트와 일치하는 기본 구조 유지(일반적으로 [3,3,3,3,3])
    filter_widths = filter_widths or [3, 3, 3, 3, 3]
    RF = _receptive_field(filter_widths)
    PAD = (RF - 1) // 2
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 2D 정규화
    norm_2d, scale_2d, pelvis_2d = _normalize_2d_sequence(kps_2d)
    T_in = int(norm_2d.shape[0])

    # 모델 로드(체크포인트 구조와 동일)
    model = _load_temporal_model(ckpt_path=ckpt_path, filter_widths=filter_widths, causal=causal, device=device)

    # 3D 추론(짧은/경계 시퀀스는 내부에서 시간축 확장 및 윈도 길이 보장)
    y3d_raw = _infer_center_crop(model, norm_2d, RF=RF, device=device)
    T_used = int(y3d_raw.shape[0])

    # 의미 좌표계 정규화
    pose3d_meaning, R_seq, S_seq = _normalize_3d_meaning(y3d_raw)

    return {
        "pose3d": pose3d_meaning.astype(np.float32),
        "pose2d_norm": norm_2d.astype(np.float32),
        "R_seq": R_seq.astype(np.float32),
        "S_seq": S_seq.astype(np.float32),
        "meta": {"RF": int(RF), "PAD": int(PAD), "device": device, "T_in": int(T_in), "T_used": int(T_used)}
    }
