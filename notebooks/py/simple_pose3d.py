# simple_pose3d.py

import time
import logging
from pathlib import Path
import numpy as np

try:
    import torch
except ImportError as e:
    raise RuntimeError("PyTorch가 필요합니다. pip install torch") from e

import sys
# VideoPose3D 레포 경로를 Colab 기준으로 추가 (필요 시 수정)
if "/content/VideoPose3D" not in sys.path:
    sys.path.append("/content/VideoPose3D")
try:
    from common.model import TemporalModel
except ImportError as e:
    raise RuntimeError("VideoPose3D 임포트 실패. 레포를 clone 후 sys.path에 추가하세요.") from e

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simple_pose3d_fixed")

# COCO17 인덱스
L_SHO, R_SHO = 5, 6
L_HIP, R_HIP = 11, 12

# 고정 설정: 요청에 따라 단일 구성
FILTER_WIDTHS = [3, 3, 3, 3, 3]  # 단일 고정


def _auto_device() -> str:
    """디바이스 자동 선택: cuda > mps > cpu"""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _receptive_field(filter_widths) -> int:
    """TemporalModel 수용영역 계산"""
    rf = 1
    for fw in filter_widths:
        rf += (fw - 1)
    return rf


def _extend_min_len(x: np.ndarray, min_len: int) -> np.ndarray:
    """시퀀스 길이가 RF 미만일 때 최소 길이까지 확장"""
    T = x.shape[0]
    if T >= min_len:
        return x
    if T == 0:
        raise RuntimeError("빈 시퀀스입니다.")
    need = min_len - T
    left = np.flip(x[:min(need // 2, T)], axis=0)
    right = np.flip(x[-min(need - min(need // 2, T), T):], axis=0)
    ext = np.concatenate([left, x, right], axis=0)
    if ext.shape[0] < min_len:
        reps = (min_len - ext.shape[0] + T - 1) // T
        ext = np.concatenate([ext] + [x] * reps, axis=0)
    return ext[:min_len]


def _normalize_2d(kps_2d: np.ndarray, min_scale: float = 1e-6):
    """
    2D 키포인트를 골반 원점/몸통 스케일로 정규화
    - 입력: (T,17,2)
    - 출력: norm_2d(T,17,2), scales(T,), pelvises(T,2)
    """
    T = kps_2d.shape[0]
    out = np.zeros_like(kps_2d, dtype=np.float32)
    scales = np.zeros((T,), dtype=np.float32)
    pelvises = np.zeros((T, 2), dtype=np.float32)
    valid = []
    for t in range(T):
        P = kps_2d[t].astype(np.float32)
        pelvis = (P[L_HIP] + P[R_HIP]) / 2.0
        shoulder = (P[L_SHO] + P[R_SHO]) / 2.0
        Pc = P - pelvis
        scale = float(np.linalg.norm(shoulder - pelvis))
        if not np.isfinite(scale) or scale < min_scale:
            scale = float(np.median(valid)) if valid else 1.0
        else:
            valid.append(scale)
        out[t] = Pc / max(scale, min_scale)
        scales[t] = scale
        pelvises[t] = pelvis
    return out, scales, pelvises


def _safe_norm(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """벡터 정규화(너무 작으면 첫 축을 1로 하는 단위 벡터 반환)"""
    n = np.linalg.norm(v)
    if n < eps:
        u = np.zeros_like(v)
        if u.size > 0:
            u[0] = 1.0
        return u
    return v / n


def _normalize_3d_meaning(P3: np.ndarray, eps: float = 1e-8):
    """
    3D 포즈를 의미 좌표계로 변환
    - 원점: 골반 중심
    - X: 오른쪽엉덩이→왼쪽엉덩이
    - Z: 위쪽 성분(어깨→골반에서 X 성분 제거 후 정규화)
    - Y: Z × X
    """
    T = P3.shape[0]
    out = np.zeros_like(P3, dtype=np.float32)
    R_seq = np.zeros((T, 3, 3), dtype=np.float32)
    S_seq = np.zeros((T,), dtype=np.float32)
    for t in range(T):
        P = P3[t].astype(np.float64)
        pelvis = (P[L_HIP] + P[R_HIP]) / 2.0
        shoulder = (P[L_SHO] + P[R_SHO]) / 2.0
        Pc = P - pelvis
        x_axis = _safe_norm(P[R_HIP] - P[L_HIP], eps)
        up = _safe_norm(shoulder - pelvis, eps)
        z_axis = _safe_norm(up - np.dot(up, x_axis) * x_axis, eps)
        y_axis = _safe_norm(np.cross(z_axis, x_axis), eps)
        R = np.stack([x_axis, y_axis, z_axis], axis=0).astype(np.float64)
        Prot = (R @ Pc.T).T
        scale = float(np.linalg.norm(R @ (shoulder - pelvis)) + eps)
        Pn = Prot / max(scale, eps)
        out[t] = Pn.astype(np.float32)
        R_seq[t] = R.astype(np.float32)
        S_seq[t] = scale
    return out, R_seq, S_seq


def infer_vpose3d(
    npz_2d_path: str,
    ckpt_path: str,
    out_npz_path: str = "vpose3d.npz",
    causal: bool = False,
    strict_load_first: bool = True
) -> str:
    """
    coco17_2d.npz → VideoPose3D(고정 FILTER_WIDTHS) → vpose3d.npz
    - npz_2d_path: simple_pose2d.py가 생성한 coco17_2d.npz 경로
    - ckpt_path: VideoPose3D 체크포인트(.bin)
    - out_npz_path: 저장 경로
    - strict_load_first: True면 state_dict strict=True 우선 적용, 실패 시 False로 재시도
    """
    # 1) 2D 로드 및 정규화
    data = np.load(npz_2d_path)
    if "kps_2d" not in data:
        raise RuntimeError("입력 npz에 'kps_2d'가 없습니다.")
    kps_2d = data["kps_2d"].astype(np.float32)
    norm_2d, _, _ = _normalize_2d(kps_2d)

    # 2) 모델 로드
    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"체크포인트 없음: {ckpt}")
    device = _auto_device()
    model = TemporalModel(17, 2, 17, filter_widths=FILTER_WIDTHS, causal=causal).to(device).eval()

    sd = torch.load(str(ckpt), map_location="cpu")
    sd = sd.get("model_pos", sd.get("model", sd))
    try:
        model.load_state_dict(sd, strict=strict_load_first)
    except Exception as e:
        logger.warning(f"[3D] strict={strict_load_first} 로드 실패 → strict=False 재시도: {e}")
        model.load_state_dict(sd, strict=False)

    # 3) RF 패딩 후 중심 윈도 추론
    RF = _receptive_field(FILTER_WIDTHS)
    pad = (RF - 1) // 2
    X = _extend_min_len(norm_2d, RF)
    Xpad = np.concatenate([np.flip(X[:pad], 0), X, np.flip(X[-pad:], 0)], axis=0)

    T = norm_2d.shape[0]
    out3d = np.zeros((T, 17, 3), dtype=np.float32)

    with torch.no_grad():
        Xt = torch.from_numpy(Xpad).float().to(device)
        for i in range(T):
            s, e = i, i + 2 * pad + 1
            y = model(Xt[s:e].unsqueeze(0))  # (1, t_out, 17, 3)
            y_np = y.cpu().numpy()
            center_idx = min((y_np.shape[1] - 1) // 2, pad)
            out3d[i] = y_np[0, center_idx]

    # 4) 의미 좌표계 변환 및 저장
    pose3d, R_seq, S_seq = _normalize_3d_meaning(out3d)
    out_p = Path(out_npz_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_p),
        pose3d=pose3d.astype(np.float32),
        pose2d_norm=norm_2d.astype(np.float32),
        R_seq=R_seq.astype(np.float32),
        S_seq=S_seq.astype(np.float32),
        RF=int(RF),
        filter_widths=np.array(FILTER_WIDTHS),
    )
    logger.info(f"[3D] 저장 완료: {out_p} | pose3d={pose3d.shape}")
    return str(out_p)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="VideoPose3D 3D 추정 (FILTER_WIDTHS=[3,3,3,3,3] 고정)")
    ap.add_argument("--npz2d", required=True, help="coco17_2d.npz 경로")
    ap.add_argument("--ckpt", required=True, help="VideoPose3D 체크포인트 경로(.bin)")
    ap.add_argument("--out", default="vpose3d.npz", help="출력 npz 경로")
    ap.add_argument("--strict_first", type=int, default=1, help="state_dict strict=True 우선 시도(1/0)")
    args = ap.parse_args()
    infer_vpose3d(args.npz2d, args.ckpt, args.out, strict_load_first=bool(args.strict_first))