import os
import logging
import time
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# 의존성 체크
try:
    import torch
    import torch.nn.functional as F
except ImportError as e:
    raise RuntimeError("PyTorch가 필요합니다. pip install torch") from e

# VideoPose3D 레포 경로가 sys.path에 포함되어 있어야 합니다.
try:
    from common.model import TemporalModel  # VideoPose3D
except ImportError as e:
    raise RuntimeError(
        "VideoPose3D 임포트 실패. 레포 경로를 sys.path에 추가했는지 확인하세요.\n"
        "예: git clone https://github.com/facebookresearch/VideoPose3D.git\n"
        "    sys.path.append('path/to/VideoPose3D')"
    ) from e

logger = logging.getLogger(__name__)

# COCO17 인덱스
L_SHO, R_SHO = 5, 6
L_HIP, R_HIP = 11, 12

# 핵심 키포인트 집합(필요 시 활용)
TORSO_KEYPOINTS = [L_SHO, R_SHO, L_HIP, R_HIP]
CORE_KEYPOINTS = [0, L_SHO, R_SHO, 7, 8, 9, 10, L_HIP, R_HIP, 13, 14, 15, 16]

@dataclass
class Pose3DConfig:
    ckpt_path: str = "/content/pretrained_h36m_cpn.bin"
    filter_widths: List[int] = None   # [3,3,3,3,3] 기본
    causal: bool = False
    device: Optional[str] = None      # None이면 자동 선택(cuda > mps > cpu)
    min_scale_2d: float = 1e-6
    normalize_eps: float = 1e-8
    confidence_threshold: float = 0.1
    max_sequence_length: int = 1000
    batch_size: int = 1
    temporal_smoothing: bool = True
    # 향후 확장 옵션들(청크 처리, 스무딩 커널 등)을 위한 예약
    # chunk_size: Optional[int] = None
    # smooth_kernel_max: int = 5

    def __post_init__(self):
        if self.filter_widths is None:
            self.filter_widths = [3, 3, 3, 3, 3]

class Pose3DError(RuntimeError):
    pass

def _safe_normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < eps:
        result = np.zeros_like(vec)
        if result.size > 0:
            result[0] = 1.0
        return result
    return vec / norm

def _pelvis_center_2d(P: np.ndarray) -> np.ndarray:
    return (P[L_HIP] + P[R_HIP]) / 2.0

def _shoulder_center_2d(P: np.ndarray) -> np.ndarray:
    return (P[L_SHO] + P[R_SHO]) / 2.0

def _validate_keypoint_quality(kps_2d: np.ndarray, confidences: Optional[np.ndarray] = None) -> Dict[str, float]:
    T, K, _ = kps_2d.shape
    stats = {
        'total_frames': T,
        'valid_keypoints_ratio': 0.0,
        'torso_stability': 0.0,
        'temporal_consistency': 0.0
    }
    try:
        if confidences is not None:
            valid_mask = confidences > 0.1
            stats['valid_keypoints_ratio'] = float(np.mean(valid_mask))

        torso_distances = []
        for t in range(T):
            pelvis = _pelvis_center_2d(kps_2d[t])
            shoulder = _shoulder_center_2d(kps_2d[t])
            dist = np.linalg.norm(shoulder - pelvis)
            if np.isfinite(dist) and dist > 0:
                torso_distances.append(dist)
        if len(torso_distances) > 1:
            torso_std = np.std(torso_distances)
            torso_mean = np.mean(torso_distances)
            stats['torso_stability'] = float(1.0 - min(1.0, torso_std / max(torso_mean, 1e-6)))

        if T > 1:
            frame_diffs = []
            for t in range(1, T):
                diff = np.mean(np.linalg.norm(kps_2d[t] - kps_2d[t-1], axis=1))
                if np.isfinite(diff):
                    frame_diffs.append(diff)
            if frame_diffs:
                avg_diff = np.mean(frame_diffs)
                stats['temporal_consistency'] = float(1.0 / (1.0 + avg_diff / 10.0))
    except Exception as e:
        logger.warning(f"Quality assessment failed: {e}")
    return stats

def _normalize_2d_sequence(
    kps_2d_seq: np.ndarray,
    confidences: Optional[np.ndarray] = None,
    min_scale: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    T = int(kps_2d_seq.shape[0])
    out = np.zeros_like(kps_2d_seq, dtype=np.float32)
    scales = np.zeros((T,), dtype=np.float32)
    pelvises = np.zeros((T, 2), dtype=np.float32)

    quality_stats = _validate_keypoint_quality(kps_2d_seq, confidences)
    valid_scales: List[float] = []

    for t in range(T):
        P = kps_2d_seq[t].astype(np.float32)
        pelvis = _pelvis_center_2d(P)
        shoulder_c = _shoulder_center_2d(P)

        P_centered = P - pelvis
        scale = float(np.linalg.norm(shoulder_c - pelvis))

        if not np.isfinite(scale) or scale < min_scale:
            if valid_scales:
                scale = float(np.median(valid_scales))
            else:
                scale = 1.0
        else:
            valid_scales.append(scale)
            if len(valid_scales) > 5:
                median_scale = float(np.median(valid_scales))
                if scale > 3 * median_scale or scale < median_scale / 3:
                    scale = median_scale

        out[t] = P_centered / max(scale, min_scale)
        scales[t] = scale
        pelvises[t] = pelvis

    # 스케일 시계열 스무딩
    if len(valid_scales) > 3:
        kernel_size = min(5, max(3, T // 3))
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size >= 3:
            scales_tensor = torch.from_numpy(scales).unsqueeze(0).unsqueeze(0).float()
            kernel = torch.ones(1, 1, kernel_size) / kernel_size
            scales_smooth = F.conv1d(scales_tensor, kernel, padding=kernel_size//2)
            scales = scales_smooth.squeeze().numpy()
            for t in range(T):
                if scales[t] > min_scale:
                    P_centered = kps_2d_seq[t] - pelvises[t]
                    out[t] = P_centered / scales[t]

    quality_stats.update({
        'scale_mean': float(np.mean(scales)),
        'scale_std': float(np.std(scales)),
        'scale_consistency': float(1.0 - np.std(scales) / max(np.mean(scales), 1e-6))
    })
    return out, scales, pelvises, quality_stats

def _normalize_3d_meaning(
    kps_3d_seq: np.ndarray,
    eps: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = int(kps_3d_seq.shape[0])
    out = np.zeros_like(kps_3d_seq, dtype=np.float32)
    R_list: List[np.ndarray] = []
    S_list: List[float] = []

    for t in range(T):
        P = kps_3d_seq[t].astype(np.float64)
        pelvis = (P[L_HIP] + P[R_HIP]) / 2.0
        shoulder = (P[L_SHO] + P[R_SHO]) / 2.0

        P_centered = P - pelvis

        x_vec = P[R_HIP] - P[L_HIP]
        x_axis = _safe_normalize(x_vec, eps)

        up_vec = shoulder - pelvis
        up_axis = _safe_normalize(up_vec, eps)

        z_axis = up_axis - np.dot(up_axis, x_axis) * x_axis
        z_axis = _safe_normalize(z_axis, eps)

        y_axis = np.cross(z_axis, x_axis)
        y_axis = _safe_normalize(y_axis, eps)

        R = np.stack([x_axis, y_axis, z_axis], axis=0).astype(np.float64)
        P_rot = (R @ P_centered.T).T

        shoulder_rot = (R @ (shoulder - pelvis))
        scale = float(np.linalg.norm(shoulder_rot) + eps)

        P_norm = P_rot / max(scale, eps)

        out[t] = P_norm.astype(np.float32)
        R_list.append(R.astype(np.float32))
        S_list.append(scale)

    return out, np.array(R_list, dtype=np.float32), np.array(S_list, dtype=np.float32)

def _receptive_field(filter_widths: List[int]) -> int:
    rf = 1
    for fw in filter_widths:
        rf += (fw - 1)
    return rf

def _pad_sequence_reflect(x: np.ndarray, pad: int) -> np.ndarray:
    if pad <= 0:
        return x
    T = x.shape[0]
    if T == 0:
        return x
    if T >= pad:
        left = np.flip(x[:pad], axis=0)
        right = np.flip(x[-pad:], axis=0)
    else:
        n_repeats = (pad + T - 1) // T
        extended = np.tile(x, (n_repeats, 1, 1))
        left = np.flip(extended[:pad], axis=0)
        right = np.flip(extended[-pad:], axis=0)
    return np.concatenate([left, x, right], axis=0)

def _extend_temporally_to_min_length(norm_2d: np.ndarray, min_len: int) -> np.ndarray:
    T = int(norm_2d.shape[0])
    if T >= min_len:
        return norm_2d
    if T == 0:
        raise Pose3DError("Empty sequence cannot be extended")
    need = min_len - T
    max_pad = T
    left_pad = min(need // 2, max_pad)
    right_pad = min(need - left_pad, max_pad)
    left = np.flip(norm_2d[:left_pad], axis=0) if left_pad > 0 else np.empty((0, *norm_2d.shape[1:]))
    right = np.flip(norm_2d[-right_pad:], axis=0) if right_pad > 0 else np.empty((0, *norm_2d.shape[1:]))
    extended = np.concatenate([left, norm_2d, right], axis=0)
    if extended.shape[0] < min_len:
        n_repeats = (min_len - extended.shape[0] + T - 1) // T
        repeats = [norm_2d for _ in range(n_repeats)]
        extended = np.concatenate([extended] + repeats, axis=0)
    return extended[:min_len]

def _auto_select_device(device_hint: Optional[str]) -> str:
    if device_hint is not None:
        return device_hint
    try:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

def _load_temporal_model(config: Pose3DConfig) -> TemporalModel:
    if not os.path.exists(config.ckpt_path):
        raise Pose3DError(f"체크포인트 파일을 찾을 수 없습니다: {config.ckpt_path}")

    device = _auto_select_device(config.device)
    try:
        model = TemporalModel(
            num_joints_in=17,
            in_features=2,
            num_joints_out=17,
            filter_widths=config.filter_widths,
            causal=config.causal
        )
        logger.info(f"Loading checkpoint from {config.ckpt_path}")
        checkpoint = torch.load(config.ckpt_path, map_location="cpu")
        if "model_pos" in checkpoint:
            state_dict = checkpoint["model_pos"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device).eval()
        logger.info(f"Model loaded successfully on {device}")
        return model
    except Exception as e:
        raise Pose3DError(f"모델 로드 실패: {e}") from e

def _infer_center_crop(
    model: TemporalModel,
    norm_2d: np.ndarray,
    RF: int,
    device: str,
    batch_size: int = 1
) -> np.ndarray:
    norm_2d_safe = _extend_temporally_to_min_length(norm_2d, RF)
    pad = (RF - 1) // 2
    X = _pad_sequence_reflect(norm_2d_safe, pad)
    T_prime = norm_2d_safe.shape[0]
    out = np.zeros((T_prime, 17, 3), dtype=np.float32)

    with torch.no_grad():
        X_tensor = torch.from_numpy(X).float().to(device)
        for batch_start in range(0, T_prime, batch_size):
            batch_end = min(batch_start + batch_size, T_prime)
            batch_outputs: List[np.ndarray] = []
            for i in range(batch_start, batch_end):
                s, e = i, i + 2 * pad + 1
                window = X_tensor[s:e].unsqueeze(0)
                try:
                    y = model(window)  # (1, t_out, 17, 3)
                    y_np = y.detach().cpu().numpy()
                    t_out = y_np.shape[1]
                    center_idx = min((t_out - 1) // 2, pad)
                    batch_outputs.append(y_np[0, center_idx])
                except Exception as e:
                    logger.warning(f"Inference failed for frame {i}: {e}")
                    if i > 0:
                        batch_outputs.append(out[i-1].copy())
                    else:
                        batch_outputs.append(np.zeros((17, 3), dtype=np.float32))
            for idx, output in enumerate(batch_outputs):
                out[batch_start + idx] = output
    return out

def run_vpose3d(
    kps_pack: Dict[str, Any],
    config: Optional[Pose3DConfig] = None
) -> Dict[str, Any]:
    if config is None:
        config = Pose3DConfig()

    start_time = time.time()

    if "kps_2d" not in kps_pack:
        raise Pose3DError("kps_pack에 'kps_2d' 키가 없습니다.")

    kps_2d = np.asarray(kps_pack["kps_2d"], dtype=np.float32)
    if kps_2d.ndim != 3 or kps_2d.shape[1:] != (17, 2):
        raise Pose3DError(f"kps_2d 형태가 (T,17,2)가 아닙니다: {kps_2d.shape}")

    T_original = kps_2d.shape[0]
    if T_original == 0:
        raise Pose3DError("빈 시퀀스입니다.")
    if T_original > config.max_sequence_length:
        logger.warning(f"Long sequence ({T_original} frames) may be slow")

    confidences = kps_pack.get("conf", None)
    if confidences is not None:
        confidences = np.asarray(confidences, dtype=np.float32)

    RF = _receptive_field(config.filter_widths)
    PAD = (RF - 1) // 2

    device = _auto_select_device(config.device)
    logger.info(f"Processing {T_original} frames with RF={RF} on {device}")

    norm_2d, scale_2d, pelvis_2d, quality_stats = _normalize_2d_sequence(
        kps_2d, confidences, config.min_scale_2d
    )

    if quality_stats.get('valid_keypoints_ratio', 1.0) < 0.5:
        logger.warning(f"Low keypoint quality: {quality_stats['valid_keypoints_ratio']:.2f}")

    model = _load_temporal_model(config)

    inference_start = time.time()
    y3d_raw = _infer_center_crop(
        model, norm_2d, RF=RF, device=device, batch_size=config.batch_size
    )
    inference_time = time.time() - inference_start

    T_used = y3d_raw.shape[0]

    pose3d_meaning, R_seq, S_seq = _normalize_3d_meaning(y3d_raw, config.normalize_eps)

    if config.temporal_smoothing and T_used > 3:
        try:
            kernel_size = min(5, max(3, T_used // 5))
            if kernel_size % 2 == 0:
                kernel_size += 1
            pose_tensor = torch.from_numpy(pose3d_meaning).permute(1, 2, 0).unsqueeze(0)  # (1,17,3,T)
            kernel = torch.ones(1, 1, kernel_size) / kernel_size
            smoothed = F.conv1d(pose_tensor.reshape(1, -1, T_used), kernel,
                                padding=kernel_size//2, groups=17*3)
            pose3d_meaning = smoothed.reshape(1, 17, 3, T_used).squeeze(0).permute(2, 0, 1).numpy()
        except Exception as e:
            logger.warning(f"Temporal smoothing failed: {e}")

    total_time = time.time() - start_time

    quality_flag = bool(
        (quality_stats.get('valid_keypoints_ratio', 1.0) < 0.5) or
        (quality_stats.get('scale_consistency', 1.0) < 0.5)
    )

    result = {
        "pose3d": pose3d_meaning.astype(np.float32),
        "pose2d_norm": norm_2d.astype(np.float32),
        "R_seq": R_seq.astype(np.float32),
        "S_seq": S_seq.astype(np.float32),
        "meta": {
            "RF": int(RF),
            "PAD": int(PAD),
            "device": device,
            "T_original": int(T_original),
            "T_used": int(T_used),
            "filter_widths": config.filter_widths,
            "total_time": total_time,
            "inference_time": inference_time,
            "fps": T_used / max(total_time, 0.001),
            "quality_flag": quality_flag,
            **quality_stats
        }
    }

    logger.info(f"3D pose estimation completed: {T_used} frames in {total_time:.2f}s "
                f"(inference: {inference_time:.2f}s, quality: {quality_stats.get('valid_keypoints_ratio', 0.0):.3f})")
    return result

def run_vpose3d_fast(kps_pack: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    config = Pose3DConfig(
        filter_widths=[3, 3, 3],
        batch_size=4,
        temporal_smoothing=False,
        **kwargs
    )
    return run_vpose3d(kps_pack, config)

def run_vpose3d_accurate(kps_pack: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    config = Pose3DConfig(
        filter_widths=[3, 3, 3, 3, 3, 3],
        batch_size=1,
        temporal_smoothing=True,
        **kwargs
    )
    return run_vpose3d(kps_pack, config)
