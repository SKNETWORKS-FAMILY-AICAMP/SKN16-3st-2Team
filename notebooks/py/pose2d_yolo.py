import os
import logging
import time
from typing import Dict, Any, Optional, Callable, Tuple, List
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# 의존성 체크
try:
    import cv2
except ImportError as e:
    raise RuntimeError("OpenCV not found. Install with: pip install opencv-python") from e

try:
    from ultralytics import YOLO
except ImportError as e:
    raise RuntimeError("ultralytics not found. Install with: pip install ultralytics") from e

logger = logging.getLogger(__name__)

COCO17_NAMES: List[str] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]
KEYPOINT_INDICES = {name: i for i, name in enumerate(COCO17_NAMES)}
SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}

@dataclass
class Pose2DConfig:
    model_name: str = "yolo11m-pose.pt"
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    fps_fallback: float = 30.0
    stride: int = 1
    resize_short: Optional[int] = None
    max_persons: int = 10
    target_keypoints: int = 17
    device: str = "auto"   # "cpu", "cuda", "mps", "auto"
    verbose: bool = False
    flip_test: bool = False
    max_video_size_mb: float = 100.0
    max_duration_sec: float = 60.0
    min_confidence: float = 0.1
    min_duration_sec: float = 1.0         # 추가: 너무 짧은 영상 방지
    max_process_seconds: Optional[float] = None  # 추가: 전체 처리 타임아웃
    imgsz: Optional[int] = None           # 추가: YOLO 입력 크기
    use_half: Optional[bool] = None       # 추가: half precision 시도(가능 시)

class Pose2DError(RuntimeError):
    pass

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) and v >= 0 else default
    except (ValueError, TypeError):
        return default

def _select_best_person(
    keypoints: np.ndarray,
    confidences: np.ndarray,
    method: str = "weighted_score"
) -> int:
    if keypoints is None or confidences is None:
        return -1
    if len(keypoints) == 0 or len(confidences) == 0:
        return -1
    try:
        n_persons = len(keypoints)
        if n_persons == 1:
            return 0
        scores = []
        for i in range(n_persons):
            kpts = keypoints[i]
            confs = confidences[i]
            avg_conf = np.mean(confs) if confs.size else 0.0
            core_indices = [
                KEYPOINT_INDICES["left_shoulder"], KEYPOINT_INDICES["right_shoulder"],
                KEYPOINT_INDICES["left_hip"], KEYPOINT_INDICES["right_hip"],
                KEYPOINT_INDICES["left_knee"], KEYPOINT_INDICES["right_knee"]
            ]
            core_vals = [confs[idx] for idx in core_indices if idx < len(confs)]
            core_conf = np.mean(core_vals) if len(core_vals) else 0.0
            hip_score = 0.0
            if len(kpts) > max(KEYPOINT_INDICES["left_hip"], KEYPOINT_INDICES["right_hip"]):
                left_hip = kpts[KEYPOINT_INDICES["left_hip"]]
                right_hip = kpts[KEYPOINT_INDICES["right_hip"]]
                left_hip_conf = confs[KEYPOINT_INDICES["left_hip"]] if len(confs) > KEYPOINT_INDICES["left_hip"] else 0.0
                right_hip_conf = confs[KEYPOINT_INDICES["right_hip"]] if len(confs) > KEYPOINT_INDICES["right_hip"] else 0.0
                if left_hip_conf > 0.3 and right_hip_conf > 0.3:
                    hip_dist = np.linalg.norm(left_hip - right_hip)
                    if 20 <= hip_dist <= 200:
                        hip_score = min(1.0, hip_dist / 100.0)
            weighted_score = (0.4 * avg_conf + 0.4 * core_conf + 0.2 * hip_score)
            scores.append(weighted_score)
        scores = np.array(scores)
        if scores.size == 0 or np.all(scores <= 0):
            return -1
        best_idx = int(np.argmax(scores))
        return best_idx if np.isfinite(scores[best_idx]) and scores[best_idx] > 0.1 else -1
    except Exception as e:
        logger.warning(f"Person selection failed: {e}")
        return -1

def _ensure_keypoint_format(
    keypoints: np.ndarray,
    confidences: np.ndarray,
    target_kpts: int = 17
) -> Tuple[np.ndarray, np.ndarray]:
    kpts = np.asarray(keypoints, dtype=np.float32)
    confs = np.asarray(confidences, dtype=np.float32)
    kpts_out = np.zeros((target_kpts, 2), dtype=np.float32)
    confs_out = np.zeros(target_kpts, dtype=np.float32)
    if kpts.ndim == 2 and kpts.shape[1] == 2:
        valid_kpts = min(kpts.shape[0], target_kpts)
        kpts_out[:valid_kpts] = kpts[:valid_kpts]
    else:
        logger.warning(f"Invalid keypoint shape: {kpts.shape}")
    if confs.ndim == 1:
        valid_confs = min(confs.shape[0], target_kpts)
        confs_out[:valid_confs] = confs[:valid_confs]
    else:
        logger.warning(f"Invalid confidence shape: {confs.shape}")
    kpts_out[~np.isfinite(kpts_out)] = 0.0
    confs_out[~np.isfinite(confs_out)] = 0.0
    confs_out = np.clip(confs_out, 0.0, 1.0)
    return kpts_out, confs_out

def _validate_video_file(video_path: str, config: Pose2DConfig) -> Dict[str, Any]:
    path = Path(video_path)
    if not path.exists():
        raise Pose2DError(f"Video file not found: {video_path}")
    if not path.is_file():
        raise Pose2DError(f"Path is not a file: {video_path}")
    if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        logger.warning(f"Unsupported video extension: {path.suffix}")
    file_size = path.stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    if file_size_mb > config.max_video_size_mb:
        raise Pose2DError(f"Video file too large: {file_size_mb:.1f}MB > {config.max_video_size_mb}MB")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise Pose2DError(f"Cannot open video with OpenCV: {video_path}")
    try:
        fps = _safe_float(cap.get(cv2.CAP_PROP_FPS), config.fps_fallback)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0.0

        if width <= 0 or height <= 0:
            raise Pose2DError(f"Invalid video dimensions: {width}x{height}")
        if frame_count <= 0:
            raise Pose2DError(f"No frames in video: {frame_count}")
        if duration > config.max_duration_sec:
            raise Pose2DError(f"Video too long: {duration:.1f}s > {config.max_duration_sec}s")
        if duration < config.min_duration_sec:
            raise Pose2DError(f"Video too short: {duration:.2f}s < {config.min_duration_sec:.2f}s")

        if fps < 5 or fps > 120:
            logger.warning(f"Suspicious FPS: {fps:.1f} (range 5~120 recommended)")
        if width < 224 or height < 224:
            logger.warning(f"Low resolution video: {width}x{height}")
        elif width > 1920 or height > 1080:
            logger.info(f"High resolution video: {width}x{height} (may be slow)")

        return {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration": duration,
            "file_size": file_size,
            "file_size_mb": file_size_mb
        }
    finally:
        cap.release()

_MODEL_CACHE: Dict[Tuple[str, str], YOLO] = {}

def _auto_select_device() -> str:
    # 간단한 자동 선택: CUDA > MPS > CPU
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        return "cpu"

def _load_yolo_model(model_name: str, device: str = "auto", imgsz: Optional[int] = None, use_half: Optional[bool] = None) -> YOLO:
    try:
        if not model_name.endswith('.pt'):
            model_name += '.pt'
        if device == "auto":
            device = _auto_select_device()
        cache_key = (model_name, device)
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]

        model = YOLO(model_name)
        model.to(device)

        # half precision 옵션 시도
        if use_half is None:
            use_half = False
            try:
                import torch
                if device == "cuda" and torch.cuda.is_available():
                    use_half = True
            except Exception:
                use_half = False

        # 더미 추론으로 검증
        test_img = np.zeros((imgsz or 320, imgsz or 320, 3), dtype=np.uint8)
        _ = model(test_img, verbose=False)

        _MODEL_CACHE[cache_key] = model
        logger.info(f"YOLO model loaded successfully: {model_name} on {device} (imgsz={imgsz}, half={use_half})")
        return model
    except Exception as e:
        raise Pose2DError(f"Failed to load YOLO model '{model_name}': {e}") from e

def _horizontal_flip_kpts_conf(kpts: np.ndarray, conf: np.ndarray, width: int) -> Tuple[np.ndarray, np.ndarray]:
    # 좌우 반전: x -> width - x, 그리고 좌우 관절 스왑
    flipped = kpts.copy()
    flipped[:, 0] = (width - 1) - flipped[:, 0]
    # 좌우 쌍 스왑 인덱스
    pairs = [
        ("left_eye", "right_eye"),
        ("left_ear", "right_ear"),
        ("left_shoulder", "right_shoulder"),
        ("left_elbow", "right_elbow"),
        ("left_wrist", "right_wrist"),
        ("left_hip", "right_hip"),
        ("left_knee", "right_knee"),
        ("left_ankle", "right_ankle"),
    ]
    idx = {n: KEYPOINT_INDICES[n] for n in KEYPOINT_INDICES}
    for l, r in pairs:
        li, ri = idx[l], idx[r]
        flipped[[li, ri]] = flipped[[ri, li]]
        conf[[li, ri]] = conf[[ri, li]]
    return flipped, conf

def _mean_confidence(cf: np.ndarray) -> float:
    vals = cf[cf > 0]
    return float(np.mean(vals)) if vals.size > 0 else 0.0

def run_yolo2d(
    video_path: str,
    config: Optional[Pose2DConfig] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    if config is None:
        config = Pose2DConfig()

    start_time = time.time()
    video_meta = _validate_video_file(video_path, config)
    logger.info(f"Processing video: {video_meta['width']}x{video_meta['height']} "
                f"@ {video_meta['fps']:.1f}fps, {video_meta['duration']:.1f}s")

    model = _load_yolo_model(config.model_name, config.device, imgsz=config.imgsz, use_half=config.use_half)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Pose2DError("Failed to open video after validation")

    try:
        kps_list: List[np.ndarray] = []
        conf_list: List[np.ndarray] = []
        frames: List[int] = []
        processing_times: List[float] = []
        fail_count = 0
        max_consecutive_fail = 0
        consecutive_fail = 0

        frame_idx = -1
        out_count = 0
        target_total = max(1, video_meta['frame_count'] // max(1, config.stride))

        while True:
            if config.max_process_seconds is not None and (time.time() - start_time) > config.max_process_seconds:
                logger.warning("Processing aborted due to max_process_seconds timeout.")
                break

            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            if config.stride > 1 and (frame_idx % config.stride != 0):
                continue

            frame_start = time.time()

            if config.resize_short and config.resize_short > 0:
                h, w = frame.shape[:2]
                short_side = min(h, w)
                if short_side > config.resize_short:
                    scale = config.resize_short / float(short_side)
                    new_w, new_h = int(round(w * scale)), int(round(h * scale))
                    frame_in = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    coord_scale = 1.0 / scale
                else:
                    frame_in = frame
                    coord_scale = 1.0
            else:
                frame_in = frame
                coord_scale = 1.0

            try:
                results = model(frame_in, conf=config.conf_thres, iou=config.iou_thres, verbose=False)
                if not results or len(results) == 0:
                    raise RuntimeError("Empty YOLO results")

                det = results[0]
                has_kpts = hasattr(det, 'keypoints') and det.keypoints is not None and hasattr(det.keypoints, 'xy')
                if has_kpts and len(det.keypoints.xy) > 0:
                    xy = np.asarray(det.keypoints.xy, dtype=np.float32)      # (N,K,2)
                    conf_all = np.asarray(det.keypoints.conf, dtype=np.float32)  # (N,K)

                    if coord_scale != 1.0:
                        xy *= coord_scale

                    # flip_test 옵션: 원본 vs 좌우반전 추론 결과 중 평균 신뢰도 높은 쪽 선택
                    if config.flip_test:
                        # 좌우 반전 프레임 생성
                        frame_flipped = cv2.flip(frame_in, 1)
                        results_flip = model(frame_flipped, conf=config.conf_thres, iou=config.iou_thres, verbose=False)
                        det_f = results_flip[0] if results_flip and len(results_flip) > 0 else None
                        if det_f is not None and hasattr(det_f, 'keypoints') and det_f.keypoints is not None and hasattr(det_f.keypoints, 'xy') and len(det_f.keypoints.xy) > 0:
                            xy_f = np.asarray(det_f.keypoints.xy, dtype=np.float32)
                            conf_f = np.asarray(det_f.keypoints.conf, dtype=np.float32)

                            if coord_scale != 1.0:
                                xy_f *= coord_scale

                            # 반전 좌표를 원본 좌표계로 복원
                            # frame_in의 width로 좌우 반전 역변환
                            h_in, w_in = frame_in.shape[:2]
                            xy_f[..., 0] = (w_in - 1) - xy_f[..., 0]
                            # 좌우 관절 스왑
                            # 사람 단위로 좌우 스왑 적용
                            for i in range(xy_f.shape[0]):
                                ktmp, ctmp = _ensure_keypoint_format(xy_f[i], conf_f[i], target_kpts=config.target_keypoints)
                                ktmp, ctmp = _horizontal_flip_kpts_conf(ktmp, ctmp, width=w_in)
                                xy_f[i] = ktmp
                                conf_f[i] = ctmp

                            # 원본/반전 각각 최적 인물 선택 후 평균 신뢰도로 비교
                            idx_o = _select_best_person(xy, conf_all)
                            idx_f = _select_best_person(xy_f, conf_f)

                            if idx_o >= 0:
                                kps_o, cf_o = _ensure_keypoint_format(xy[idx_o], conf_all[idx_o], target_kpts=config.target_keypoints)
                                mean_o = _mean_confidence(cf_o)
                            else:
                                kps_o = np.zeros((config.target_keypoints, 2), dtype=np.float32)
                                cf_o = np.zeros(config.target_keypoints, dtype=np.float32)
                                mean_o = 0.0
                            if idx_f >= 0:
                                kps_f, cf_f = _ensure_keypoint_format(xy_f[idx_f], conf_f[idx_f], target_kpts=config.target_keypoints)
                                mean_f = _mean_confidence(cf_f)
                            else:
                                kps_f = np.zeros((config.target_keypoints, 2), dtype=np.float32)
                                cf_f = np.zeros(config.target_keypoints, dtype=np.float32)
                                mean_f = 0.0

                            if mean_f > mean_o:
                                kps, cf = kps_f, cf_f
                            else:
                                kps, cf = kps_o, cf_o
                        else:
                            # 반전 추론 실패 시 원본만 사용
                            person_idx = _select_best_person(xy, conf_all)
                            if person_idx >= 0:
                                kps, cf = _ensure_keypoint_format(xy[person_idx], conf_all[person_idx], target_kpts=config.target_keypoints)
                            else:
                                kps = np.zeros((config.target_keypoints, 2), dtype=np.float32)
                                cf = np.zeros(config.target_keypoints, dtype=np.float32)
                    else:
                        person_idx = _select_best_person(xy, conf_all)
                        if person_idx >= 0:
                            kps, cf = _ensure_keypoint_format(xy[person_idx], conf_all[person_idx], target_kpts=config.target_keypoints)
                        else:
                            kps = np.zeros((config.target_keypoints, 2), dtype=np.float32)
                            cf = np.zeros(config.target_keypoints, dtype=np.float32)
                else:
                    kps = np.zeros((config.target_keypoints, 2), dtype=np.float32)
                    cf = np.zeros(config.target_keypoints, dtype=np.float32)

            except Exception as e:
                logger.warning(f"YOLO inference failed at frame {frame_idx}: {e}")
                kps = np.zeros((config.target_keypoints, 2), dtype=np.float32)
                cf = np.zeros(config.target_keypoints, dtype=np.float32)

            # 실패 통계
            if np.all(cf <= 0):
                fail_count += 1
                consecutive_fail += 1
            else:
                max_consecutive_fail = max(max_consecutive_fail, consecutive_fail)
                consecutive_fail = 0

            kps_list.append(kps)
            conf_list.append(cf)
            frames.append(frame_idx)
            processing_times.append(time.time() - frame_start)
            out_count += 1

            if progress_cb and target_total > 0:
                try:
                    progress_cb(out_count, target_total)
                except Exception as e:
                    logger.warning(f"Progress callback failed: {e}")

        if len(kps_list) == 0:
            logger.warning("No frames processed - creating dummy output")
            kps_list = [np.zeros((config.target_keypoints, 2), dtype=np.float32)]
            conf_list = [np.zeros(config.target_keypoints, dtype=np.float32)]
            frames = [0]
            max_consecutive_fail = 1
            fail_count = 1

        total_time = time.time() - start_time
        valid_conf_means = [np.mean(cf[cf > 0]) for cf in conf_list if np.any(cf > 0)]
        avg_conf = np.mean(valid_conf_means) if len(valid_conf_means) else 0.0

        result = {
            'kps_2d': np.stack(kps_list, axis=0).astype(np.float32),
            'conf': np.stack(conf_list, axis=0).astype(np.float32),
            'frames': np.array(frames, dtype=np.int32),
            'meta': {
                'width': video_meta['width'],
                'height': video_meta['height'],
                'fps': video_meta['fps'],
                'duration': video_meta['duration'],
                'frame_count': video_meta['frame_count'],
                'skeleton': 'coco17',
                'joint_names': COCO17_NAMES,
                'source': Path(video_path).name,
                'model': config.model_name,
                'device': config.device,
                'stride': config.stride,
                'resize_short': config.resize_short,
                'processing_time': total_time,
                'fps_processed': len(kps_list) / max(total_time, 0.001),
                'avg_frame_time': float(np.mean(processing_times)) if processing_times else 0.0,
                'avg_confidence': float(avg_conf) if np.isfinite(avg_conf) else 0.0,
                'valid_frames': len(kps_list),
                'fail_frames': int(fail_count),
                'fail_ratio': float(fail_count / max(1, len(kps_list))),
                'max_consecutive_fail': int(max_consecutive_fail),
            }
        }
        logger.info(f"Pose extraction completed: {len(kps_list)} frames in {total_time:.2f}s "
                    f"(avg conf: {avg_conf:.3f}, fail_ratio: {result['meta']['fail_ratio']:.2f})")
        return result

    finally:
        try:
            cap.release()
        except Exception:
            pass

def run_yolo2d_fast(video_path: str, **kwargs) -> Dict[str, Any]:
    config = Pose2DConfig(
        model_name="yolo11n-pose.pt",
        stride=2,
        resize_short=320,
        imgsz=320,
        **kwargs
    )
    return run_yolo2d(video_path, config)

def run_yolo2d_accurate(video_path: str, **kwargs) -> Dict[str, Any]:
    config = Pose2DConfig(
        model_name="yolo11x-pose.pt",
        conf_thres=0.1,
        stride=1,
        flip_test=True,
        resize_short=None,
        imgsz=640,
        **kwargs
    )
    return run_yolo2d(video_path, config)
