import os
from typing import Dict, Any, Optional, Callable, Tuple, List
import numpy as np

try:
    import cv2
except Exception as e:
    raise RuntimeError("OpenCV not found. Install with: pip install opencv-python") from e

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError("ultralytics not found. Install with: pip install ultralytics") from e


COCO17_NAMES: List[str] = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]


class Pose2DError(RuntimeError):
    pass


def _safe_float(x, default: float) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) and v > 0 else default
    except Exception:
        return default


def _select_person(xy: np.ndarray, conf: np.ndarray) -> int:
    if xy is None or conf is None or len(xy) == 0 or len(conf) == 0:
        return -1
    try:
        scores = conf.mean(axis=1)
        if scores.size == 0:
            return -1
        idx = int(np.argmax(scores))
        return idx if np.isfinite(scores[idx]) else -1
    except Exception:
        return -1


def _ensure_shape_17(kps: np.ndarray, cf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    K = 17
    kps = np.asarray(kps, dtype=np.float32)
    cf = np.asarray(cf, dtype=np.float32)

    if kps.ndim != 2 or kps.shape[1] != 2:
        kps_out = np.zeros((K, 2), dtype=np.float32)
    else:
        if kps.shape[0] == K:
            kps_out = kps
        elif kps.shape[0] > K:
            kps_out = kps[:K]
        else:
            kps_out = np.zeros((K, 2), dtype=np.float32)
            k = min(kps.shape[0], K)
            kps_out[:k] = kps[:k]

    if cf.ndim != 1:
        cf_out = np.zeros((K,), dtype=np.float32)
    else:
        if cf.shape[0] == K:
            cf_out = cf
        elif cf.shape[0] > K:
            cf_out = cf[:K]
        else:
            cf_out = np.zeros((K,), dtype=np.float32)
            k = min(cf.shape[0], K)
            cf_out[:k] = cf[:k]

    kps_out[~np.isfinite(kps_out)] = 0.0
    cf_out[~np.isfinite(cf_out)] = 0.0
    cf_out = np.clip(cf_out, 0.0, 1.0)
    return kps_out, cf_out


def run_yolo2d(
    video_path: str,
    model_name: str = "yolo11m-pose.pt",
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    fps_fallback: float = 30.0,
    stride: int = 1,
    resize_short: Optional[int] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    if not os.path.exists(video_path):
        raise Pose2DError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Pose2DError("Failed to open video. Check codec/file integrity.")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = _safe_float(cap.get(cv2.CAP_PROP_FPS), fps_fallback)

        try:
            model = YOLO(model_name)
        except Exception as e:
            raise Pose2DError(f"Failed to load YOLO model: {e}")

        kps_list: List[np.ndarray] = []
        conf_list: List[np.ndarray] = []
        frames: List[int] = []

        frame_idx = -1
        out_count = 0
        target_total = total_frames // max(1, stride)

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if stride > 1 and (frame_idx % stride != 0):
                continue

            if resize_short and resize_short > 0:
                h, w = frame.shape[:2]
                short_side = min(h, w)
                scale = resize_short / float(short_side)
                new_w, new_h = int(round(w * scale)), int(round(h * scale))
                frame_in = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                frame_in = frame

            try:
                res = model(frame_in, conf=conf_thres, iou=iou_thres, verbose=False)
                kp = getattr(res[0], "keypoints", None)
                if kp is None or getattr(kp, "xy", None) is None or len(kp.xy) == 0:
                    kps = np.zeros((17, 2), dtype=np.float32)
                    cf = np.zeros((17,), dtype=np.float32)
                else:
                    xy = np.asarray(kp.xy, dtype=np.float32)
                    cf_all = np.asarray(kp.conf, dtype=np.float32)
                    pid = _select_person(xy, cf_all)
                    if pid < 0:
                        kps = np.zeros((17, 2), dtype=np.float32)
                        cf = np.zeros((17,), dtype=np.float32)
                    else:
                        kps, cf = _ensure_shape_17(xy[pid], cf_all[pid])
            except Exception:
                kps = np.zeros((17, 2), dtype=np.float32)
                cf = np.zeros((17,), dtype=np.float32)

            kps_list.append(kps)
            conf_list.append(cf)
            frames.append(frame_idx)
            out_count += 1

            if progress_cb and target_total > 0:
                try:
                    progress_cb(out_count, target_total)
                except Exception:
                    pass

        if len(kps_list) == 0:
            kps_list = [np.zeros((17, 2), dtype=np.float32)]
            conf_list = [np.zeros((17,), dtype=np.float32)]
            frames = [0]

        pack = dict(
            kps_2d=np.stack(kps_list, axis=0).astype(np.float32),
            conf=np.stack(conf_list, axis=0).astype(np.float32),
            frames=np.array(frames, dtype=np.int32),
            meta=dict(
                width=width, height=height, fps=float(fps),
                skeleton="coco17", joint_names=COCO17_NAMES,
                source=os.path.basename(video_path), model=model_name,
                stride=int(stride), resize_short=int(resize_short) if resize_short else None
            )
        )
        return pack
    finally:
        try:
            cap.release()
        except Exception:
            pass