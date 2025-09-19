# simple_pose2d.py

import time, json, logging
from pathlib import Path
from typing import List
import numpy as np
import cv2

try:
    from ultralytics import YOLO
except ImportError as e:
    raise RuntimeError("ultralytics가 필요합니다. 설치: pip install ultralytics") from e

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simple_pose2d_optimized")

COCO17_NAMES = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]

def _auto_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

def _validate_video(video_path: str):
    p = Path(video_path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_path}")
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise ValueError(f"비디오를 열 수 없습니다: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if min(fps, frame_count, width, height) <= 0:
            raise ValueError(f"영상 메타 비정상: fps={fps}, frames={frame_count}, size={width}x{height}")
        return {"fps": fps, "frame_count": frame_count, "width": width, "height": height}
    finally:
        cap.release()

def extract_2d_poses(
    video_path: str,
    output_path: str = "coco17_2d.npz",
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    stride: int = 1,
    resize_short: int = 0,
    lowconf_mask: float = 0.3
) -> str:
    meta_vid = _validate_video(video_path)
    device = _auto_device()
    model_name = "yolo11m-pose.pt"
    logger.info(f"비디오 정보: {meta_vid['width']}x{meta_vid['height']}, {meta_vid['fps']:.1f}fps, {meta_vid['frame_count']}프레임")
    logger.info(f"모델 로드: {model_name} on {device}")

    model = YOLO(model_name)
    try:
        if device != "cpu":
            model.to(device)
    except Exception:
        logger.warning("모델 디바이스 이동 실패, CPU로 진행합니다.")
    _ = model(np.zeros((320,320,3), dtype=np.uint8), verbose=False)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("검증 후 캡처 오픈 실패")

    kps_list: List[np.ndarray] = []
    conf_list: List[np.ndarray] = []
    frames: List[int] = []

    t0 = time.time()
    frame_idx = -1
    processed = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if stride > 1 and (frame_idx % stride != 0):
                continue

            frame_in = frame
            coord_scale = 1.0
            if resize_short and resize_short > 0:
                h, w = frame.shape[:2]
                short_side = min(h, w)
                if short_side > resize_short:
                    scale = resize_short / float(short_side)
                    new_w, new_h = int(round(w * scale)), int(round(h * scale))
                    frame_in = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    coord_scale = 1.0 / scale

            try:
                results = model(frame_in, conf=conf_thres, iou=iou_thres, verbose=False)
                det = results[0] if results and len(results) > 0 else None

                if det is not None and hasattr(det, "keypoints") and det.keypoints is not None and hasattr(det.keypoints, "xy") and len(det.keypoints.xy) > 0:
                    xy_all = np.asarray(det.keypoints.xy, dtype=np.float32)    # (N,17,2)
                    cf_all = np.asarray(det.keypoints.conf, dtype=np.float32)  # (N,17)
                    if coord_scale != 1.0:
                        xy_all *= coord_scale
                    means = [float(np.mean(c)) for c in cf_all]
                    pidx = int(np.argmax(means))
                    kps = xy_all[pidx]
                    cf = cf_all[pidx]
                    if lowconf_mask is not None and lowconf_mask > 0:
                        mask = (cf < float(lowconf_mask))
                        if np.any(mask):
                            kps[mask] = 0.0
                            cf[mask] = 0.0
                else:
                    kps = np.zeros((17,2), np.float32)
                    cf = np.zeros((17,), np.float32)
            except Exception as e:
                logger.warning(f"YOLO 추론 실패 frame={frame_idx}: {e}")
                kps = np.zeros((17,2), np.float32)
                cf = np.zeros((17,), np.float32)

            kps_list.append(kps.astype(np.float32))
            conf_list.append(cf.astype(np.float32))
            frames.append(frame_idx)
            processed += 1

            if processed % 100 == 0:
                logger.info(f"처리 중... {processed}/{meta_vid['frame_count']}")

    finally:
        cap.release()

    if not kps_list:
        raise RuntimeError("처리된 프레임이 없습니다.")

    kps_2d = np.stack(kps_list, axis=0).astype(np.float32)  # (T,17,2)
    conf = np.stack(conf_list, axis=0).astype(np.float32)   # (T,17)
    frames_arr = np.array(frames, dtype=np.int32)

    logger.info(f"추출 완료: {kps_2d.shape[0]} 프레임, 소요 {time.time()-t0:.2f}s")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "video": Path(video_path).name,
        "width": meta_vid["width"],
        "height": meta_vid["height"],
        "fps": meta_vid["fps"],
        "frame_count": meta_vid["frame_count"],
        "model": model_name,
        "device": device,
        "stride": stride,
        "resize_short": resize_short,
        "processed_frames": int(len(frames_arr)),
        "elapsed_sec": time.time() - t0,
        "joint_names": COCO17_NAMES,
    }
    np.savez_compressed(
        str(out),
        kps_2d=kps_2d,
        conf=conf,
        frames=frames_arr,
        meta=json.dumps(meta, ensure_ascii=False)
    )
    logger.info(f"저장 완료: {out} | kps_2d={kps_2d.shape}, conf={conf.shape}")
    return str(out)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="YOLO11m → coco17_2d.npz 생성기(단일 인물)")
    ap.add_argument("--video", required=True, help="입력 비디오 경로")
    ap.add_argument("--out", default="coco17_2d.npz", help="출력 npz 경로")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="YOLO IoU threshold")
    ap.add_argument("--stride", type=int, default=1, help="프레임 스킵(성능용)")
    ap.add_argument("--resize_short", type=int, default=0, help="짧은 변 축소(0이면 비활성)")
    ap.add_argument("--lowconf_mask", type=float, default=0.3, help="이 값 미만 conf 관절 0 마스킹")
    args = ap.parse_args()
    extract_2d_poses(args.video, args.out, args.conf, args.iou, args.stride, args.resize_short, args.lowconf_mask)