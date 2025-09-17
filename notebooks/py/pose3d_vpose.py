# pose3d_vpose.py
# 목적:
#  - 전문가/사용자 공통 경로(2D→3D→의미좌표 정규화)로 안정적 3D 시퀀스를 생성
#  - 체크포인트 구조와 일치하는 모델을 사용하되, 짧은/경계 시퀀스에서도 conv1d 에러 없이 추론
#  - 비교 파이프라인에서 바로 사용 가능하도록 최소/필수 정보만 반환

import os
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

try:
    import torch
    import torch.nn.functional as F  # noqa: F401 (일부 환경에서 conv 관련 디버깅용)
except Exception as e:
    raise RuntimeError("PyTorch가 필요합니다. pip install torch") from e

# VideoPose3D 레포를 git clone 후, sys.path에 루트를 추가해야 합니다.
# 예) !git clone https://github.com/facebookresearch/VideoPose3D.git
#    import sys; sys.path.append("/content/VideoPose3D")
try:
    from common.model import TemporalModel  # VideoPose3D
except Exception as e:
    raise RuntimeError("VideoPose3D 임포트 실패. 레포 경로를 sys.path에 추가했는지 확인하세요.") from e


# COCO17 인덱스(어깨/엉덩이 중심 계산에 필요)
L_SHO, R_SHO = 5, 6
L_HIP, R_HIP = 11, 12


class Pose3DError(RuntimeError):
    pass


def _pelvis_center_2d(P: np.ndarray) -> np.ndarray:
    """2D 키포인트에서 골반 중심점을 계산합니다."""
    return (P[L_HIP] + P[R_HIP]) / 2.0


def _shoulder_center_2d(P: np.ndarray) -> np.ndarray:
    """2D 키포인트에서 어깨 중심점을 계산합니다."""
    return (P[L_SHO] + P[R_SHO]) / 2.0


def _pelvis_center_3d(P: np.ndarray) -> np.ndarray:
    """3D 키포인트에서 골반 중심점 (루트 조인트)을 계산합니다."""
    return (P[L_HIP] + P[R_HIP]) / 2.0


def _normalize_2d_sequence(kps_2d_seq: np.ndarray, min_scale: float = 1e-6) -> np.ndarray:
    """
    (T,17,2) → 각 프레임을 골반 중심 평행이동 후, 골반-어깨 거리로 스케일 정규화.
    반환: (T,17,2) float32
    """
    T = int(kps_2d_seq.shape[0])
    out = np.zeros_like(kps_2d_seq, dtype=np.float32)
    for t in range(T):
        P = kps_2d_seq[t].astype(np.float32)
        pelvis = _pelvis_center_2d(P)
        shoulder_c = _shoulder_center_2d(P)
        P = P - pelvis  # 골반 중심으로 이동
        scale = float(np.linalg.norm(shoulder_c - pelvis))  # 골반-어깨 거리로 스케일 계산
        if not np.isfinite(scale) or scale < min_scale: # 스케일이 비정상이거나 너무 작으면
            scale = 1.0  # 스케일 정규화 안함
        out[t] = P / scale
    return out


def _receptive_field(filter_widths: List[int]) -> int:
    """모델의 유효 수용장(receptive field) 크기를 계산합니다."""
    rf = 1
    for fw in filter_widths:
        rf += (fw - 1) * 2 # VideoPose3D 모델은 각 레이어에서 양방향으로 (fw-1)/2 씩 확장됩니다.
    return rf


def _pad_sequence_reflect(x: np.ndarray, pad: int) -> np.ndarray:
    """
    시간축 반사 패딩. (T,17,2) → (T+2*pad,17,2)
    """
    if pad <= 0:
        return x
    T = x.shape[0]
    if T == 0:
        return x
    
    # 파이토치 reflect 패딩과 유사하게 동작하도록 구현
    # numpy의 flip은 T < pad 일 때 문제가 있을 수 있으므로,
    # 필요 길이만큼 반복된 슬라이스를 이용해 더 안전하게 구현합니다.
    if T < pad: # 시퀀스 길이가 패딩 길이보다 짧을 경우
        # 처음 요소를 pad번 반복 후 뒤집기 (left_pad용)
        left = np.flip(np.repeat(x[0:1], pad, axis=0), axis=0)
        # 마지막 요소를 pad번 반복 후 뒤집기 (right_pad용)
        right = np.flip(np.repeat(x[-1:], pad, axis=0), axis=0)
    else: # 시퀀스 길이가 충분한 경우
        left = np.flip(x[1:pad+1], axis=0) # 첫 프레임을 제외한 앞쪽 pad개 (1~pad)를 뒤집음
        right = np.flip(x[-(pad+1):-1], axis=0) # 마지막 프레임을 제외한 뒤쪽 pad개 (T-pad-1~T-2)를 뒤집음

    return np.concatenate([left, x, right], axis=0)


def _ensure_min_time_len(X: np.ndarray, need_len: int) -> np.ndarray:
    """
    시간축 길이가 need_len 미만이면 마지막 프레임 반복으로 확장.
    """
    if X.shape[0] >= need_len:
        return X
    lack = need_len - X.shape[0]
    tail = np.repeat(X[-1:, :, :], repeats=lack, axis=0) # 마지막 프레임 반복
    return np.concatenate([X, tail], axis=0)


def _load_temporal_model(
    ckpt_path: str,
    filter_widths: List[int],
    causal: bool,
    device: str
) -> TemporalModel:
    """
    체크포인트 구조와 동일한 모델을 생성해 strict=True로 로딩합니다.
    """
    if not os.path.exists(ckpt_path):
        raise Pose3DError(f"체크포인트 파일을 찾을 수 없습니다: {ckpt_path}")
    
    # VideoPose3D 모델은 입력 채널(2D) 2개, 출력 채널(3D) 3개. num_joints는 17.
    model = TemporalModel(num_joints_in=17, in_features=2, num_joints_out=17, 
                          filter_widths=filter_widths, causal=causal,
                          out_features=3, # 3D 출력이므로
                          dropout=0.25) # 기본 드롭아웃 설정. 학습 시 사용된 것과 동일하게.
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_pos"], strict=True)
    except Exception as e:
        raise Pose3DError(f"체크포인트 로드 실패: {e}")
    return model.to(device).eval()


def _infer_sequence_center_sliding(
    model: TemporalModel,
    norm_2d: np.ndarray,
    RF: int,
    device: str,
    min_sequence_length: int = 81 # 안정적인 추론을 위한 최소 시퀀스 길이 (RF보다 크거나 같아야 함)
) -> np.ndarray:
    """
    센터-슬라이딩 추론.
    - 내부 스택의 유효 커널 폭이 커도 안전하도록, 윈도 길이 하한(min_sequence_length)을 강제 보장.
    - 끝단에서는 Xt를 필요 길이만큼 즉시 확장(tail repeat)하여 부족분 보강.
    """
    pad = (RF - 1) // 2 # receptive field의 절반 (중심 프레임 기준 좌우 패딩)

    # 1) 입력 시퀀스 자체 길이 보강: min_sequence_length 이상으로 확장
    T_in = int(norm_2d.shape[0])
    if T_in < min_sequence_length:
        lack = min_sequence_length - T_in
        ext_tail = np.repeat(norm_2d[-1:, :, :], repeats=lack, axis=0)
        norm_2d_safe = np.concatenate([norm_2d, ext_tail], axis=0)
    else:
        norm_2d_safe = norm_2d.copy() # 필요 길이 충분

    T_safe = int(norm_2d_safe.shape[0])
    output_3d = np.zeros((T_safe, 17, 3), dtype=np.float32)

    with torch.no_grad():
        for t_idx in range(T_safe):
            # 윈도우 시작/끝 인덱스 계산 (중심 프레임 t_idx 기준)
            start_idx = t_idx - pad
            end_idx = t_idx + pad + 1 # exclusive

            # 윈도우 추출
            # 해당 윈도우 영역이 norm_2d_safe 범위를 벗어날 수 있음.
            # 이 부분은 _pad_sequence_reflect와 유사하게 처리하여 채움.
            
            # 윈도우 추출 및 패딩
            current_window_np = np.zeros((RF, 17, 2), dtype=np.float32)
            
            # 유효한 부분만 복사
            copy_start_in_safe = max(0, start_idx)
            copy_end_in_safe = min(T_safe, end_idx)
            
            window_start_in_current_window = max(0, -start_idx)
            window_end_in_current_window = window_start_in_current_window + (copy_end_in_safe - copy_start_in_safe)
            
            current_window_np[window_start_in_current_window:window_end_in_current_window] = \
                norm_2d_safe[copy_start_in_safe:copy_end_in_safe]

            # 패딩 부분 처리 (reflection padding)
            # 윈도우의 시작 부분이 시퀀스 시작보다 앞일 때
            if start_idx < 0:
                pad_len = -start_idx
                if T_safe < pad_len: # 시퀀스가 패딩 길이보다 짧을 때
                     # 첫 프레임 반복 후 뒤집는 방식으로 보강
                    current_window_np[0:pad_len] = np.flip(np.repeat(norm_2d_safe[0:1], pad_len, axis=0), axis=0)
                else: # 충분한 길이일 때
                    # norm_2d_safe의 앞부분을 reflect 패딩
                    current_window_np[0:pad_len] = np.flip(norm_2d_safe[1:pad_len+1], axis=0)


            # 윈도우의 끝 부분이 시퀀스 끝보다 뒤일 때
            if end_idx > T_safe:
                pad_len = end_idx - T_safe
                if T_safe < pad_len: # 시퀀스가 패딩 길이보다 짧을 때
                    # 마지막 프레임 반복 후 뒤집는 방식으로 보강
                    current_window_np[RF-pad_len:RF] = np.flip(np.repeat(norm_2d_safe[-1:], pad_len, axis=0), axis=0)
                else: # 충분한 길이일 때
                    # norm_2d_safe의 뒷부분을 reflect 패딩
                    current_window_np[RF-pad_len:RF] = np.flip(norm_2d_safe[-(pad_len+1):-1], axis=0)


            # PyTorch 텐서로 변환 및 모델 입력 준비
            input_2d_tensor = torch.from_numpy(current_window_np).unsqueeze(0).to(device) # (1, RF, 17, 2)
            
            # 추론
            predicted_3d_tensor = model(input_2d_tensor) # (1, RF, 17, 3)
            
            # 중심 프레임 결과만 저장
            output_3d[t_idx] = predicted_3d_tensor.squeeze(0)[pad].cpu().numpy()

    # 원본 시퀀스 길이에 맞춰 결과 잘라내기 (확장했던 부분이 있다면 제거)
    return output_3d[:T_in]


def predict_3d_poses(
    kps_2d_seq: np.ndarray,
    ckpt_path: str,
    filter_widths: List[int] = None,
    causal: bool = False,
    device: Optional[str] = None
) -> np.ndarray:
    """
    2D 키포인트 시퀀스로부터 3D 포즈 시퀀스를 추정하고 루트 정규화를 수행합니다.

    Args:
        kps_2d_seq (np.ndarray): 입력 2D 키포인트 시퀀스. (T, 17, 2) 형태.
        ckpt_path (str): VideoPose3D 모델 체크포인트(.pth) 파일 경로.
        filter_widths (List[int], optional): 모델의 Temporal Convolution 필터 너비 리스트.
                                             VideoPose3D의 기본값 [3,3,3,3,3]이 권장됩니다.
        causal (bool, optional): 모델이 Causal 모드인지 여부. 기본값 False (Non-causal).
        device (str, optional): 모델을 로드할 장치 ('cuda' 또는 'cpu'). 기본값은 'cuda' 우선.

    Returns:
        np.ndarray: 루트 정규화된 추정 3D 포즈 시퀀스. (T, 17, 3) 형태.
                    골반(Pelvis)이 (0,0,0)에 위치합니다.

    Raises:
        Pose3DError: PyTorch나 VideoPose3D 모듈 임포트 실패, 체크포인트 로드 실패 등.
    """
    if filter_widths is None:
        filter_widths = [3, 3, 3, 3, 3] # VideoPose3D 기본 설정
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if kps_2d_seq.ndim != 3 or kps_2d_seq.shape[1] != 17 or kps_2d_seq.shape[2] != 2:
        raise ValueError("kps_2d_seq는 (T, 17, 2) 형태의 numpy 배열이어야 합니다.")
    if kps_2d_seq.shape[0] == 0:
        return np.empty((0, 17, 3), dtype=np.float32)

    # 1. 2D 키포인트 시퀀스 정규화 (골반 중심 이동, 스케일 정규화)
    norm_2d_seq = _normalize_2d_sequence(kps_2d_seq)

    # 2. TemporalModel 로드
    model = _load_temporal_model(ckpt_path, filter_widths, causal, device)

    # 3. 모델 수용장(receptive field) 계산
    RF = _receptive_field(filter_widths)
    # 최소 시퀀스 길이: RF 값, 또는 안전하게 RF보다 큰 값으로 설정. 
    # V-Pose3D에서 자주 사용되는 윈도우 사이즈인 81을 사용하는 경우도 많습니다.
    # conv1d가 에러나지 않도록 RF보다 크게 잡아주는 것이 중요합니다.
    min_infer_len = max(RF, 81) 

    # 4. 슬라이딩 윈도우 방식으로 3D 포즈 추정
    predicted_3d_seq_raw = _infer_sequence_center_sliding(model, norm_2d_seq, RF, device, min_infer_len)

    # 5. 3D 포즈 루트 조인트 정규화 (골반을 원점으로)
    # 추정된 3D 포즈 시퀀스에서 각 프레임의 골반 중심을 찾아서 모든 조인트에서 뺍니다.
    predicted_3d_seq_normalized = predicted_3d_seq_raw.copy()
    for t in range(predicted_3d_seq_raw.shape[0]):
        pelvis_3d = _pelvis_center_3d(predicted_3d_seq_raw[t])
        predicted_3d_seq_normalized[t] -= pelvis_3d
        
    return predicted_3d_seq_normalized

# --- 사용 예시 (이 코드를 직접 실행하는 경우) ---
if __name__ == '__main__':
    print("VideoPose3D 3D 포즈 추정 유틸리티 스크립트입니다.")
    print("주요 함수: predict_3d_poses(kps_2d_seq, ckpt_path, ...)")
    print("\n--- 데모 데이터로 실행 예시 ---")

    # 1. 가상의 2D 키포인트 데이터 생성 (T, 17, 2)
    # 실제 데이터는 OpenPose, AlphaPose 등으로 추출된 2D 키포인트가 됩니다.
    num_frames = 100
    dummy_kps_2d = np.random.rand(num_frames, 17, 2) * 200 + 50 # 대략적인 이미지 좌표 (50~250)

    # 2. VideoPose3D 체크포인트 파일 경로 설정
    # 이 부분은 사용자 환경에 맞게 수정해야 합니다.
    # 예: 'checkpoints/epoch_448.pth' (VideoPose3D 기본 모델)
    # ckpt_file = "path/to/your/videopose3d_checkpoint.pth" 
    ckpt_file = "/content/VideoPose3D/checkpoint/pretrained_h36m_detectron_coco.bin" # 예시 경로

    if not os.path.exists(ckpt_file):
        print(f"\n경고: 체크포인트 파일 '{ckpt_file}'을 찾을 수 없습니다.")
        print("실제 사용을 위해서는 VideoPose3D 체크포인트 파일을 다운로드하여 경로를 수정해주세요.")
        print("예: wget https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin -P /content/VideoPose3D/checkpoint/")
        print("데모 실행을 건너뜁니다.")
    else:
        print(f"\n2D 키포인트 시퀀스 shape: {dummy_kps_2d.shape}")
        print(f"체크포인트 파일: {ckpt_file}")

        try:
            # 3. 3D 포즈 추정
            print("\n3D 포즈 추정 시작...")
            estimated_3d_poses = predict_3d_poses(dummy_kps_2d, ckpt_file)
            print("3D 포즈 추정 완료!")

            # 4. 결과 확인
            print(f"추정된 3D 포즈 시퀀스 shape: {estimated_3d_poses.shape}")
            print("\n첫 5개 프레임의 첫 조인트 3D 좌표:")
            print(estimated_3d_poses[:5, 0, :])
            print("\n루트 정규화 확인 (골반(L_HIP, R_HIP)의 중심이 거의 (0,0,0)에 가까운지):")
            # 임의 프레임의 골반 중심 확인 (idx 11, 12)
            pelvis_center_frame_0 = (estimated_3d_poses[0, 11] + estimated_3d_poses[0, 12]) / 2
            print(f"프레임 0의 골반 중심: {pelvis_center_frame_0}")
            
            # 5. 전문가 파일 생성 예시
            expert_filename = "expert_pose_data.npz"
            np.savez_compressed(expert_filename, poses_3d=estimated_3d_poses)
            print(f"\n전문가 3D 포즈 데이터가 '{expert_filename}'으로 저장되었습니다.")

            # 6. 전문가 파일 로드 예시 및 비교 프로세스 안내
            print(f"\n--- 전문가 파일 활용 안내 ---")
            print(f"저장된 '{expert_filename}' 파일은 전문가의 3D 포즈 시퀀스를 담고 있습니다.")
            print(f"사용자 영상 비교 시 이 파일을 로드하여 비교 분석을 수행하시면 됩니다.")
            print(f"예시 로드 코드:\nloaded_expert_poses = np.load('{expert_filename}')['poses_3d']")
            print(f"이제 'loaded_expert_poses'와 '사용자 영상에서 추출된 3D 포즈'를 비교하여 피드백을 생성할 수 있습니다.")

        except Pose3DError as e:
            print(f"\n오류 발생: {e}")
        except Exception as e:
            print(f"\n예상치 못한 오류 발생: {e}")
