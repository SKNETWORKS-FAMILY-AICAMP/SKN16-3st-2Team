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


def _get_receptive_field_size(filter_widths: List[int], causal: bool) -> int:
    """
    VideoPose3D 모델의 실제 Receptive Field 크기를 계산합니다.
    filter_widths=[3,3,3,3,3] (H36M Pretrained 모델의 기본값) 기준:
    Non-causal (causal=False): RF = 243
    Causal (causal=True): RF = 73
    """
    # VideoPose3D 논문에 기반한 값
    if filter_widths == [3, 3, 3, 3, 3]:
        return 73 if causal else 243
    # 그 외의 filter_widths에 대해서는 보수적인 추정치를 반환합니다.
    # 더 정확한 계산이 필요하면 common/model.py 의 _calculate_receptive_field 함수를 참조해야 합니다.
    # 여기서는 안전한 값을 위해 최대치를 가정합니다.
    rf = 1
    for fw in filter_widths:
        # non-causal인 경우 양쪽으로 (fw-1) * dilation 이 영향을 미치므로 2를 곱함
        # 각 레이어의 dilation은 1, 3, 9, ... 등으로 가정 (3의 거듭제곱)
        rf += (fw - 1) * 2 * (3 ** (filter_widths.index(fw))) # 단순 추정. 정확한 구현은 복잡.
    return max(rf, 243) # 최소 243으로 보장하여 오류 회피


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
    # 시퀀스 길이가 패딩 + 1보다 짧으면 전체를 반복해서 뒤집는 방식으로.
    if T <= pad: # 예를 들어 T=1, pad=5 이면 x[1:pad+1]과 x[-(pad+1):-1]은 오류
        left = np.flip(np.repeat(x[0:1], pad, axis=0), axis=0) # 첫 프레임을 pad번 반복 후 뒤집기
        right = np.flip(np.repeat(x[-1:], pad, axis=0), axis=0) # 마지막 프레임을 pad번 반복 후 뒤집기
    else: # 시퀀스 길이가 충분한 경우
        # x[0]을 기준으로 그 다음 pad개를 뒤집음
        left = np.flip(x[1 : pad + 1], axis=0)
        # x[-1]을 기준으로 그 앞 pad개를 뒤집음
        right = np.flip(x[-(pad + 1) : -1], axis=0)
    
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
    
    model = TemporalModel(num_joints_in=17, in_features=2, num_joints_out=17, 
                          filter_widths=filter_widths, causal=causal,
                          out_features=3, 
                          dropout=0.25)
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_pos"], strict=True)
    except Exception as e:
        raise Pose3DError(f"체크포인트 로드 실패: {e}")
    return model.to(device).eval()


def predict_3d_poses(
    kps_2d_seq: np.ndarray,
    ckpt_path: str,
    filter_widths: List[int] = None,
    causal: bool = False,
    device: Optional[str] = None
) -> np.ndarray:
    """
    2D 키포인트 시퀀스로부터 3D 포즈 시퀀스를 추정하고 루트 정규화를 수행합니다.
    짧은 시퀀스에 대해서는 전체 시퀀스를 단일 윈도우로 확장하여 추론합니다.

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

    original_num_frames = kps_2d_seq.shape[0]

    # 1. 2D 키포인트 시퀀스 정규화 (골반 중심 이동, 스케일 정규화)
    norm_2d_seq = _normalize_2d_sequence(kps_2d_seq)

    # 2. TemporalModel 로드
    model = _load_temporal_model(ckpt_path, filter_widths, causal, device)

    # 3. 모델의 Receptive Field (RF) 크기 가져오기
    # 이 RF는 모델이 안정적으로 추론하기 위한 최소 윈도우 길이입니다.
    required_rf = _get_receptive_field_size(filter_widths, causal)
    
    # 4. 3D 포즈 추정 로직 (짧은 시퀀스 처리 강화)
    predicted_3d_seq_raw = np.zeros((original_num_frames, 17, 3), dtype=np.float32)

    with torch.no_grad():
        if original_num_frames < required_rf:
            # 시퀀스가 RF보다 짧을 경우: 전체 시퀀스를 RF 길이로 확장하고 단일 윈도우로 추론
            print(f"경고: 입력 시퀀스 길이({original_num_frames})가 모델 RF({required_rf})보다 짧습니다. 단일 윈도우 확장 추론을 수행합니다.")
            
            # 시퀀스를 required_rf 길이까지 확장 (마지막 프레임 반복)
            extended_norm_2d_seq = _ensure_min_time_len(norm_2d_seq, required_rf)
            
            # 확장된 시퀀스에 대해 반사 패딩 적용
            # 이 단일 윈도우는 자체가 RF 크기이므로 별도의 window_pad 계산 없이 그대로 전달합니다.
            # _pad_sequence_reflect는 여기서는 필요하지 않습니다. (아래 모델 입력 형성 참고)

            # 모델 입력 텐서 형성: (1, required_rf, 17, 2)
            input_2d_tensor = torch.from_numpy(extended_norm_2d_seq).unsqueeze(0).to(device)
            
            # 추론
            predicted_3d_tensor = model(input_2d_tensor) # (1, required_rf, 17, 3)
            
            # 추론된 시퀀스에서 원래 길이만큼만 가져옵니다.
            # 이 경우, 모든 프레임이 동일한 (확장된) 컨텍스트로 추론되었음을 의미합니다.
            predicted_3d_seq_raw = predicted_3d_tensor.squeeze(0)[:original_num_frames].cpu().numpy()

        else:
            # 시퀀스 길이가 RF보다 길거나 같을 경우: 슬라이딩 윈도우 추론
            
            # RF의 절반만큼 패딩 (중심 프레임을 위한)
            window_pad = (required_rf - 1) // 2
            
            # 전체 시퀀스에 대해 반사 패딩을 미리 적용합니다.
            padded_norm_2d_seq = _pad_sequence_reflect(norm_2d_seq, window_pad)
            
            for t_idx in range(original_num_frames):
                # padded_norm_2d_seq에서 t_idx를 중심으로 RF 크기의 윈도우를 추출
                current_window_np = padded_norm_2d_seq[t_idx : t_idx + required_rf]
                
                # PyTorch 텐서로 변환 및 모델 입력 준비: (1, required_rf, 17, 2)
                input_2d_tensor = torch.from_numpy(current_window_np).unsqueeze(0).to(device) 
                
                # 추론
                predicted_3d_tensor = model(input_2d_tensor) # (1, required_rf, 17, 3)
                
                # 중심 프레임 결과만 저장
                predicted_3d_seq_raw[t_idx] = predicted_3d_tensor.squeeze(0)[window_pad].cpu().numpy()

    # 5. 3D 포즈 루트 조인트 정규화 (골반을 원점으로)
    predicted_3d_seq_normalized = predicted_3d_seq_raw.copy()
    for t in range(original_num_frames):
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
    # 짧은 시퀀스 테스트 (이전 에러 상황 재현을 위한)
    num_frames_short = 50 # 243보다 훨씬 짧은 시퀀스로 테스트
    dummy_kps_2d_short = np.random.rand(num_frames_short, 17, 2) * 200 + 50 # 대략적인 이미지 좌표 (50~250)

    # 긴 시퀀스 테스트
    num_frames_long = 300 # 243보다 긴 시퀀스로 테스트
    dummy_kps_2d_long = np.random.rand(num_frames_long, 17, 2) * 200 + 50 # 대략적인 이미지 좌표 (50~250)


    # 2. VideoPose3D 체크포인트 파일 경로 설정
    # 이 부분은 사용자 환경에 맞게 수정해야 합니다.
    ckpt_file = "/content/VideoPose3D/checkpoint/pretrained_h36m_detectron_coco.bin" # 예시 경로
    # VideoPose3D 레포가 /content 에 클론되어 있고, 체크포인트가 그 아래 있다고 가정.

    if not os.path.exists(ckpt_file):
        print(f"\n경고: 체크포인트 파일 '{ckpt_file}'을 찾을 수 없습니다.")
        print("실제 사용을 위해서는 VideoPose3D 체크포인트 파일을 다운로드하여 경로를 수정해주세요.")
        print("예: wget https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin -P /content/VideoPose3D/checkpoint/")
        print("데모 실행을 건너뜁니다.")
    else:
        print(f"\n[짧은 시퀀스 테스트: {num_frames_short} 프레임]")
        print(f"2D 키포인트 시퀀스 shape: {dummy_kps_2d_short.shape}")
        try:
            print("\n3D 포즈 추정 시작...")
            estimated_3d_poses_short = predict_3d_poses(dummy_kps_2d_short, ckpt_file)
            print(f"3D 포즈 추정 완료! Shape: {estimated_3d_poses_short.shape}")
            # 5. 전문가 파일 생성 예시
            expert_filename_short = "expert_pose_data_short.npz"
            np.savez_compressed(expert_filename_short, poses_3d=estimated_3d_poses_short)
            print(f"전문가 3D 포즈 데이터가 '{expert_filename_short}'으로 저장되었습니다.")

        except Pose3DError as e:
            print(f"\n짧은 시퀀스에서 오류 발생: {e}")
        except Exception as e:
            print(f"\n짧은 시퀀스에서 예상치 못한 오류 발생: {e}")
            
        print(f"\n{'-'*50}\n")

        print(f"\n[긴 시퀀스 테스트: {num_frames_long} 프레임]")
        print(f"2D 키포인트 시퀀스 shape: {dummy_kps_2d_long.shape}")
        try:
            print("\n3D 포즈 추정 시작...")
            estimated_3d_poses_long = predict_3d_poses(dummy_kps_2d_long, ckpt_file)
            print(f"3D 포즈 추정 완료! Shape: {estimated_3d_poses_long.shape}")
            # 5. 전문가 파일 생성 예시
            expert_filename_long = "expert_pose_data_long.npz"
            np.savez_compressed(expert_filename_long, poses_3d=estimated_3d_poses_long)
            print(f"전문가 3D 포즈 데이터가 '{expert_filename_long}'으로 저장되었습니다.")

        except Pose3DError as e:
            print(f"\n긴 시퀀스에서 오류 발생: {e}")
        except Exception as e:
            print(f"\n긴 시퀀스에서 예상치 못한 오류 발생: {e}")
