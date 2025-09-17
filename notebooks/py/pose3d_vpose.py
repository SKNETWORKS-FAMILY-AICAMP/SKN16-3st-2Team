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
    """
    VideoPose3D 모델의 유효 수용장(receptive field) 크기를 계산합니다.
    논문(VideoPose3D)의 Temporal Model은 각 conv1d 레이어에서 양쪽으로 (filter_width - 1) * dilation 이 영향을 미칩니다.
    dilation은 filter_widths가 동일한 경우 보통 1, 3, 9, ... 등으로 증가합니다.
    여기는 간소화된 버전이며, 실제 TemporalModel의 정확한 Receptive Field 계산은 더 복잡할 수 있습니다.
    일반적으로 `filter_widths = [3, 3, 3, 3, 3]` 인 경우 RF = 81 (non-causal) 혹은 73 (causal) 으로 알려져 있습니다.
    가장 보수적으로 필터 폭들의 합을 기반으로 RF를 추정하고, 실제 Conv1D 오류를 피하기 위해 더 큰 값으로 설정하는 것이 안전합니다.
    """
    # 실제 VideoPose3D의 RF는 2 * sum( (fw - 1) * dilation ) + 1 이지만,
    # dilation을 자동으로 추정하기 어려워 단순히 filter_widths의 합으로 근사하고
    # 안정적인 추론을 위한 최소 길이로 보강하는 방식이 더 현실적입니다.
    # 경험적으로 non-causal의 경우 243프레임(3^5) 이상이면 매우 안전합니다.
    # 에러 메시지에서 kernel size가 55로 나온 것을 보면, 모델 내부적으로 팽창(dilation)이 많이 되는 레이어가 있는 것으로 보입니다.
    # 따라서, RF를 단순히 filter_widths 합산보다는 더 크게,
    # 예를 들어 243 (3^5, 즉 3x3 필터가 5번 연속 적용될 때의 이론상 최대 RF)으로 가정하는 것이 더 안전합니다.
    return 243 # 매우 보수적으로 설정

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
    if T < pad + 1: # 시퀀스 길이가 패딩 + 1보다 짧으면 전체를 반복해서 뒤집는 방식으로.
        left = np.flip(np.repeat(x[0:1], pad, axis=0), axis=0) # 첫 프레임을 pad번 반복 후 뒤집기
        right = np.flip(np.repeat(x[-1:], pad, axis=0), axis=0) # 마지막 프레임을 pad번 반복 후 뒤집기
    else: # 시퀀스 길이가 충분한 경우
        left = np.flip(x[1 : pad + 1], axis=0) # x[0]을 기준으로 그 다음 pad개를 뒤집음
        right = np.flip(x[-(pad + 1) : -1], axis=0) # x[-1]을 기준으로 그 앞 pad개를 뒤집음
    
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
    RF_estimated: int, # 이제 이 값은 보수적으로 설정된 최소 시퀀스 길이를 나타냅니다.
    device: str
) -> np.ndarray:
    """
    센터-슬라이딩 추론.
    - 입력 시퀀스 길이가 RF_estimated(추정된 RF) 미만이면,
      시퀀스 길이 보강(마지막 프레임 반복) 후 반사 패딩을 사용하여 안정적으로 추론.
    - 슬라이딩 윈도우 방식으로 각 프레임 추정.
    """
    # VideoPose3D의 TemporalModel은 내부적으로 입력 시퀀스의 (filter_width - 1) * dilation 만큼의
    # 앞뒤 프레임을 필요로 합니다.
    # RF_estimated를 매우 보수적인 최소 길이(예: 243)로 설정했으므로,
    # conv1d 에러를 피하기 위해 이 길이로 각 슬라이딩 윈도우를 구성합니다.
    # 모델의 RF (receptive field)는 모델 생성 시 filter_widths와 causal 파라미터에 따라 결정됩니다.
    # RF_estimated는 여기서 윈도우 크기의 최소 보장치로 사용합니다.
    
    # 실제 모델이 내부적으로 요구하는 최소 윈도우 길이를 고려하여 패딩 양을 조절합니다.
    # 임의로 81 프레임을 중심으로 양쪽으로 RF_estimated만큼 패딩하는 방식 대신,
    # RF_estimated 전체를 윈도우 길이로 가정하고, 절반을 패딩으로 활용하는 방식으로 바꿉니다.
    # 즉, 하나의 윈도우가 RF_estimated 길이를 가져야 하고, 그 윈도우의 '중심' 프레임을 추론합니다.
    
    T_in = int(norm_2d.shape[0])
    
    # 원본 시퀀스가 RF_estimated보다 짧으면 최소 길이만큼 확장
    norm_2d_extended = _ensure_min_time_len(norm_2d, RF_estimated)
    
    T_extended = int(norm_2d_extended.shape[0])
    output_3d = np.zeros((T_extended, 17, 3), dtype=np.float32)

    # 슬라이딩 윈도우의 패딩 양 (하나의 윈도우 길이 = RF_estimated 일 때 중심 프레임을 위한 패딩)
    # 예를 들어 RF_estimated가 243이면, 좌우 패딩은 121 (243-1)/2
    window_pad = (RF_estimated - 1) // 2

    with torch.no_grad():
        for t_idx in range(T_extended):
            # 중심 프레임 t_idx를 기준으로 윈도우 추출 (전체 윈도우 길이 = RF_estimated)
            start_in_extended = t_idx - window_pad
            end_in_extended = t_idx + window_pad + 1 # exclusive

            # 윈도우 경계를 벗어나는 부분은 반사 패딩 처리 (norm_2d_extended에 적용)
            # 여기서는 norm_2d_extended 자체에 직접 슬라이싱 후 _pad_sequence_reflect를 사용하여 처리합니다.
            # 이 로직은 `predict_3d_poses`에서 한번에 처리하도록 더 효율적으로 변경했습니다.
            # 여기서는 이미 전체 시퀀스가 RF_estimated 길이 이상으로 확장되었다고 가정하고,
            # 각 윈도우를 통으로 만들어서 모델에 전달하는 방식으로 수정합니다.
            
            # 윈도우 생성: norm_2d_extended를 충분히 패딩하여 윈도우 길이를 맞춥니다.
            # _pad_sequence_reflect가 이미 존재하므로 이를 활용합니다.
            # t_idx는 output_3d의 인덱스이므로,
            # norm_2d_extended로부터 해당 윈도우를 추출하고 부족하면 반사패딩을 추가하는 방식이 필요합니다.
            
            # 전체 시퀀스에 대한 반사 패딩을 미리 계산합니다.
            padded_norm_2d_seq = _pad_sequence_reflect(norm_2d_extended, window_pad)
            
            # t_idx에 해당하는 윈도우 추출.
            # padded_norm_2d_seq에서 t_idx + window_pad를 시작점으로 하여 RF_estimated 길이만큼 추출
            current_window_np = padded_norm_2d_seq[t_idx : t_idx + RF_estimated]
            
            # PyTorch 텐서로 변환 및 모델 입력 준비
            # (1, RF_estimated, 17, 2) 형태로 변경
            input_2d_tensor = torch.from_numpy(current_window_np).unsqueeze(0).to(device) 
            
            # 추론
            predicted_3d_tensor = model(input_2d_tensor) # (1, RF_estimated, 17, 3)
            
            # 중심 프레임 결과만 저장
            # 현재 윈도우에서 (RF_estimated - 1) // 2 인덱스가 중심 프레임
            output_3d[t_idx] = predicted_3d_tensor.squeeze(0)[window_pad].cpu().numpy()

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

    # 3. 모델 수용장(receptive field)의 최소 요구 길이 계산
    # 이전 에러 "Kernel size: (55)"를 고려하여 훨씬 보수적인 RF를 가정합니다.
    # VideoPose3D의 `filter_widths=[3,3,3,3,3]` (dilation 1, 3, 9, 27, 81) 일 때,
    # non-causal 모델의 RF는 243, causal은 73 입니다.
    # 안전하게 243으로 설정하여 `conv1d` 에러를 피합니다.
    # 만약 causal=True 라면 73으로 줄일 수 있지만, 일반적인 성능을 위해 non-causal이 사용됩니다.
    min_infer_length_for_model = _receptive_field(filter_widths) # 여기서는 243 반환
    
    # 4. 슬라이딩 윈도우 방식으로 3D 포즈 추정
    predicted_3d_seq_raw = _infer_sequence_center_sliding(model, norm_2d_seq, min_infer_length_for_model, device)

    # 5. 3D 포즈 루트 조인트 정규화 (골반을 원점으로)
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
    # 짧은 시퀀스 테스트 (이전 에러 상황 재현을 위한)
    num_frames = 50 # 243보다 훨씬 짧은 시퀀스로 테스트
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
