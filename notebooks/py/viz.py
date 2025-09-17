import numpy as np
import math
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt

# 필요한 유틸은 compare_dtw에서 재사용
from compare_dtw import (
    L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI, L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK,
    wrist_center, shoulder_center, pelvis_center, joint_angle_3p, trunk_tilt_deg
)

# 스켈레톤 연결 정의
EDGES: List[Tuple[int, int]] = [
    (L_HIP, L_KNE), (L_KNE, L_ANK),
    (R_HIP, R_KNE), (R_KNE, R_ANK),
    (L_HIP, R_HIP),
    (L_SHO, R_SHO),
    (L_SHO, L_ELB), (L_ELB, L_WRI),
    (R_SHO, R_ELB), (R_ELB, R_WRI),
    (L_SHO, L_HIP), (R_SHO, R_HIP)
]

def plot_overview(segA: Dict[str, Any],
                  segB: Dict[str, Any],
                  poseA: np.ndarray,
                  poseB: np.ndarray,
                  figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    손목 z 시계열(전문가 A, 사용자 B)과 프레임 간 속도(norm)를 2행 플롯으로 표시합니다.
    segA/segB의 페이즈 범위를 반투명 영역으로 오버레이합니다.
    """
    T_A = int(poseA.shape[0]) if poseA is not None else 0
    T_B = int(poseB.shape[0]) if poseB is not None else 0
    if T_A == 0 or T_B == 0:
        raise ValueError("poseA/poseB가 비어 있습니다.")

    wcA = np.array([wrist_center(poseA[t])[2] for t in range(T_A)], dtype=np.float32)
    wcB = np.array([wrist_center(poseB[t])[2] for t in range(T_B)], dtype=np.float32)

    def step_speed(X: np.ndarray) -> np.ndarray:
        if X.shape[0] < 2:
            return np.zeros((0,), dtype=np.float32)
        d = np.diff(X, axis=0)
        return np.linalg.norm(d, axis=1)

    spA = step_speed(poseA)
    spB = step_speed(poseB)

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=False)
    ax0, ax1 = axes

    # 상단: 손목 z
    ax0.plot(wcA, label="A wrist z", color="blue")
    ax0.plot(wcB, label="B wrist z", color="orange", alpha=0.85)
    ax0.set_title("Wrist z overview")
    ax0.legend()
    for n, s, e in segA.get("phases", []):
        ax0.axvspan(s, e, color="blue", alpha=0.06)

    # 하단: step-wise speed
    ax1.plot(spA, label="A step speed", color="blue")
    ax1.plot(spB, label="B step speed", color="orange", alpha=0.85)
    ax1.set_title("Step-wise Speed Norm")
    ax1.legend()
    for n, s, e in segB.get("phases", []):
        ax1.axvspan(max(0, s-1), max(0, e-1), color="orange", alpha=0.06)  # diff로 1프레임 줄어든 것 보정

    plt.tight_layout()
    plt.show()

def match_from_path(path: List[Tuple[int, int]], iA: int) -> int:
    """
    DTW path에서 A 프레임 iA와 가장 가까운 매칭의 B 프레임 인덱스를 반환합니다.
    """
    if not path:
        raise ValueError("DTW path가 비어 있습니다.")
    arr = np.array(path, dtype=np.int32)
    idx = int(np.argmin(np.abs(arr[:, 0] - int(iA))))
    return int(arr[idx, 1])

def frame_metrics(P: np.ndarray) -> Dict[str, float]:
    """
    단일 프레임의 핵심 지표를 계산합니다.
    """
    m: Dict[str, float] = {
        "knee_L":  float(math.degrees(joint_angle_3p(P[L_HIP], P[L_KNE], P[L_ANK]))),
        "knee_R":  float(math.degrees(joint_angle_3p(P[R_HIP], P[R_KNE], P[R_ANK]))),
        "hip_L":   float(math.degrees(joint_angle_3p(shoulder_center(P), P[L_HIP], P[L_KNE]))),
        "hip_R":   float(math.degrees(joint_angle_3p(shoulder_center(P), P[R_HIP], P[R_KNE]))),
        "elbow_L": float(math.degrees(joint_angle_3p(P[L_SHO], P[L_ELB], P[L_WRI]))),
        "elbow_R": float(math.degrees(joint_angle_3p(P[R_SHO], P[R_ELB], P[R_WRI]))),
        "trunk_tilt": float(trunk_tilt_deg(P)),
    }
    wc = wrist_center(P)
    body_c = (pelvis_center(P) + shoulder_center(P)) / 2.0
    m["wrist_xy_dist"] = float(np.linalg.norm((wc - body_c)[:2]))
    return m

def representative_summary(seg_A: Dict[str, Any],
                           path: List[Tuple[int, int]],
                           poseA: np.ndarray,
                           poseB: np.ndarray,
                           keys: Tuple[str, ...] = ("second_pull_peak", "catch")) -> List[Dict[str, Any]]:
    """
    대표 이벤트 키(예: second_pull_peak, catch)에 대해 A/B 프레임을 매칭하고
    각 프레임의 지표 및 차이(B-A)를 계산하여 리스트로 반환합니다.
    """
    reps: List[Dict[str, Any]] = []
    marksA = seg_A.get("marks", {})
    for key in keys:
        if key not in marksA:
            continue
        iA = int(marksA[key])
        jB = match_from_path(path, iA)
        if iA < 0 or iA >= poseA.shape[0]:
            continue
        if jB < 0 or jB >= poseB.shape[0]:
            continue
        P3A = poseA[iA]
        P3B = poseB[jB]
        mA = frame_metrics(P3A)
        mB = frame_metrics(P3B)
        diff = {k: float(mB[k] - mA[k]) for k in mA.keys()}
        reps.append(dict(key=key, iA=iA, jB=jB, metricsA=mA, metricsB=mB, diff=diff))
    return reps

def plot_pose_3d(P: np.ndarray,
                 title: str = "",
                 lim: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = ((-1.2, 1.2), (-1.2, 1.2), (-0.6, 1.4)),
                 ax: Optional[Any] = None) -> Any:
    """
    단일 3D 포즈 프레임을 스캐터+엣지로 그립니다. ax가 없으면 새 Figure를 생성합니다.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (필요한 백엔드 준비)
    close_fig = False
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
        close_fig = True

    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c="r", s=20)
    for a, b in EDGES:
        ax.plot([P[a, 0], P[b, 0]], [P[a, 1], P[b, 1]], [P[a, 2], P[b, 2]], "b-", lw=1.5)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = lim
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.view_init(elev=15, azim=-70)

    if close_fig:
        plt.tight_layout()
        plt.show()
    return ax

def plot_representatives(seg_A: Dict[str, Any],
                         path: List[Tuple[int, int]],
                         poseA: np.ndarray,
                         poseB: np.ndarray,
                         keys: Tuple[str, ...] = ("second_pull_peak", "catch")) -> None:
    """
    대표 이벤트들에 대해 A/B 프레임을 나란히 시각화합니다.
    """
    for key in keys:
        if key not in seg_A.get("marks", {}):
            continue
        iA = int(seg_A["marks"][key])
        jB = match_from_path(path, iA)
        if iA < 0 or iA >= poseA.shape[0] or jB < 0 or jB >= poseB.shape[0]:
            continue

        P3A = poseA[iA]
        P3B = poseB[jB]

        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")

        plot_pose_3d(P3A, title=f"A - {key} (i={iA})", ax=ax1)
        plot_pose_3d(P3B, title=f"B - {key} (j={jB})", ax=ax2)

        plt.tight_layout()
        plt.show()